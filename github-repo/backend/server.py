"""Serve the gravity workspace on loopback; no uploads or remote dependencies."""
import argparse
from base64 import b64encode
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback
from urllib.parse import urlsplit

import numpy as np
import torch

from model import load_model, predict
from prisms import build_model


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SAMPLES = {
    "figure8": ("八字棱柱", "16_八字棱柱", "斜向交错的棱柱组合"),
    "single": ("单棱柱", "12_单棱柱", "单一密度异常体"),
    "scattered": ("分散棱柱", "9_分散棱柱", "空间分离的多个异常体"),
    "staircase": ("阶梯棱柱", "11_阶梯棱柱", "埋深呈阶梯变化的密度结构"),
}
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/density-3d.js": ("density-3d.js", "text/javascript; charset=utf-8"),
          "/vendor/three.min.js": ("vendor/three.min.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}


@lru_cache(maxsize=len(SAMPLES))
def sample(key):
    with np.load(str(HERE / "data" / (key + ".npz")), allow_pickle=False) as pack:
        return {entry: pack[entry] for entry in pack.files}


def sample_metadata(fixture):
    return {
        "truth": fixture["truth"].tolist(), "bounds": fixture["bounds"].tolist(),
        "defaultLayer": int(fixture["default_layer"]),
        "densityLimit": float(np.max(np.abs(fixture["density"]))),
        "seed": int(fixture["seed"]), "trainingUsed": int(fixture["training_used"]),
    }


class Engine:
    def __init__(self, thesis, preference):
        self.thesis = thesis
        self.lock = threading.Lock()
        self.dev = torch.device("cuda" if preference != "cpu" and torch.cuda.is_available() else "cpu")
        self.key = None
        self.model = None

    def checkpoint(self, key):
        relative = Path(SAMPLES[key][1]) / "fourier_gravity_3d.pt"
        root = self.thesis / "毕设代码" / "最佳训练模型" if self.thesis else HERE / "weights"
        return root / relative

    def run(self, key, data=None):
        original = sample(key)
        data = original if data is None else data
        stats = original["stats"]
        with self.lock:
            try:
                if self.key != key:
                    self.model = None
                    self.key = None
                    self.model = load_model(self.checkpoint(key), self.dev)
                    self.key = key
                prediction, ms = predict(self.model, data["density"], stats, self.dev)
            except RuntimeError:
                if self.dev.type != "cuda":
                    raise
                traceback.print_exc()
                self.model, self.key = None, None
            else:
                return evaluate(prediction, data["truth"], ms, self.dev.type, False)

            # Run after leaving the except block so failed CUDA intermediates can be freed.
            torch.cuda.empty_cache()
            self.dev = torch.device("cpu")
            self.model = load_model(self.checkpoint(key), self.dev)
            self.key = key
            prediction, ms = predict(self.model, data["density"], stats, self.dev)
            return evaluate(prediction, data["truth"], ms, self.dev.type, True)


def evaluate(prediction, truth, ms, device, fallback):
    if prediction.shape != truth.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("Invalid output array")
    p, t = prediction.astype(np.float64), truth.astype(np.float64)
    difference = p - t
    mse = np.mean(difference ** 2)
    truth_power = np.mean(t ** 2)
    correlation = np.corrcoef(p.ravel(), t.ravel())[0, 1] if np.ptp(p) and np.ptp(t) else None
    return {
        "prediction": prediction.tolist(),
        "error": difference.tolist(),
        "device": device.upper(),
        "fallback": fallback,
        "metrics": {"ms": ms, "rmse": float(np.sqrt(mse)),
                    "rrmse": float(100 * np.sqrt(mse / truth_power)) if truth_power else None,
                    "corr": float(correlation) if correlation is not None and np.isfinite(correlation) else None},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "GravityLab/0.1"

    def reply(self, payload, mime, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def json_reply(self, item, status=HTTPStatus.OK):
        self.reply(json.dumps(item, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8", status)

    def custom_model(self, infer):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get_content_type() != "application/json" or not 0 < size <= 8192:
                raise ValueError("自定义参数须为有效的 JSON 数据。")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("自定义参数须为有效的 JSON 对象。")
            key = payload.get("model")
            if not isinstance(key, str) or key not in SAMPLES:
                raise ValueError("请选择有效的 FNO 权重。")
            data = build_model(payload.get("prisms"))
        except (ValueError, UnicodeError) as exc:
            message = "自定义参数不是有效的 JSON。" if isinstance(exc, (json.JSONDecodeError, UnicodeError)) else str(exc)
            return self.json_reply({"error": message}, HTTPStatus.BAD_REQUEST)

        ready = self.server.engine.checkpoint(key).is_file()
        if infer:
            if not ready:
                return self.json_reply({"error": "未找到所选 FNO 权重。"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return self.json_reply(self.server.engine.run(key, data))

        density = data.pop("density").astype("<f4", copy=False)
        data["density"] = b64encode(density.tobytes()).decode("ascii")
        data["truth"] = data["truth"].tolist()
        data["ready"] = ready
        return self.json_reply(data)

    def dispatch(self, method):
        hosts = {"{}:{}".format(host, self.server.server_port) for host in ("127.0.0.1", "localhost")}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts or (origin and origin not in {"http://" + h for h in hosts}):
            return self.json_reply({"error": "仅支持本机访问。"}, HTTPStatus.FORBIDDEN)
        path = urlsplit(self.path).path
        try:
            if method == "GET" and path == "/runtime.js":
                return self.reply(b"window.GRAVITY_LOCAL_INFERENCE = true;\n", "text/javascript; charset=utf-8")
            if method == "GET" and path in STATIC:
                filename, mime = STATIC[path]
                return self.reply((PROJECT / filename).read_bytes(), mime)
            if method == "GET" and path == "/api/samples":
                return self.json_reply([
                    {"id": k, "name": title, "description": desc, "ready": self.server.engine.checkpoint(k).is_file()}
                    for k, (title, _, desc) in SAMPLES.items()
                ])
            if method == "POST" and path in ("/api/custom/preview", "/api/custom/predict"):
                if origin not in {"http://" + host for host in hosts}:
                    return self.json_reply({"error": "仅允许本机页面提交自定义参数。"}, HTTPStatus.FORBIDDEN)
                return self.custom_model(infer=path.endswith("/predict"))
            items = path.split("/")
            if len(items) == 4 and items[1] == "api" and items[3] in SAMPLES:
                key = items[3]
                if method == "GET" and items[2] == "sample":
                    return self.json_reply(sample_metadata(sample(key)))
                if method == "GET" and items[2] == "density":
                    density = sample(key)["density"].astype("<f4", copy=False)
                    return self.reply(density.tobytes(order="C"), "application/octet-stream")
                if method == "POST" and items[2] == "predict":
                    if not self.server.engine.checkpoint(key).is_file():
                        return self.json_reply({"error": "未找到该样本对应的毕设权重。请检查毕设目录。"},
                                               HTTPStatus.SERVICE_UNAVAILABLE)
                    return self.json_reply(self.server.engine.run(key))
            return self.json_reply({"error": "找不到请求的资源。"}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            traceback.print_exc()
            return self.json_reply({"error": "计算或读取数据失败，请查看本地服务窗口。"},
                                   HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4175)
    parser.add_argument("--thesis", type=Path, help="Optional original thesis folder containing the checkpoints")
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.engine = Engine(args.thesis, args.device)
    print("Gravity Lab | http://127.0.0.1:{}".format(server.server_port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
