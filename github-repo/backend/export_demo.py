"""Run the four checkpoints once and export the results for GitHub Pages."""
import argparse
import json
from pathlib import Path

import torch

from server import Engine, PROJECT, SAMPLES, sample, sample_metadata


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis", type=Path, help="Optional original thesis folder containing the checkpoints")
    args = parser.parse_args()
    torch.set_num_threads(4)
    engine = Engine(args.thesis, "cpu")
    for key in SAMPLES:
        if not engine.checkpoint(key).is_file():
            raise FileNotFoundError(engine.checkpoint(key))

    output = PROJECT / "demo-data"
    output.mkdir(exist_ok=True)
    manifest = []
    for key, (name, _, description) in SAMPLES.items():
        fixture = sample(key)
        result = engine.run(key)
        density = fixture["density"].astype("<f4", copy=False)
        (output / (key + ".bin")).write_bytes(density.tobytes(order="C"))
        write_json(output / (key + ".json"), {"fixture": sample_metadata(fixture), "result": result})
        manifest.append({"id": key, "name": name, "description": description})
        print(key, result["metrics"], flush=True)

    write_json(output / "samples.json", manifest)


if __name__ == "__main__":
    main()
