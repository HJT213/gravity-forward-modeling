"""Attach the offline Python runtime, application and weights to the Windows launcher."""
from argparse import ArgumentParser
from pathlib import Path
import os
import subprocess
import sys
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy

from make_icon import write_icon


HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
PROJECT = BACKEND.parent
BASE = Path(sys.base_prefix)
EXCLUDE = {"__pycache__", "test", "tests", "idlelib", "tkinter", "turtledemo",
           "site-packages", "ensurepip", "benchmarks", "include", "bin", "share"}
ASSETS = [
    "backend/server.py", "backend/model.py", "backend/prisms.py",
    "index.html", "app.js", "style.css", "density-3d.js",
    "vendor/three.min.js", "vendor/LICENSE", "vendor/LUCIDE-LICENSE",
]
WEIGHTS = ["9_分散棱柱", "11_阶梯棱柱", "12_单棱柱", "16_八字棱柱"]


def add_tree(archive, source, prefix, exclude=EXCLUDE):
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or any(p in exclude for p in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".pdb", ".lib", ".a", ".h", ".hpp"}:
            continue
        archive.write(path, (prefix / relative).as_posix())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--thesis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise RuntimeError("Requires Windows x64 Python 3.12.")
    if not (args.cpu / "torch/lib/torch_cpu.dll").is_file() or list((args.cpu / "torch/lib").glob("*cuda*")):
        raise RuntimeError("A CPU-only Windows x64 PyTorch installation is required.")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    icon = args.build_dir / "gravity-lab.ico"
    write_icon(icon)
    staged = args.output.with_name("bundle-" + uuid4().hex + ".exe")
    csc = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    subprocess.run([
        str(csc), "/nologo", "/target:winexe", "/platform:x64", "/optimize+", "/codepage:65001",
        "/out:" + str(staged), "/win32manifest:" + str(HERE / "app.manifest"),
        "/win32icon:" + str(icon),
        "/reference:System.Core.dll", "/reference:System.Drawing.dll",
        "/reference:System.IO.Compression.dll", "/reference:System.Windows.Forms.dll",
        str(HERE / "Launcher.cs"),
    ], check=True)
    weights = args.thesis / "毕设代码" / "最佳训练模型"
    with ZipFile(staged, "a", compression=ZIP_DEFLATED, compresslevel=3, allowZip64=False) as bundle:
        print("Bundling Python and CPU dependencies", flush=True)
        for filename in ("python.exe", "python312.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"):
            bundle.write(BASE / filename, filename)
        add_tree(bundle, BASE / "Lib", Path("Lib"))
        add_tree(bundle, BASE / "DLLs", Path("DLLs"))
        bundle.writestr("python312._pth", ".\nLib\nDLLs\npackages\napp/backend\n")
        add_tree(bundle, args.cpu, Path("packages"))
        site = Path(numpy.__file__).resolve().parent.parent
        add_tree(bundle, site / "numpy", Path("packages/numpy"))
        add_tree(bundle, site / "numpy.libs", Path("packages/numpy.libs"))
        for folder in site.glob("numpy-*.dist-info"):
            add_tree(bundle, folder, Path("packages") / folder.name)
        for asset in ASSETS:
            bundle.write(PROJECT / asset, "app/" + asset)
        for asset in sorted((BACKEND / "data").glob("*.npz")):
            bundle.write(asset, "app/backend/data/" + asset.name, compress_type=ZIP_STORED)
        for name in WEIGHTS:
            print("Bundling checkpoint:", name, flush=True)
            bundle.write(weights / name / "fourier_gravity_3d.pt",
                         "app/毕设代码/最佳训练模型/" + name + "/fourier_gravity_3d.pt",
                         compress_type=ZIP_STORED)
        bundle.writestr("bundle-id.txt", uuid4().hex)
    with ZipFile(staged) as bundle:
        assert all(name in bundle.namelist() for name in ("app/backend/server.py", "packages/torch/lib/torch_cpu.dll"))
    staged.replace(args.output)
    print("{} ({:,} bytes)".format(args.output.resolve(), args.output.stat().st_size), flush=True)


if __name__ == "__main__":
    main()
