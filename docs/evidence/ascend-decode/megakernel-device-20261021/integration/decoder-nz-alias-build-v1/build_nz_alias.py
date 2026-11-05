#!/usr/bin/env python3
"""Build an isolated storage-alias helper against the actual TorchNPU wheel."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("Build this host helper without any NPU devices")
    import torch
    import torch_npu

    if (torch.__version__, torch_npu.__version__) != ("2.12.0+cu130", "2.12.0"):
        raise RuntimeError("Use the recorded Torch/TorchNPU 2.12 wheel pair")
    if torch.npu.is_initialized():
        raise RuntimeError("The build must not initialize an NPU")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "nz_alias.cpp"
    shutil.copyfile(Path(__file__).with_name("nz_alias.cpp"), source)
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    compiler = os.environ.get("CXX", "g++")
    obj, dependencies = args.output / "nz_alias.o", args.output / "nz_alias.d"
    command = [compiler, "-std=c++17", "-O2", "-fPIC",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    for include in (torch_root / "include", torch_root / "include/torch/csrc/api/include",
                    npu_root / "include", args.cann / "include"):
        command.extend(["-I", str(include)])
    command.extend(["-MD", "-MF", str(dependencies), "-c", str(source), "-o", str(obj)])
    library = args.output / "nz_alias.so"
    link = [compiler, "-shared", str(obj), "-o", str(library)]
    for directory in (torch_root / "lib", npu_root / "lib"):
        link.extend(["-L", str(directory)])
    link.extend(["-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu"])
    report = {"scope": "Host StorageImpl alias helper; no operator arithmetic or CANN kernel changes",
              "image_id": args.image_id, "python": sys.version,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
              "source_sha256": digest(source), "script_sha256": digest(Path(__file__)),
              "commands": [command, link], "completed": False}
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    for label, argv in (("compile", command), ("link", link)):
        with (args.output / f"{label}.log").open("w") as output:
            subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, check=True)
    report["dependencies_sha256"] = {
        name: digest(Path(name))
        for name in dependencies.read_text().replace("\\\n", " ").split()[1:]
        if Path(name).is_file()
    }
    torch.ops.load_library(str(library))
    report["schemas"] = {name: str(getattr(torch.ops.tilert_nz_storage, name).default._schema)
                         for name in ("canonical_alias", "raw_storage", "describe")}
    report.update(completed=True, library_sha256=digest(library), object_sha256=digest(obj),
                  npu_initialized=torch.npu.is_initialized())
    if report["npu_initialized"]:
        raise RuntimeError("The CPU load initialized an NPU")
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "completed", "library_sha256", "npu_initialized", "schemas")}, indent=2))


if __name__ == "__main__":
    main()
