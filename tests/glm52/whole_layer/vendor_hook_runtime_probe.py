#!/usr/bin/env python3
"""Record driver enumeration and Torch binding in one process without vLLM."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback


ENVIRONMENT_KEYS = (
    "ASCEND_DOCKER_RUNTIME", "ASCEND_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH", "ASCEND_PROCESS_LOG_PATH",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PATH",
)
HASHED_LIBRARIES = (
    "libascend_hal.so", "libascend_base_hal.so", "libascendcl.so",
    "libruntime.so", "libtorch_npu.so",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_libraries():
    paths = sorted({
        fields[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(fields := line.split()) >= 6 and fields[-1].startswith("/")
        and any(part in fields[-1].lower()
                for part in ("ascend", "cann-", "torch_npu", "libruntime"))
    })
    hashes = {}
    for filename in paths:
        path = Path(filename)
        if any(path.name.startswith(name) for name in HASHED_LIBRARIES):
            hashes[filename] = sha256(path)
    return {"paths": paths, "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "report.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = {
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "pid": os.getpid(),
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "scope": "single process; no vLLM, model, allocations or NPU kernels",
        "stages": [],
    }

    def save():
        report["mapped_libraries"] = mapped_libraries()
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    def stage(name, action):
        item = {"name": name}
        try:
            item["result"] = action()
            item["success"] = True
        except Exception:
            item["success"] = False
            item["error"] = traceback.format_exc()
        report["stages"].append(item)
        save()
        print(json.dumps(item), flush=True)
        return item

    save()

    def enumerate_driver():
        library = ctypes.CDLL("libascend_hal.so")
        function = library.drvGetDevNum
        function.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        function.restype = ctypes.c_int
        count = ctypes.c_uint32(0)
        result = function(ctypes.byref(count))
        return {"return_code": result, "device_count": count.value}

    driver = stage("drvGetDevNum", enumerate_driver)

    def import_torch():
        import torch
        import torch_npu

        report["torch"] = torch.__version__
        report["torch_npu"] = torch_npu.__version__
        report["torch_file"] = torch.__file__
        report["torch_npu_file"] = torch_npu.__file__
        return {"npu_initialized": torch.npu.is_initialized()}

    imported = stage("torch_import", import_torch)
    if imported["success"]:
        import torch

        stage("torch_device_count", lambda: {
            "device_count": torch.npu.device_count(),
            "npu_initialized": torch.npu.is_initialized(),
        })

        def bind_device():
            torch.npu.set_device(0)
            return {"device_count": torch.npu.device_count(),
                    "device_name": torch.npu.get_device_name(0),
                    "npu_initialized": torch.npu.is_initialized()}

        bound = stage("torch_set_device", bind_device)
        report["npu_initialized_after"] = torch.npu.is_initialized()
    else:
        bound = {"success": False}
    report["success"] = bool(
        driver["success"] and driver["result"]["return_code"] == 0
        and driver["result"]["device_count"] == 1 and bound["success"]
    )
    save()
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
