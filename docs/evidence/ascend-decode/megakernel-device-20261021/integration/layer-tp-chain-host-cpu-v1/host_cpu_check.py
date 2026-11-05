"""Load the real TP host ABI and reject invalid calls without a mapped NPU."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

from host import Library, Window


def run(library_path, library_sha, local_path, output):
    import torch
    import torch_npu

    if (not torch.__version__.startswith("2.12.0")
            or torch_npu.__version__ != "2.12.0"
            or torch.npu.is_initialized()
            or list(Path("/dev").glob("davinci[0-9]*"))):
        raise RuntimeError("Use actual Torch212 with no mapped NPU")
    if hashlib.sha256(local_path.read_bytes()).hexdigest() != (
            "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f"):
        raise RuntimeError("The co-loaded local parent changed")
    local = ctypes.CDLL(str(local_path))
    library = Library(library_path, library_sha)
    cases = [
        ("invalid_device", lambda: library.initialize(-1, 0, 2, "127.0.0.1:1234", 1 << 28)),
        ("invalid_rank", lambda: library.initialize(0, 2, 2, "127.0.0.1:1234", 1 << 28)),
        ("single_rank", lambda: library.initialize(0, 0, 1, "127.0.0.1:1234", 1 << 28)),
        ("empty_endpoint", lambda: library.initialize(0, 0, 2, "", 1 << 28)),
        ("small_heap", lambda: library.initialize(0, 0, 2, "127.0.0.1:1234", 0)),
        ("invalid_capacity", lambda: Window(library, 0)),
        ("uninitialized_window", lambda: Window(library, 32)),
        ("uninitialized_finalize", library.finalize),
        ("unknown_window", lambda: library.check(library.handle.layer_tp_window_destroy(0))),
        ("invalid_submit", lambda: library.check(library.handle.layer_tp_submit(
            -1, 0, None, 0, None, 0, True, 0, 0, 0))),
    ]
    rejected = {}
    for name, operation in cases:
        try:
            operation()
        except RuntimeError as error:
            rejected[name] = str(error)
        else:
            raise RuntimeError(f"Invalid TP operation accepted: {name}")
        if torch.npu.is_initialized():
            raise RuntimeError(f"The invalid call initialized NPU: {name}")
    local.layer_local_config_bytes.restype = ctypes.c_size_t
    local.layer_local_address_bytes.restype = ctypes.c_size_t
    if (local.layer_local_config_bytes(), local.layer_local_address_bytes()) != (664, 928):
        raise RuntimeError("The loaded local parent ABI differs")
    report = {
        "library_sha256": library_sha, "parent_coload": True,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "negative_controls": rejected, "pass": True,
        "mapped_davinci_devices": [], "npu_initialized_before": False,
        "npu_initialized_after": False, "device_execution": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "negative_controls": len(rejected),
                      "npu_initialized": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.library.resolve(), args.library_sha256, args.local.resolve(), args.output)
