"""Record both loaded libraries' complete host-generated Down tiling bytes."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


ARMS = {
    "separate": ("ffn_original_shared_gu_static",
                 "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"),
    "fused": ("ffn_down_n512",
              "c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032"),
}


def capture(parent, candidate, output):
    import torch
    import torch_npu

    report = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "tiling_bytes": 200, "arms": {}}
    for arm, path in (("separate", parent), ("fused", candidate)):
        namespace, expected = ARMS[arm]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected frozen library: {path}")
        torch.ops.load_library(str(path))
        library = ctypes.CDLL(str(path))
        tiler = getattr(library, namespace + "_body_build_routed_tiling")
        tiler.argtypes = ([ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int64])
        tiler.restype = ctypes.c_int
        layouts = {}
        for layout in (0, 1):
            samples = []
            for _ in range(2):
                buffer = ctypes.create_string_buffer(200)
                status = tiler(64, 6144, 128, 6144, buffer, layout)
                if status:
                    raise RuntimeError(f"Host tiler rejected {arm}/{layout}: {status}")
                samples.append(buffer.raw)
            if samples[0] != samples[1]:
                raise RuntimeError("The host tiling contains unstable bytes")
            layouts[str(layout)] = {
                "hex": samples[0].hex(),
                "sha256": hashlib.sha256(samples[0]).hexdigest(),
            }
        report["arms"][arm] = {"path": str(path), "library_sha256": actual,
                               "symbol": namespace + "_body_build_routed_tiling",
                               "layouts": layouts}
    report["npu_initialized"] = torch.npu.is_initialized()
    if report["npu_initialized"]:
        raise RuntimeError("Host tiling capture initialized an NPU")
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    capture(args.parent.resolve(), args.candidate.resolve(), args.output.resolve())
