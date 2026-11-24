"""Reproduce the captured shared SwiGLU scale through the original operator."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


CAPTURE_SHA = "977b888e50c84f55d3dd358be8b25073741b9ebc5ad6823067b6b1920424bf52"
EXTENSION_SHA = "acac5e7220b51bdd24e41352ffc52f88aac5f58dd9d262b0a43aad2ca9efd0fc"
OPAPI_SHA = "3eebcd29e93860ea2aafaf9dfe269e98a28dc90c7cc325914e479be43bd247cf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(torch, actual, expected):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError("The original shared tensor contract changed")
    a = actual.contiguous().view(torch.uint8)
    b = expected.contiguous().view(torch.uint8)
    return {"bitwise_equal": torch.equal(a, b),
            "different_bytes": int((a != b).sum()),
            "different_elements": int((actual != expected).sum()),
            "sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "One original shared activation on captured M32 operands",
        "script_sha256": digest(Path(__file__)),
        "capture_sha256": digest(args.capture),
        "extension_sha256": digest(args.extension),
        "completed": False, "cpu_only": args.cpu_only, "call_count": 0,
        "timing": False,
    }

    def persist():
        (args.output / "dispatch.json").write_text(
            json.dumps(report, indent=2) + "\n")

    try:
        if report["capture_sha256"] != CAPTURE_SHA:
            raise ValueError("The retained shared diagnostic changed")
        if report["extension_sha256"] != EXTENSION_SHA:
            raise ValueError("The original operator binding changed")
        import torch
        import torch_npu

        torch.set_num_threads(1)
        captured = torch.load(args.capture, weights_only=True, map_location="cpu")
        reference = captured["reference"]
        shapes = {"gate_int32": ((32, 256), torch.int32),
                  "gate_scale": ((256,), torch.float32),
                  "input_scale": ((32,), torch.float32),
                  "middle": ((32, 128), torch.int8),
                  "middle_scale": ((32,), torch.float32)}
        for name, (shape, dtype) in shapes.items():
            if tuple(reference[name].shape) != shape or reference[name].dtype != dtype:
                raise ValueError(f"The captured {name} contract changed")
        torch.ops.load_library(str(args.extension))
        resolver = ctypes.CDLL(str(args.extension)).decoder_official_api_library
        resolver.argtypes = [ctypes.c_char_p]
        resolver.restype = ctypes.c_char_p
        providers = {}
        for symbol in ("aclnnDequantSwigluQuantV2",
                       "aclnnDequantSwigluQuantV2GetWorkspaceSize"):
            address = resolver(symbol.encode())
            if address is None:
                raise RuntimeError(f"The original operator did not resolve: {symbol}")
            provider = Path(address.decode()).resolve()
            if digest(provider) != OPAPI_SHA:
                raise RuntimeError(f"The original API provider changed: {provider}")
            providers[symbol] = {"path": str(provider), "sha256": OPAPI_SHA}
        operator = torch.ops._C_ascend.npu_dequant_swiglu_quant
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__,
                      providers=providers, schema=str(operator.default._schema),
                      npu_initialized_before=torch.npu.is_initialized())
        if args.cpu_only:
            if torch.npu.is_initialized():
                raise RuntimeError("The CPU preflight initialized the NPU")
            report["completed"] = True
            return
        if os.environ.get("ASCEND_GLOBAL_LOG_LEVEL") != "0":
            raise RuntimeError("Require command-local CANN DEBUG dispatch logs")
        torch.npu.set_device(0)
        if torch.npu.device_count() != 1:
            raise RuntimeError("Expose exactly one NPU")
        operands = {
            name: torch_npu.npu_format_cast(reference[name].to("npu"), 2)
            for name in ("gate_int32", "gate_scale", "input_scale")}
        for name, value in operands.items():
            if not compare(torch, value.cpu(), reference[name])["bitwise_equal"]:
                raise RuntimeError(f"Device transfer changed {name}")
        kwargs = {"bias": None, "quant_scale": None, "quant_offset": None,
                  "group_index": None, "activate_left": True,
                  "quant_mode": 1, "swiglu_mode": 1, "clamp_limit": 0.0}
        report["scalar_arguments"] = kwargs
        torch.npu.synchronize()
        report["call_start_unix_ns"] = time.time_ns()
        print("SHARED_ACTIVATION_CALL_BEGIN", flush=True)
        with torch.inference_mode():
            codes, scales = operator(
                x=operands["gate_int32"], weight_scale=operands["gate_scale"],
                activation_scale=operands["input_scale"], **kwargs)
        report["call_count"] = 1
        torch.npu.synchronize()
        report["call_end_unix_ns"] = time.time_ns()
        print("SHARED_ACTIVATION_CALL_END", flush=True)
        actual = {"middle": codes.cpu(), "middle_scale": scales.cpu()}
        report["reproduction"] = {
            name: compare(torch, value, reference[name])
            for name, value in actual.items()}
        torch.save(actual, args.output / "activation.pt")
        if not all(v["bitwise_equal"] for v in report["reproduction"].values()):
            raise RuntimeError("The original activation did not reproduce its capture")
        report["completed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
