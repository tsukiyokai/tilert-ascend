"""Exercise the pinned BF16 gate call with its original NK weight layout."""

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from fixtures import bridge_case


def dispatch_identity(torch):
    name = "aten::mm.dtype"
    return {
        "schema": str(torch.ops.aten.mm.dtype._schema),
        "privateuse1_kernel": torch._C._dispatch_has_kernel_for_dispatch_key(
            name, "PrivateUse1"),
        "dispatch_table": torch._C._dispatch_dump_table(name),
    }


def descriptor(torch_npu, value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "format": torch_npu.get_npu_format(value),
            "storage_offset": value.storage_offset()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 12, 32])
    parser.add_argument("--layouts", nargs="+", choices=("nd", "nz"),
                        default=["nd", "nz"])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    import torch
    import torch_npu

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Pinned AscendGateLinear torch.mm BF16 x BF16 to FP32 entry",
        "source_commit": "8237ee6d690d1399bc6eb48ef5977ac2bd470f3f",
        "source_call": "torch.mm(x, self.weight.t(), out_dtype=torch.float32)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "dispatch": dispatch_identity(torch), "cases": [],
        "npu_initialized_before_probe": torch.npu.is_initialized(),
    }
    if not args.preflight_only:
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        with torch.inference_mode():
            for rows in args.rows:
                hidden, _, _, logical_kn, _ = bridge_case(torch, rows, 0)
                x = torch_npu.npu_format_cast(hidden.to(args.device), 2)
                original_nk = logical_kn.t().contiguous().to(args.device)
                for layout in args.layouts:
                    weight = torch_npu.npu_format_cast(
                        original_nk, 2 if layout == "nd" else 29)
                    transposed = weight.t()
                    case = {"rows": rows, "layout": layout,
                            "weight": descriptor(torch_npu, weight),
                            "operand": descriptor(torch_npu, transposed)}
                    try:
                        result = torch.mm(x, transposed, out_dtype=torch.float32)
                        torch.npu.synchronize()
                        case.update({"pass": result.dtype == torch.float32,
                                     "output": descriptor(torch_npu, result),
                                     "finite": bool(torch.isfinite(result).all())})
                    except Exception:
                        case.update({"pass": False, "error": traceback.format_exc()})
                    report["cases"].append(case)
    report["npu_initialized"] = torch.npu.is_initialized()
    report["pass"] = all(case["pass"] for case in report["cases"])
    if args.preflight_only and report["npu_initialized"]:
        raise RuntimeError("CPU preflight initialized NPU")
    (args.output / "gate-entry.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "cases": len(report["cases"]),
                      "privateuse1_kernel": report["dispatch"]["privateuse1_kernel"],
                      "device_calls_executed": not args.preflight_only}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
