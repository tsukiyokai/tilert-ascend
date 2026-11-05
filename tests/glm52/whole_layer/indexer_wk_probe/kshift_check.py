#!/usr/bin/env python3
"""Replay the captured WK using natural and MatMulV2 K traversal in one L0C."""

import argparse
import ctypes
import json
from pathlib import Path
import sys
import traceback

import check


def outer_order(column_tile, shifted):
    if not shifted:
        return list(range(24))
    return [((23 - i if column_tile & 1 else i) + (column_tile // 2) * 4) % 24
            for i in range(24)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2 or (not args.cpu_only and args.library is None):
        parser.error("Require repeats>=2 and a device library unless --cpu-only")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Captured M32 N160 K6144 only; natural/shifted BF16 Mmad with 48 updates of one FP32 L0C tile",
        "script": check.file_identity(__file__), "helper": check.file_identity(check.__file__),
        "capture": check.file_identity(args.capture), "python": sys.version,
        "cpu_only": args.cpu_only, "cases": [], "completed": False, "pass": False,
        "geometry": {"blocks": 20, "M": 32, "N": 160, "K": 6144,
                     "single_core_M": 16, "single_core_N": 16, "mmad_K": 128,
                     "K_segment": 256, "K_segments": 24},
        "orders": {str(n): outer_order(n, True) for n in range(10)},
    }

    def persist():
        (args.output / "wk-kshift.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        import torch

        torch.set_num_threads(1)
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        vendor, native = captured["vendor"], captured["native"]
        dot, witnesses = check.cpu_contract(torch, captured)
        mappings = []
        for item in witnesses:
            row, column = item["coordinate"]
            mappings.append({"coordinate": [row, column], "core": (column // 16) * 2 + row // 16,
                             "M_tile": row // 16, "N_tile": column // 16,
                             "K256_order": outer_order(column // 16, True)})
        if any(sorted(order) != list(range(24)) for order in report["orders"].values()):
            raise RuntimeError("Shifted K traversal repeats or omits a segment")
        report.update(torch=torch.__version__, witnesses=witnesses, witness_mapping=mappings)
        if args.cpu_only:
            report.update(completed=True, **{"pass": True})
            return

        import torch_npu

        torch.npu.set_device(0)
        if torch.npu.device_count() != 1 or torch.npu.get_device_name(0) != "Ascend910_9392":
            raise RuntimeError("Require one exposed Ascend910_9392")
        torch.npu.config.allow_internal_format = True

        def nd(value):
            return torch_npu.npu_format_cast(value.contiguous().to("npu"), 2)

        hidden = nd(vendor["key_hidden"])
        weight = nd(vendor["key_weight"])
        original_weight = torch_npu.npu_format_cast(weight, 29)
        if torch_npu.get_npu_format(original_weight) != 29:
            raise RuntimeError("Original F.linear requires the captured NZ weight descriptor")
        reference = torch.nn.functional.linear(hidden, original_weight, bias=None)
        torch.npu.synchronize()
        report["production_reproduction"] = check.compare(torch, reference.cpu(), vendor["key_projection"])
        if not report["production_reproduction"]["bitwise_equal"]:
            raise RuntimeError("Original F.linear did not reproduce captured WK")
        if torch_npu.get_npu_format(weight) != 2:
            raise RuntimeError("Native WK input weight changed physical format")
        output = nd(torch.empty(32, 160, dtype=torch.bfloat16))
        accumulators = nd(torch.empty(32, 160, dtype=torch.float32))
        library = ctypes.CDLL(str(args.library.resolve()))
        launch = library.wk_kshift_launch
        launch.argtypes = [*([ctypes.c_void_p] * 5), ctypes.c_int32]
        launch.restype = None
        report.update(library=check.file_identity(args.library), torch_npu=torch_npu.__version__,
                      torch_npu_extension=check.file_identity(torch_npu._C.__file__),
                      device=torch.npu.get_device_name(0),
                      device_properties=str(torch.npu.get_device_properties(0)),
                      input_descriptors={name: check.device_descriptor(torch_npu, value)
                                         for name, value in (("hidden", hidden), ("weight", weight),
                                                             ("original_weight", original_weight),
                                                             ("original_right_operand", original_weight.t()))})
        retained = {}
        persist()
        for repeat in range(args.repeats):
            for shifted in ((False, True) if repeat % 2 == 0 else (True, False)):
                output.fill_(float("nan"))
                accumulators.fill_(float("nan"))
                stream = torch.npu.current_stream().npu_stream
                launch(stream, hidden.data_ptr(), weight.data_ptr(), accumulators.data_ptr(),
                       output.data_ptr(), int(shifted))
                torch.npu.synchronize()
                actual_acc = accumulators.cpu().clone()
                actual_bf16 = output.cpu().clone()
                if not torch.isfinite(actual_acc).all() or not torch.isfinite(actual_bf16).all():
                    raise RuntimeError("Kernel left nonfinite output elements")
                name = "shifted" if shifted else "natural"
                case = {"arm": name, "repeat": repeat,
                        "accumulators": check.tensor_identity(torch, actual_acc),
                        "projection": check.tensor_identity(torch, actual_bf16),
                        "captured_native_accumulators": check.compare(torch, actual_acc, native["key_accumulators"]),
                        "captured_native_projection": check.compare(torch, actual_bf16, native["key_projection"]),
                        "captured_vendor_projection": check.compare(torch, actual_bf16, vendor["key_projection"]),
                        "direct_bf16_vs_fp32_cast": check.compare(torch, actual_bf16, actual_acc.bfloat16()),
                        "cpu_fp64_rounding": check.compare(torch, actual_bf16, dot.bfloat16()),
                        "witnesses": [{"coordinate": item["coordinate"],
                                       "accumulator": check.scalar_identity(torch, actual_acc[tuple(item["coordinate"])]),
                                       "projection": check.scalar_identity(torch, actual_bf16[tuple(item["coordinate"])])}
                                      for item in witnesses]}
                if repeat:
                    case["repeatability_accumulators"] = check.compare(torch, actual_acc, retained[name]["accumulators"])
                    case["repeatability_projection"] = check.compare(torch, actual_bf16, retained[name]["projection"])
                else:
                    retained[name] = {"accumulators": actual_acc, "projection": actual_bf16}
                report["cases"].append(case)
                torch.save(retained, args.output / "wk-kshift-outputs.pt")
                persist()
                print(json.dumps(case), flush=True)
                if not shifted and not case["captured_native_accumulators"]["bitwise_equal"]:
                    raise RuntimeError("Natural-K Mmad control does not reproduce all captured FP32 accumulators")
                if repeat and not all(case[key]["bitwise_equal"] for key in
                                      ("repeatability_accumulators", "repeatability_projection")):
                    raise RuntimeError("Repeated native result changed")
        report["completed"] = True
        report["pass"] = all(case["captured_vendor_projection"]["bitwise_equal"]
                             for case in report["cases"] if case["arm"] == "shifted")
        if not report["pass"]:
            raise RuntimeError("Shifted-K candidate does not reproduce the production BF16 projection")
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
