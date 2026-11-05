#!/usr/bin/env python3
"""Replay the captured WK using natural and MatMulV2 K traversal in one L0C."""

import argparse
import ctypes
import json
from pathlib import Path
import sys
import traceback

import contract as check


def outer_order(column_tile, shifted):
    if not shifted:
        return list(range(12))
    return [((11 - i if column_tile & 1 else i) + (column_tile // 2) * 2) % 12
            for i in range(12)]


def retain_raw_outputs(torch, accumulators, projection, path):
    """Persist all physical output rows before validating the active row."""
    values = {"accumulators": accumulators, "projection": projection}
    saved, summary = {}, {}
    for name, value in values.items():
        owned = value.detach().cpu().contiguous().clone()
        finite = torch.isfinite(owned)
        guard = owned[1:].view(torch.uint8) == 0xa5
        saved[name] = owned
        saved[name + "_finite"] = finite
        saved[name + "_guard_bytes_equal"] = guard
        summary[name] = {
            "tensor": check.tensor_identity(torch, owned),
            "nonfinite_active_coordinates": (~finite[:1]).nonzero().tolist(),
            "finite_elements_per_row": finite.sum(dim=1).tolist(),
            "guard_different_bytes": int((~guard).sum()),
        }
    torch.save(saved, path)
    return {"dump": check.file_identity(path), "outputs": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2 or (not args.cpu_only and args.library is None):
        parser.error("Require repeats>=2 and a device library unless --cpu-only")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Captured M1 N160 K6144 only; natural/shifted BF16 Mmad with 48 updates of one FP32 L0C tile",
        "script": check.file_identity(__file__), "helper": check.file_identity(check.__file__),
        "capture": check.file_identity(args.capture), "python": sys.version,
        "capture_report": check.file_identity(args.capture_report),
        "cpu_only": args.cpu_only, "cases": [], "completed": False, "pass": False,
        "geometry": {"blocks": 10, "M": 1, "N": 160, "K": 6144,
                     "single_core_M": 1, "single_core_N": 16, "mmad_K": 128,
                     "K_segment": 512, "K_segments": 12},
        "orders": {str(n): outer_order(n, True) for n in range(10)},
    }

    def persist():
        (args.output / "wk-kshift.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        import torch

        torch.set_num_threads(1)
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        vendor, native = captured["vendor"], captured["native"]
        if report["capture"]["sha256"] != check.CAPTURE_SHA256:
            raise ValueError("The captured M1 input identity changed")
        if report["capture_report"]["sha256"] != check.REPORT_SHA256:
            raise ValueError("The captured M1 descriptor report changed")
        original_descriptors = check.capture_descriptors(
            json.loads(args.capture_report.read_text()))
        witnesses = check.cpu_contract(torch, captured)
        dot = captured["oracles"]["key_projection_cpu_fp64"]
        mappings = []
        for item in witnesses:
            row, column = item["coordinate"]
            mappings.append({"coordinate": [row, column], "core": column // 16,
                             "M_tile": row // 16, "N_tile": column // 16,
                             "K512_order": outer_order(column // 16, True)})
        if any(sorted(order) != list(range(12)) for order in report["orders"].values()):
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
        original_weight = weight
        for name, value in (("key_hidden", hidden), ("key_weight", weight)):
            check.require_device_descriptor(
                check.device_descriptor(torch_npu, value), original_descriptors[name])
        reference = torch.nn.functional.linear(hidden, original_weight, bias=None)
        torch.npu.synchronize()
        report["production_reproduction"] = check.compare(torch, reference.cpu(), vendor["key_projection"])
        if not report["production_reproduction"]["bitwise_equal"]:
            raise RuntimeError("Original F.linear did not reproduce captured WK")
        if torch_npu.get_npu_format(weight) != 2:
            raise RuntimeError("Native WK input weight changed physical format")
        output = nd(torch.empty(16, 160, dtype=torch.bfloat16))
        accumulators = nd(torch.empty(16, 160, dtype=torch.float32))
        library = ctypes.CDLL(str(args.library.resolve()))
        launch = library.wk_m1_kshift_launch
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
                output[:1].fill_(float("nan"))
                accumulators[:1].fill_(float("nan"))
                output[1:].view(torch.uint8).fill_(0xa5)
                accumulators[1:].view(torch.uint8).fill_(0xa5)
                stream = torch.npu.current_stream().npu_stream
                launch(stream, hidden.data_ptr(), weight.data_ptr(), accumulators.data_ptr(),
                       output.data_ptr(), int(shifted))
                torch.npu.synchronize()
                all_acc = accumulators.cpu().clone()
                all_bf16 = output.cpu().clone()
                name = "shifted" if shifted else "natural"
                raw = retain_raw_outputs(
                    torch, all_acc, all_bf16,
                    args.output / f"wk-kshift-raw-{name}-r{repeat}.pt")
                raw.update(arm=name, repeat=repeat)
                report.setdefault("raw_calls", []).append(raw)
                persist()
                if not (bool((all_acc[1:].view(torch.uint8) == 0xa5).all())
                        and bool((all_bf16[1:].view(torch.uint8) == 0xa5).all())):
                    raise RuntimeError("Kernel overwrote the fifteen guard rows after M1")
                actual_acc, actual_bf16 = all_acc[:1], all_bf16[:1]
                if not torch.isfinite(actual_acc).all() or not torch.isfinite(actual_bf16).all():
                    raise RuntimeError("Kernel left nonfinite output elements")
                case = {"arm": name, "repeat": repeat, "fifteen_guard_rows_unchanged": True,
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
                if not shifted and not case["captured_native_projection"]["bitwise_equal"]:
                    raise RuntimeError("Natural-K Mmad control does not reproduce the captured BF16 projection")
                if not case["direct_bf16_vs_fp32_cast"]["bitwise_equal"]:
                    raise RuntimeError("Direct BF16 Fixpipe differs from the same L0C FP32 RNE cast")
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
