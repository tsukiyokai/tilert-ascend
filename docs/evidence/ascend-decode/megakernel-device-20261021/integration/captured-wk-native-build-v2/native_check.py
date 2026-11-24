#!/usr/bin/env python3
"""Compare accepted N32 and full-N160 native WK on the captured operands."""

import argparse
import ctypes
import json
from pathlib import Path
import sys
import traceback

import check


FIELD_NAMES = (
    "M", "N", "Ka", "Kb", "singleCoreM", "singleCoreN", "singleCoreK",
    "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM", "stepN",
    "stepKa", "stepKb", "iterateOrder", "shareL1Size", "shareL0CSize",
    "shareUbSize", "dbL0A", "dbL0B", "dbL0C",
)


def packed_weight(torch, weight, columns):
    if columns not in (32, 160) or weight.shape != (160, 6144):
        raise ValueError("Expected N32 or N160 packing of the captured WK weight")
    packed = weight.reshape(160 // columns, columns, 6144).transpose(1, 2).contiguous()
    restored = packed.transpose(1, 2).reshape(160, 6144)
    if not check.compare(torch, restored, weight)["bitwise_equal"]:
        raise ValueError("WK tile packing changed logical weights")
    return packed


def unpack_output(output, columns):
    return output.reshape(160 // columns, 32, columns).transpose(0, 1).reshape(32, 160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=24)
    args = parser.parse_args()
    if args.repeats < 2 or not 1 <= args.blocks <= 24:
        parser.error("Require at least two repeats and 1..24 AIC groups")
    if not args.cpu_only and args.library is None:
        parser.error("--library is required for the device probe")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Captured M32 WK only; accepted N32 FP32 accumulator control and native full-N160 geometry",
        "script": check.file_identity(__file__),
        "helpers": check.file_identity(check.__file__),
        "capture": check.file_identity(args.capture), "python": sys.version,
        "cpu_only": args.cpu_only, "blocks": args.blocks,
        "cases": [], "completed": False, "pass": False,
    }

    def persist():
        (args.output / "wk-native.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        import torch

        torch.set_num_threads(1)
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        vendor, native = captured["vendor"], captured["native"]
        dot, witnesses = check.cpu_contract(torch, captured)
        weights_cpu = {n: packed_weight(torch, vendor["key_weight"], n)
                       for n in (32, 160)}
        for columns in (32, 160):
            sentinel = torch.arange(32 * 160, dtype=torch.float32).reshape(32, 160)
            tiled = sentinel.reshape(32, 160 // columns, columns).transpose(0, 1).contiguous()
            if not torch.equal(unpack_output(tiled, columns), sentinel):
                raise RuntimeError("Output tile decoding does not preserve coordinates")
        report.update(torch=torch.__version__, witnesses=witnesses,
                      packing={str(n): check.tensor_identity(torch, value)
                               for n, value in weights_cpu.items()})
        if args.cpu_only:
            report.update(completed=True, **{"pass": True})
            return

        import torch_npu

        torch.npu.set_device(0)
        if torch.npu.device_count() != 1 or torch.npu.get_device_name(0) != "Ascend910_9392":
            raise RuntimeError("This probe requires one exposed Ascend910_9392")
        torch.npu.config.allow_internal_format = True

        def nd(value):
            return torch_npu.npu_format_cast(value.contiguous().to("npu"), 2)

        hidden = nd(vendor["key_hidden"])
        original_weight = torch_npu.npu_format_cast(nd(vendor["key_weight"]), 29)
        if torch_npu.get_npu_format(original_weight) != 29:
            raise RuntimeError("Original F.linear weight must retain NZ format")
        reference = torch.nn.functional.linear(hidden, original_weight, bias=None)
        torch.npu.synchronize()
        report["production_reproduction"] = check.compare(
            torch, reference.cpu(), vendor["key_projection"])
        if not report["production_reproduction"]["bitwise_equal"]:
            raise RuntimeError("Original F.linear did not reproduce the captured WK projection")
        library = ctypes.CDLL(str(args.library.resolve()))
        prepare = library.wk_geometry_tiling
        prepare.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
        prepare.restype = ctypes.c_int
        launch = library.wk_geometry_launch
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, *([ctypes.c_void_p] * 5),
                           ctypes.c_int32]
        launch.restype = None
        report.update(library=check.file_identity(args.library),
                      torch_npu=torch_npu.__version__,
                      torch_npu_extension=check.file_identity(torch_npu._C.__file__),
                      device=torch.npu.get_device_name(0),
                      device_properties=str(torch.npu.get_device_properties(0)),
                      original_weight=check.device_descriptor(torch_npu, original_weight),
                      original_right_operand=check.device_descriptor(torch_npu, original_weight.t()),
                      native_inputs={}, tiling={}, tiling_raw_hex={})
        prepared = {}
        for columns in (32, 160):
            raw = (ctypes.c_uint8 * 512)()
            fields = (ctypes.c_int64 * len(FIELD_NAMES))()
            result = prepare(columns, raw, fields)
            if result != 0:
                raise RuntimeError(f"N{columns} tiling failed: {result}")
            tiling_cpu = torch.tensor(list(raw), dtype=torch.uint8)
            weight = nd(weights_cpu[columns])
            if not check.compare(torch, weight.cpu(), weights_cpu[columns])["bitwise_equal"]:
                raise RuntimeError("Device copy changed the packed WK values")
            prepared[columns] = (weight, nd(tiling_cpu))
            report["native_inputs"][str(columns)] = check.device_descriptor(torch_npu, weight)
            report["tiling"][str(columns)] = dict(zip(FIELD_NAMES, fields))
            report["tiling_raw_hex"][str(columns)] = bytes(raw).hex()
        if report["tiling"]["32"]["baseK"] != 512:
            raise RuntimeError("The original accepted N32 tiling changed")
        output = nd(torch.empty(32 * 160, dtype=torch.float32))
        workspace = nd(torch.empty(16 << 20, dtype=torch.uint8))
        report["workspace_bytes"] = workspace.numel()
        outputs = {}
        persist()
        for iteration in range(args.repeats):
            for columns in ((32, 160) if iteration % 2 == 0 else (160, 32)):
                weight, tiling = prepared[columns]
                output.fill_(float("nan"))
                # The stream property flushes queued Torch producers before CABI launch.
                stream = torch.npu.current_stream().npu_stream
                launch(stream, args.blocks, hidden.data_ptr(), weight.data_ptr(),
                       output.data_ptr(), tiling.data_ptr(), workspace.data_ptr(), columns)
                torch.npu.synchronize()
                actual = unpack_output(output.cpu().clone(), columns).contiguous()
                if not torch.isfinite(actual).all():
                    raise RuntimeError(f"N{columns} left nonfinite output elements")
                case = {
                    "columns": columns, "iteration": iteration,
                    "accumulators": check.tensor_identity(torch, actual),
                    "captured_native_accumulators": check.compare(
                        torch, actual, native["key_accumulators"]),
                    "captured_native_projection": check.compare(
                        torch, actual.bfloat16(), native["key_projection"]),
                    "captured_vendor_projection": check.compare(
                        torch, actual.bfloat16(), vendor["key_projection"]),
                    "cpu_fp64_rounding": check.compare(torch, actual.bfloat16(), dot.bfloat16()),
                    "witnesses": [{"coordinate": item["coordinate"],
                                   "accumulator": check.scalar_identity(
                                       torch, actual[tuple(item["coordinate"])])}
                                  for item in witnesses],
                }
                if iteration:
                    case["repeatability"] = check.compare(torch, actual, outputs[columns])
                else:
                    outputs[columns] = actual
                report["cases"].append(case)
                torch.save(outputs, args.output / "wk-native-outputs.pt")
                persist()
                print(json.dumps(case), flush=True)
                if columns == 32 and not case["captured_native_accumulators"]["bitwise_equal"]:
                    raise RuntimeError("Original N32 native control did not reproduce all FP32 accumulators")
                if iteration and not case["repeatability"]["bitwise_equal"]:
                    raise RuntimeError("Native repeated result changed")
        report["completed"] = True
        report["pass"] = all(case["captured_vendor_projection"]["bitwise_equal"]
                             for case in report["cases"] if case["columns"] == 160)
        if not report["pass"]:
            raise RuntimeError("Full-N160 native candidate does not match the production projection")
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
