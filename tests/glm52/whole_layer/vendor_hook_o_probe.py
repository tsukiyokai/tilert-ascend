#!/usr/bin/env python3
"""Replay captured O projection with original and exactly widened scales."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def tensor_identity(torch, value):
    value = value.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest()}


def compare(torch, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Comparison requires equal shape and dtype")
    a = actual.contiguous().reshape(-1).view(torch.uint8)
    b = expected.contiguous().reshape(-1).view(torch.uint8)
    mask = (a != b).reshape(-1, actual.element_size()).any(1)
    details = []
    for index in mask.nonzero().flatten()[:16].tolist():
        coordinate = tuple(int(item) for item in torch.unravel_index(
            torch.tensor(index), actual.shape))
        details.append({"coordinate": list(coordinate),
                        "actual": actual[coordinate].item(),
                        "expected": expected[coordinate].item()})
    return {"bitwise_equal": not bool(mask.any()),
            "different_elements": int(mask.sum()),
            "different_bytes": int((a != b).sum()),
            "elements": actual.numel(), "first_differences": details}


def scalar_identity(torch, value):
    value = value.reshape(1)
    if value.dtype == torch.float32:
        bits = int(value.view(torch.int32)[0]) & 0xffffffff
        width = 8
    elif value.dtype == torch.bfloat16:
        bits = int(value.view(torch.int16)[0]) & 0xffff
        width = 4
    else:
        raise ValueError("Expected FP32 or BF16")
    return {"value": value[0].item(), "bits": f"0x{bits:0{width}x}"}


def cpu_reference(torch, captured):
    vendor, native = captured["vendor"], captured["native"]
    codes, weight = vendor["o_codes"], vendor["o_weight"]
    if codes.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("Captured O input and weight must be INT8")
    if codes.shape != (12, 1024) or weight.shape != (1024, 6144):
        raise ValueError("This discriminator is limited to the captured M12 O shape")
    channel = vendor["o_weight_scale"]
    token = vendor["o_token_scale"]
    if channel.dtype != torch.bfloat16 or token.dtype != torch.float32:
        raise ValueError("Expected original BF16 channel and FP32 token scales")
    if "o_bias" in vendor:
        raise ValueError("This captured production call has no bias")
    wide = codes.long() @ weight.long()
    if wide.min() < -(2 ** 31) or wide.max() >= 2 ** 31:
        raise ValueError("Captured INT8 product exceeds INT32")
    accumulators = wide.int()
    if not compare(torch, accumulators, native["o_accumulators"])["bitwise_equal"]:
        raise ValueError("Captured native INT32 accumulators differ from the integer oracle")
    acc = accumulators.float()
    channel = channel.float().reshape(1, 6144)
    token = token.reshape(12, 1)
    channel_product = acc * channel
    combined_scale = channel * token
    channel_then_token = channel_product * token
    combined = acc * combined_scale
    formulas = {"channel_then_token": channel_then_token.bfloat16(),
                "combined_scales": combined.bfloat16()}
    witnesses = []
    for row, column in ((0, 4423), (5, 2813)):
        witnesses.append({
            "coordinate": [row, column],
            "accumulator_int32": int(accumulators[row, column]),
            "accumulator_float32": scalar_identity(torch, acc[row, column]),
            "channel_float32": scalar_identity(torch, channel[0, column]),
            "token_float32": scalar_identity(torch, token[row, 0]),
            "accumulator_times_channel": scalar_identity(
                torch, channel_product[row, column]),
            "channel_times_token": scalar_identity(
                torch, combined_scale[row, column]),
            "channel_then_token_float32": scalar_identity(
                torch, channel_then_token[row, column]),
            "combined_scales_float32": scalar_identity(
                torch, combined[row, column]),
            "channel_then_token_bfloat16": scalar_identity(
                torch, formulas["channel_then_token"][row, column]),
            "combined_scales_bfloat16": scalar_identity(
                torch, formulas["combined_scales"][row, column]),
            "captured_vendor_bfloat16": scalar_identity(
                torch, vendor["o_quant_matmul_output"][row, column]),
            "captured_native_bfloat16": scalar_identity(
                torch, native["output"][row, column]),
        })
    return formulas, witnesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Captured M12 INT8 O projection; scale descriptor is the only changed input",
              "script": file_identity(__file__), "capture": file_identity(args.capture),
              "python": sys.version, "platform": platform.platform(),
              "cpu_only": args.cpu_only, "cases": [], "pass": False}

    def persist():
        (args.output / "o-projection.json").write_text(
            json.dumps(report, indent=2) + "\n")

    try:
        import torch

        torch.set_num_threads(1)
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        vendor = captured["vendor"]
        formulas, witnesses = cpu_reference(torch, captured)
        report.update(torch=torch.__version__, witnesses=witnesses,
                      inputs={key: tensor_identity(torch, vendor[key]) for key in (
                          "o_codes", "o_weight", "o_weight_scale", "o_token_scale")},
                      captured_vendor=tensor_identity(torch, vendor["o_quant_matmul_output"]),
                      captured_native=tensor_identity(torch, captured["native"]["output"]),
                      cpu_formulas={name: compare(torch, value,
                                                vendor["o_quant_matmul_output"])
                                    for name, value in formulas.items()})
        persist()
        if args.cpu_only:
            report["pass"] = True
            return

        import torch_npu

        torch.npu.set_device(0)
        if torch.npu.device_count() != 1:
            raise RuntimeError("Expose exactly one NPU")
        torch.npu.config.allow_internal_format = True

        def nd(value):
            return torch_npu.npu_format_cast(value.contiguous().to("npu"), 2)

        codes = nd(vendor["o_codes"])
        weight = torch_npu.npu_format_cast(nd(vendor["o_weight"]), 29)
        token_scale = nd(vendor["o_token_scale"])
        scales = {"bfloat16": nd(vendor["o_weight_scale"]),
                  "float32": nd(vendor["o_weight_scale"].float())}
        if torch_npu.get_npu_format(weight) != 29:
            raise RuntimeError("The production O weight must remain NZ")
        for name, value in (("o_codes", codes), ("o_weight", weight),
                            ("o_token_scale", token_scale)):
            if tensor_identity(torch, value)["sha256"] != report["inputs"][name]["sha256"]:
                raise RuntimeError(f"Device packing changed {name}")
        if not torch.equal(scales["bfloat16"].cpu().float(), scales["float32"].cpu()):
            raise RuntimeError("Scale widening changed numeric values")
        report.update(torch_npu=torch_npu.__version__,
                      torch_npu_extension=file_identity(torch_npu._C.__file__),
                      device=torch.npu.get_device_name(0),
                      device_properties=str(torch.npu.get_device_properties(0)),
                      quant_matmul_schema=str(torch.ops.npu.npu_quant_matmul.default._schema),
                      call={"bias": None, "offset_passed": False,
                            "output_dtype": "torch.bfloat16", "weight_format": 29},
                      device_inputs={name: {"data_ptr": value.data_ptr(),
                                            "storage_ptr": value.untyped_storage().data_ptr(),
                                            "storage_offset": value.storage_offset(),
                                            "shape": list(value.shape),
                                            "stride": list(value.stride()),
                                            "dtype": str(value.dtype),
                                            "format": torch_npu.get_npu_format(value)}
                                     for name, value in (("codes", codes), ("weight", weight),
                                                         ("token_scale", token_scale),
                                                         *scales.items())})
        outputs = {}
        for iteration in range(3):
            order = ("bfloat16", "float32") if iteration % 2 == 0 else ("float32", "bfloat16")
            for dtype in order:
                result = torch_npu.npu_quant_matmul(
                    codes, weight, scales[dtype], pertoken_scale=token_scale,
                    bias=None, output_dtype=torch.bfloat16)
                torch.npu.synchronize()
                actual = result.cpu().clone()
                outputs[f"{dtype}_{iteration}"] = actual
                case = {"iteration": iteration, "scale_dtype": dtype,
                        "output": tensor_identity(torch, actual),
                        "captured_vendor": compare(torch, actual, vendor["o_quant_matmul_output"]),
                        "captured_native": compare(torch, actual, captured["native"]["output"]),
                        "formulas": {name: compare(torch, actual, expected)
                                     for name, expected in formulas.items()}}
                report["cases"].append(case)
                persist()
                print(json.dumps(case), flush=True)
        paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "/" in line and "libopapi" in line.split()[-1]}
        report["loaded_opapi"] = [file_identity(path) for path in sorted(paths)]
        torch.save(outputs, args.output / "o-projection-outputs.pt")
        report["pass"] = (
            all(case["captured_vendor"]["bitwise_equal"] for case in report["cases"]
                if case["scale_dtype"] == "bfloat16")
            and all(any(value["bitwise_equal"] for value in case["formulas"].values())
                    for case in report["cases"])
            and all(compare(torch, outputs[f"{dtype}_0"], outputs[f"{dtype}_{iteration}"])["bitwise_equal"]
                    for dtype in scales for iteration in (1, 2)))
        if not report["pass"]:
            raise RuntimeError("Captured-call reproduction, formula coverage or repeatability failed")
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
