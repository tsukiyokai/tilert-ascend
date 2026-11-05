#!/usr/bin/env python3
"""Replay the captured M32 WK projection across geometry and layout choices."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback


@dataclass(frozen=True)
class Arm:
    name: str
    columns: int
    layout: str
    operation: str
    float_output: bool = False


ARMS = (
    Arm("production_nz_full", 160, "nk_nz", "linear"),
    Arm("nz_split", 32, "nk_nz", "linear"),
    Arm("nd_full", 160, "nk_nd", "linear"),
    Arm("nd_split", 32, "nk_nd", "linear"),
    Arm("nd_kn_full", 160, "kn_nd", "mm"),
    Arm("nd_kn_split", 32, "kn_nd", "mm"),
    Arm("nz_full_mm", 160, "nk_nz", "mm"),
    Arm("nz_full_mm_float32", 160, "nk_nz", "mm", True),
    Arm("nd_kn_split_float32", 32, "kn_nd", "mm", True),
)


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


def scalar_identity(torch, value):
    value = value.reshape(1)
    if value.dtype == torch.bfloat16:
        bits = int(value.view(torch.int16)[0]) & 0xffff
        width = 4
    elif value.dtype == torch.float32:
        bits = int(value.view(torch.int32)[0]) & 0xffffffff
        width = 8
    else:
        raise ValueError("Scalar bit witnesses require BF16 or FP32")
    return {"value": value.item(), "bits": f"0x{bits:0{width}x}"}


def compare(torch, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Comparison requires equal shape and dtype")
    left = actual.contiguous().reshape(-1).view(torch.uint8)
    right = expected.contiguous().reshape(-1).view(torch.uint8)
    mask = (left != right).reshape(-1, actual.element_size()).any(1)
    details = []
    for flat in mask.nonzero().flatten()[:16].tolist():
        coordinate = tuple(int(item) for item in torch.unravel_index(
            torch.tensor(flat), actual.shape))
        details.append({"coordinate": list(coordinate),
                        "actual": scalar_identity(torch, actual[coordinate]),
                        "expected": scalar_identity(torch, expected[coordinate])})
    return {"bitwise_equal": not bool(mask.any()),
            "different_elements": int(mask.sum()),
            "different_bytes": int((left != right).sum()),
            "elements": actual.numel(), "first_differences": details}


def cpu_contract(torch, captured):
    vendor, native = captured["vendor"], captured["native"]
    expected = {"key_hidden": (32, 6144), "key_weight": (160, 6144),
                "key_projection": (32, 160)}
    for name, shape in expected.items():
        value = vendor[name]
        if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
            raise ValueError(f"Captured {name} must be BF16{shape}")
    for left, right in ((vendor["key_hidden"], native["hidden"]),
                        (vendor["key_weight"], native["key_weight"])):
        if not compare(torch, left, right)["bitwise_equal"]:
            raise ValueError("Captured WK operands differ between the two arms")
    accumulators = native["key_accumulators"]
    if accumulators.shape != (32, 160) or accumulators.dtype != torch.float32:
        raise ValueError("Expected captured FP32[32,160] native accumulators")
    if not compare(torch, accumulators.bfloat16(),
                   native["key_projection"])["bitwise_equal"]:
        raise ValueError("Native projection is not the RNE cast of its accumulator")
    dot = vendor["key_hidden"].double() @ vendor["key_weight"].double().t()
    retained = captured["oracles"]["key_projection_cpu_fp64_dot"]
    if not torch.equal(dot, retained):
        raise ValueError("CPU FP64 replay differs from the retained auxiliary dot")
    mask = vendor["key_projection"].view(torch.int16) != native[
        "key_projection"].view(torch.int16)
    witnesses = []
    for row, column in mask.nonzero().tolist():
        original = vendor["key_projection"][row, column]
        candidate = native["key_projection"][row, column]
        accumulator = accumulators[row, column]
        midpoint = (original.double().item() + candidate.double().item()) / 2
        exact = dot[row, column].item()
        witnesses.append({
            "coordinate": [row, column],
            "vendor": scalar_identity(torch, original),
            "native": scalar_identity(torch, candidate),
            "native_accumulator": scalar_identity(torch, accumulator),
            "cpu_fp64_dot": exact,
            "cpu_fp64_cast_fp32": scalar_identity(torch, dot[row, column].float()),
            "cpu_fp64_cast_bfloat16": scalar_identity(
                torch, dot[row, column].bfloat16()),
            "bf16_midpoint": midpoint,
            "native_accumulator_minus_fp64": accumulator.item() - exact,
            "native_accumulator_minus_midpoint": accumulator.item() - midpoint,
            "fp64_minus_midpoint": exact - midpoint,
        })
    return dot, witnesses


def device_descriptor(torch_npu, value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "format": torch_npu.get_npu_format(value),
            "data_ptr": value.data_ptr(),
            "storage_ptr": value.untyped_storage().data_ptr(),
            "storage_offset": value.storage_offset()}


def prepare_arm(torch, torch_npu, source, arm):
    weights, descriptors = [], []
    for start in range(0, 160, arm.columns):
        expected = source[start:start + arm.columns].contiguous()
        logical = expected.t().contiguous() if arm.layout == "kn_nd" else expected
        weight = torch_npu.npu_format_cast(logical.to("npu"), 2)
        if arm.layout == "nk_nz":
            weight = torch_npu.npu_format_cast(weight, 29)
        wanted_format = 29 if arm.layout == "nk_nz" else 2
        if torch_npu.get_npu_format(weight) != wanted_format:
            raise RuntimeError(f"{arm.name}: requested physical format was not retained")
        restored = weight.cpu().t() if arm.layout == "kn_nd" else weight.cpu()
        if not compare(torch, restored, expected)["bitwise_equal"]:
            raise RuntimeError(f"{arm.name}: packing changed WK values")
        operand = weight if arm.layout == "kn_nd" else weight.t()
        weights.append((weight, operand))
        descriptors.append({"start_column": start,
                            "weight": device_descriptor(torch_npu, weight),
                            "mm_right_operand": device_descriptor(torch_npu, operand)})
    return weights, descriptors


def run_arm(torch, hidden, weights, arm):
    pieces = []
    for weight, operand in weights:
        if arm.operation == "linear":
            result = torch.nn.functional.linear(hidden, weight, bias=None)
        elif arm.float_output:
            result = torch.ops.aten.mm.dtype(hidden, operand, out_dtype=torch.float32)
        else:
            result = torch.mm(hidden, operand)
        pieces.append(result)
    torch.npu.synchronize()
    # Concatenation is on CPU, after every GEMM has completed.
    return torch.cat([value.cpu().clone() for value in pieces], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least two")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Captured BF16 M32 K6144 N160 WK geometry/layout discriminator; no timing or full-layer acceptance",
        "script": file_identity(__file__), "capture": file_identity(args.capture),
        "python": sys.version, "platform": platform.platform(),
        "cpu_only": args.cpu_only, "cases": [], "completed": False,
        "interpretation": "Output dtype may change CANN tiling. A direct BF16 versus FP32-result comparison alone does not isolate Fixpipe rounding.",
    }

    def persist():
        (args.output / "wk-projection.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        import torch

        torch.set_num_threads(1)
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        vendor, native = captured["vendor"], captured["native"]
        dot, witnesses = cpu_contract(torch, captured)
        report.update(
            torch=torch.__version__, witnesses=witnesses,
            inputs={name: tensor_identity(torch, vendor[name])
                    for name in ("key_hidden", "key_weight")},
            captured_vendor=tensor_identity(torch, vendor["key_projection"]),
            captured_native=tensor_identity(torch, native["key_projection"]),
            captured_accumulators=tensor_identity(torch, native["key_accumulators"]),
            cpu_fp64_rounding={
                "vendor": compare(torch, dot.bfloat16(), vendor["key_projection"]),
                "native": compare(torch, dot.bfloat16(), native["key_projection"])})
        persist()
        if args.cpu_only:
            report["completed"] = True
            return

        import torch_npu

        torch.npu.set_device(0)
        if torch.npu.device_count() != 1:
            raise RuntimeError("Expose exactly one NPU")
        torch.npu.config.allow_internal_format = True
        hidden = torch_npu.npu_format_cast(vendor["key_hidden"].to("npu"), 2)
        if tensor_identity(torch, hidden) != report["inputs"]["key_hidden"]:
            raise RuntimeError("Device copy changed the captured hidden input")
        report.update(
            torch_npu=torch_npu.__version__,
            torch_npu_extension=file_identity(torch_npu._C.__file__),
            device=torch.npu.get_device_name(0),
            device_properties=str(torch.npu.get_device_properties(0)),
            schemas={"mm": str(torch.ops.aten.mm.default._schema),
                     "mm_dtype": str(torch.ops.aten.mm.dtype._schema)},
            hidden=device_descriptor(torch_npu, hidden), arms={})
        prepared, packing = {}, {}
        for arm in ARMS:
            key = (arm.columns, arm.layout)
            if key not in packing:
                packing[key] = prepare_arm(torch, torch_npu, vendor["key_weight"], arm)
            prepared[arm.name], descriptors = packing[key]
            report["arms"][arm.name] = {**vars(arm), "bias": None,
                                        "inputs": descriptors}
        persist()
        outputs = {}
        for iteration in range(args.repeats):
            order = ARMS if iteration % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                actual = run_arm(torch, hidden, prepared[arm.name], arm)
                wanted_dtype = torch.float32 if arm.float_output else torch.bfloat16
                if actual.shape != (32, 160) or actual.dtype != wanted_dtype:
                    raise RuntimeError(f"{arm.name}: unexpected output descriptor")
                if not torch.isfinite(actual).all():
                    raise RuntimeError(f"{arm.name}: nonfinite result")
                rounded = actual.bfloat16()
                case = {
                    "arm": arm.name, "iteration": iteration,
                    "output": tensor_identity(torch, actual),
                    "captured_vendor": compare(torch, rounded, vendor["key_projection"]),
                    "captured_native": compare(torch, rounded, native["key_projection"]),
                    "cpu_fp64_rounding": compare(torch, rounded, dot.bfloat16()),
                    "witnesses": [{"coordinate": item["coordinate"],
                                   "output": scalar_identity(
                                       torch, actual[tuple(item["coordinate"])])}
                                  for item in witnesses],
                }
                if arm.float_output:
                    case["captured_native_accumulators"] = compare(
                        torch, actual, native["key_accumulators"])
                if iteration:
                    case["repeatability"] = compare(torch, actual, outputs[arm.name])
                else:
                    outputs[arm.name] = actual
                report["cases"].append(case)
                torch.save(outputs, args.output / "wk-projection-outputs.pt")
                persist()
                print(json.dumps(case), flush=True)
                if arm.name == "production_nz_full" and not case[
                        "captured_vendor"]["bitwise_equal"]:
                    raise RuntimeError("The original full-N NZ call did not reproduce the captured WK output")
                if iteration and not case["repeatability"]["bitwise_equal"]:
                    raise RuntimeError(f"{arm.name}: output changed across identical repetitions")
        report["paired_output_dtype"] = compare(
            torch, outputs["nz_full_mm_float32"].bfloat16(), outputs["nz_full_mm"])
        paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "/" in line and "libopapi" in line.split()[-1]}
        report["loaded_opapi"] = [file_identity(path) for path in sorted(paths)]
        report["production_reproduction_pass"] = True
        report["repeatability_pass"] = True
        report["completed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
