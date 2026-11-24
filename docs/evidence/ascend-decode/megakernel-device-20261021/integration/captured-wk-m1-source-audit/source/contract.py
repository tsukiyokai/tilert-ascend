"""Validate the retained M1 WK inputs and finite rounding witnesses."""

import hashlib
from pathlib import Path


CAPTURE_SHA256 = "a95e0c289c7e8b4f0808f185aecffc8739e602d6a72896894baf79c640e54c41"
REPORT_SHA256 = "6b10c24c9754b8084699905dcbfd12c2b3abcf6a56f55d0d1488c65187e50a87"
DESCRIPTOR_FIELDS = ("shape", "stride", "dtype", "format", "storage_offset")


def capture_descriptors(report):
    if (report["original_graph_pass"] is not False
            or report["intermediate_evidence_accepted"] is not True
            or report["first_unequal_boundary_in_recorded_order"] != "key_projection"
            or not report["original_failure_reproduction"]["pass"]):
        raise ValueError("The original graph failure was not reproduced and localized")
    for group in ("live_input_checks", "live_producer_checks"):
        if len(report[group]) != 4 or not all(
                value["bitwise_equal"] for value in report[group].values()):
            raise ValueError(f"The original {group} were not accepted")
    actual = report["vendor_device_descriptors"]
    expected = {
        "key_hidden": {"shape": [1, 6144], "stride": [6144, 1]},
        "key_weight": {"shape": [160, 6144], "stride": [6144, 1]},
    }
    for name, geometry in expected.items():
        descriptor = {key: actual[name][key] for key in DESCRIPTOR_FIELDS}
        wanted = dict(geometry, dtype="torch.bfloat16", format=2,
                      storage_offset=0)
        if descriptor != wanted:
            raise ValueError(f"The retained {name} descriptor changed")
    return {name: actual[name] for name in expected}


def require_device_descriptor(actual, captured):
    if any(actual[key] != captured[key] for key in DESCRIPTOR_FIELDS):
        raise ValueError("The replay operand descriptor differs from the capture")


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
        bits, width = int(value.view(torch.int16)[0]) & 0xffff, 4
    elif value.dtype == torch.float32:
        bits, width = int(value.view(torch.int32)[0]) & 0xffffffff, 8
    else:
        raise ValueError("Scalar witnesses require BF16 or FP32")
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
    shapes = {"key_hidden": (1, 6144), "key_weight": (160, 6144),
              "key_projection": (1, 160)}
    for name, shape in shapes.items():
        value = vendor[name]
        if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
            raise ValueError(f"Captured {name} must be BF16{shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Captured {name} is not finite")
    for left, right in ((vendor["key_hidden"], native["hidden"]),
                        (vendor["key_weight"], native["key_weight"])):
        if not compare(torch, left, right)["bitwise_equal"]:
            raise ValueError("Captured WK operands differ between the two arms")
    accumulators = native["key_accumulators"]
    if accumulators.shape != (1, 160) or accumulators.dtype != torch.float32:
        raise ValueError("Expected captured FP32[1,160] native accumulators")
    if not compare(torch, accumulators.bfloat16(),
                   native["key_projection"])["bitwise_equal"]:
        raise ValueError("Native projection differs from its accumulator RNE cast")
    dot = vendor["key_hidden"].double() @ vendor["key_weight"].double().t()
    if not torch.equal(dot, captured["oracles"]["key_projection_cpu_fp64"]):
        raise ValueError("CPU FP64 replay differs from the retained auxiliary dot")
    differences = (vendor["key_projection"].view(torch.int16)
                   != native["key_projection"].view(torch.int16))
    witnesses = []
    for row, column in differences.nonzero().tolist():
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
    if [value["coordinate"] for value in witnesses] != [[0, 25]]:
        raise ValueError("The retained M1 WK failure changed")
    return witnesses


def device_descriptor(torch_npu, value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "format": torch_npu.get_npu_format(value),
            "data_ptr": value.data_ptr(),
            "storage_ptr": value.untyped_storage().data_ptr(),
            "storage_offset": value.storage_offset()}
