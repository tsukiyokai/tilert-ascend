"""Observe native A3 Div/Mul/RINT on frozen DynamicQuant inputs."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fp32_bits(value):
    return f"{int(value.view(torch.int32)) & 0xffffffff:08x}"


@torch.inference_mode()
def run(args):
    report_path = args.capture / "result.json"
    report = json.loads(report_path.read_text())
    source = args.capture / "input.pt"
    device_source = args.capture / "device.pt"
    assert digest(source) == report["prepared_sha256"]
    assert digest(device_source) == report["device_sha256"]
    saved = torch.load(source, map_location="cpu", weights_only=True)
    actual = torch.load(device_source, map_location="cpu", weights_only=True)
    normalized = saved["reference"]["input_norm"].flatten().float()
    maximum = normalized.abs().max()
    half_rows = (normalized.abs() == maximum / 2).nonzero().flatten()
    assert 0 < half_rows.numel() <= 32
    selected = torch.cat((half_rows, torch.arange(64 - half_rows.numel())))
    x = normalized[selected]
    literal = torch.tensor([-63.5, 63.5, -62.5, 62.5,
                            -1.5, 1.5, -0.5, 0.5]).repeat(8)
    inputs = torch.stack((x, maximum.expand(64), literal))
    reference_factor = torch.full_like(x, 127) / maximum
    reference_product = x * reference_factor

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    capacity = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(1536 * 2**20 / capacity)
    torch.npu.reset_peak_memory_stats()
    device_inputs = inputs.to(args.device)
    floats = torch.empty(2, 64, dtype=torch.float32, device=args.device)
    integers = torch.empty(2, 64, dtype=torch.int32, device=args.device)
    library = ctypes.CDLL(str(args.library.resolve()))
    launch = library.quant_div_probe_launch
    launch.argtypes = [ctypes.c_void_p] * 4
    launch.restype = None
    stream = torch.npu.current_stream(args.device)
    stream.synchronize()
    launch(stream.npu_stream, device_inputs.data_ptr(), floats.data_ptr(),
           integers.data_ptr())
    stream.synchronize()
    floats, integers = floats.cpu(), integers.cpu()
    expected_codes = actual["input_codes"].flatten()[selected].int()
    controls_exact = torch.equal(integers[1], literal.round().int())
    quantizer_exact = torch.equal(integers[0], expected_codes)
    payload = {"inputs": inputs, "floats": floats, "integers": integers,
               "reference_factor": reference_factor,
               "reference_product": reference_product,
               "frozen_codes": expected_codes, "selected_rows": selected}
    torch.save(payload, args.output.with_suffix(".pt"))
    result = {
        "scope": "One AIV, native FP32 Div/Mul/CAST_RINT, 64 lanes",
        "device": args.device, "performance_measured": False,
        "amax": float(maximum), "half_amax_lanes": half_rows.numel(),
        "literal_rint_exact": controls_exact,
        "frozen_quantizer_codes_exact": quantizer_exact,
        "factor_ulp": (floats[0].view(torch.int32)
                       - reference_factor.view(torch.int32)).tolist(),
        "half_amax": [{
            "input_row": int(selected[i]), "x": float(x[i]),
            "factor": float(floats[0, i]),
            "factor_bits": fp32_bits(floats[0, i]),
            "product": float(floats[1, i]),
            "product_bits": fp32_bits(floats[1, i]),
            "rint": int(integers[0, i]),
            "cpu_product": float(reference_product[i]),
        } for i in range(half_rows.numel())],
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        "artifacts": {str(p): digest(p) for p in (
            source, device_source, report_path, args.library,
            args.output.with_suffix(".pt"), Path(__file__),
            Path(__file__).with_name("quant_div_probe.cpp"),
            Path(__file__).with_name("CMakeLists.txt"))},
        "status": "pass" if controls_exact and quantizer_exact else "fail",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "status", "amax", "literal_rint_exact",
        "frozen_quantizer_codes_exact", "half_amax")}))
    return result["status"] == "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(not run(args))


if __name__ == "__main__":
    main()
