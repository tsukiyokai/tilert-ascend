"""Check an exponent-independent Div bias over positive BF16 maxima.

The existing 64-lane kernel supplies native FP32 Div/Mul and CAST_RINT.
A 128-entry table is emitted only when every mantissa has the same signed
factor-bit difference across all 248 tested exponents. The measured domain
is positive BF16 maxima with exponents [-120, 127].
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402


LANES = 64
FIRST_BITS = 0x0380
STOP_BITS = 0x7F80
MANTISSAS = 128
EXPONENTS = 248
CAPACITY_BYTES = 1536 * 2**20


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare():
    bits = torch.arange(FIRST_BITS, STOP_BITS, dtype=torch.int32)
    maximum = (bits << 16).view(torch.float32)
    x = maximum * 0.5
    literals = torch.tensor([-63.5, 63.5, -62.5, 62.5,
                             -1.5, 1.5, -0.5, 0.5]).repeat(bits.numel() // 8)
    inputs = torch.stack((x, maximum, literals))
    inputs = inputs.reshape(3, -1, LANES).permute(1, 0, 2).contiguous()
    factor = torch.div(torch.full_like(maximum, 127), maximum)
    assert bits.numel() == EXPONENTS * MANTISSAS == 31744
    assert inputs.shape == (496, 3, LANES)
    return bits, inputs, factor


def analyze(bits, inputs, floats, integers, cpu_factor):
    native_factor = floats[:, 0].reshape(-1)
    delta = (native_factor.view(torch.int32).long()
             - cpu_factor.view(torch.int32).long()).reshape(
                 EXPONENTS, MANTISSAS,
             )
    consistent = (delta == delta[0]).all(dim=0)
    finite = torch.isfinite(native_factor) & (native_factor > 0)
    controls = integers[:, 1] == inputs[:, 2].round().int()
    cpu_product = inputs[:, 0] * floats[:, 0]
    products = (floats[:, 1].contiguous().view(torch.int32)
                == cpu_product.contiguous().view(torch.int32))
    rounded = integers[:, 0] == floats[:, 1].round().int()
    complete = bool(consistent.all() and finite.all() and controls.all()
                    and products.all() and rounded.all())
    failures = []
    for mantissa in (~consistent).nonzero().flatten().tolist():
        row = int((delta[:, mantissa] != delta[0, mantissa])
                  .nonzero()[0, 0])
        index = row * MANTISSAS + mantissa
        failures.append({
            "mantissa": mantissa, "first_different_exponent": row - 120,
            "amax_bf16_bits": f"0x{int(bits[index]):04x}",
            "bias_at_exponent_minus_120": int(delta[0, mantissa]),
            "different_bias": int(delta[row, mantissa]),
            "observed_biases": delta[:, mantissa].unique().tolist(),
        })
    values, counts = delta.unique(return_counts=True)
    return delta, {
        "status": "pass" if complete else "fail",
        "table_supported": complete,
        "bias_definition": "nativeFactorBits - CPUFactorBits (signed int64)",
        "mantissa_bias": delta[0].tolist() if complete else None,
        "coverage": {
            "first_bf16_bits": "0x0380", "last_bf16_bits": "0x7f7f",
            "exponents": [-120, 127], "exponents_per_mantissa": EXPONENTS,
            "mantissas": MANTISSAS, "input_count": bits.numel(),
            "launches": inputs.shape[0], "lanes_per_launch": LANES,
            "consistent_mantissas": int(consistent.sum()),
            "positive_finite_native_factors": int(finite.sum()),
            "literal_controls_passed": int(controls.sum()),
            "literal_controls_total": controls.numel(),
            "product_bitwise_passed": int(products.sum()),
            "product_bitwise_total": products.numel(),
            "product_rint_passed": int(rounded.sum()),
            "product_rint_total": rounded.numel(),
        },
        "bias_histogram": {str(int(v)): int(n)
                           for v, n in zip(values, counts)},
        "inconsistent_mantissas": failures,
    }


@torch.inference_mode()
def run(args):
    bits, inputs, cpu_factor = prepare()
    import torch_npu

    torch.npu.set_device(args.device)
    capacity = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(
        CAPACITY_BYTES / capacity, args.device,
    )
    torch.npu.reset_peak_memory_stats(args.device)
    device_inputs = inputs.to(args.device)
    floats = torch.empty(496, 2, LANES, dtype=torch.float32,
                         device=args.device)
    integers = torch.empty(496, 2, LANES, dtype=torch.int32,
                           device=args.device)
    library = ctypes.CDLL(str(args.library.resolve()))
    launch = library.quant_div_probe_launch
    launch.argtypes = [ctypes.c_void_p] * 4
    launch.restype = None
    stream = torch.npu.current_stream(args.device)
    input_pointer = device_inputs.data_ptr()
    float_pointer, integer_pointer = floats.data_ptr(), integers.data_ptr()
    input_stride = 3 * LANES * inputs.element_size()
    output_stride = 2 * LANES * floats.element_size()
    stream.synchronize()
    for batch in range(inputs.shape[0]):
        launch(stream.npu_stream, input_pointer + batch * input_stride,
               float_pointer + batch * output_stride,
               integer_pointer + batch * output_stride)
    stream.synchronize()
    floats, integers = floats.cpu(), integers.cpu()
    delta, result = analyze(bits, inputs, floats, integers, cpu_factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload_path = args.output.with_suffix(".pt")
    torch.save({
        "amax_bf16_bits": bits, "inputs": inputs,
        "floats": floats, "integers": integers,
        "cpu_factor": cpu_factor,
        "cpu_product": inputs[:, 0].reshape(-1) * cpu_factor,
        "factor_bit_delta": delta,
    }, payload_path)
    result.update({
        "scope": "Native 64-lane AIV Div/Mul/RINT; x=amax/2",
        "float_output_planes": ["factor", "product"],
        "integer_output_planes": ["rint_product", "rint_literal"],
        "device": args.device,
        "device_name": torch.npu.get_device_name(args.device),
        "versions": {"torch": torch.__version__,
                     "torch_npu": torch_npu.__version__},
        "performance_measured": False, "reference_modified": False,
        "allocator_limit_bytes": CAPACITY_BYTES,
        "peak_reserved_bytes": torch.npu.max_memory_reserved(args.device),
        "artifacts": {str(p): digest(p) for p in (
            args.library, payload_path, Path(__file__),
            Path(__file__).with_name("quant_div_probe.cpp"),
            Path(__file__).with_name("CMakeLists.txt"))},
    })
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "status", "table_supported", "coverage", "bias_histogram",
    )}))
    return result["table_supported"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(not run(args))


if __name__ == "__main__":
    main()
