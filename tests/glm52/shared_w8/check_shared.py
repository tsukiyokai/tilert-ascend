"""Compare the TP16 shared W8A8 body with the installed CANN chain."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mla_tail"))
from check import capture, event_time_us, metrics  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=HERE / "build/shared_w8.so")
    parser.add_argument("--reference-source-library", type=Path,
                        help="unmodified vLLM SwiGLU kernel with fixed-shape tiling")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 12, 32])
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-pairs", type=int, default=5)
    parser.add_argument("--queued-producer", action="store_true")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(row < 1 or row > 32 for row in args.widths):
        parser.error("widths must lie in [1,32]")
    if args.blocks not in range(1, 25) or args.replays < 4:
        parser.error("use 1..24 core groups and at least four changing inputs")
    if min(args.iterations, args.timing_pairs) <= 0:
        parser.error("timing counts must be positive")
    return args


def load_library(path):
    library = ctypes.CDLL(str(path.resolve()))
    library.shared_w8_prepare.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.shared_w8_prepare.restype = ctypes.c_int
    library.shared_w8_workspace_field.argtypes = [ctypes.c_int, ctypes.c_int]
    library.shared_w8_workspace_field.restype = ctypes.c_uint64
    library.shared_w8_launch.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64]
        + [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 2
    )
    library.shared_w8_launch.restype = None
    return library


def load_source_reference(path):
    library = ctypes.CDLL(str(path.resolve()))
    launch = library.shared_swiglu_reference_launch
    launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 5
    launch.argtypes += [ctypes.c_int64]
    launch.restype = None
    return library


def run_case(torch, torch_npu, library, args, rows, reference=None):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    generator = torch.Generator().manual_seed(args.seed)
    gu_cpu = torch.randint(-128, 128, (6144, 256), dtype=torch.int8,
                           generator=generator)
    down_cpu = torch.randint(-128, 128, (128, 6144), dtype=torch.int8,
                             generator=generator)
    gu_tiles = nd(gu_cpu.reshape(6, 1024, 4, 64).permute(0, 2, 1, 3)
                  .contiguous())
    down_tiles = nd(down_cpu.reshape(128, 24, 256).permute(1, 0, 2).contiguous())
    gu_nz = torch_npu.npu_format_cast(nd(gu_cpu), 29)
    down_nz = torch_npu.npu_format_cast(nd(down_cpu), 29)
    gu_scales = nd(torch.rand(256, generator=generator) * 0.001 + 0.0005)
    down_scales = nd(torch.rand(6144, generator=generator) * 0.001 + 0.0005)
    source = torch.randint(-128, 128, (rows, 6144), generator=generator,
                           dtype=torch.int8)
    token_scale = torch.rand(rows, generator=generator) * 0.02 + 0.005
    inputs, input_scales = nd(source), nd(token_scale)
    capacity = 16 if rows <= 16 else 32
    tiling_bytes = ctypes.create_string_buffer(1024)
    sync = ctypes.c_uint64()
    status = library.shared_w8_prepare(
        capacity, args.blocks, tiling_bytes, ctypes.byref(sync),
    )
    if status:
        raise RuntimeError(f"shared_w8_prepare returned {status}")
    tiling = nd(torch.frombuffer(bytearray(tiling_bytes.raw), dtype=torch.uint8))
    layout = [library.shared_w8_workspace_field(capacity, field)
              for field in range(7)]
    scratch = nd(torch.empty(layout[-1], dtype=torch.uint8))
    output = nd(torch.empty(rows, 6144, dtype=torch.bfloat16))
    reference_codes = nd(torch.empty(rows, 128, dtype=torch.int8))
    reference_scales = nd(torch.empty(rows, dtype=torch.float32))

    def current():
        if args.queued_producer:
            return inputs.clone(), input_scales.clone()
        return inputs, input_scales

    def native():
        codes, scales = current()
        stream_pointer = torch.npu.current_stream(args.device).npu_stream
        addresses = (codes, scales, gu_tiles, gu_scales, down_tiles,
                     down_scales, scratch, tiling, output)
        library.shared_w8_launch(args.blocks, stream_pointer, sync.value,
            *(tensor.data_ptr() for tensor in addresses), rows, capacity)
        return output

    def vendor():
        codes, scales = current()
        product = torch_npu.npu_quant_matmul(
            codes, gu_nz, gu_scales, output_dtype=torch.int32,
        )
        if reference is None:
            middle, middle_scale = torch_npu.npu_dequant_swiglu_quant(
                product, weight_scale=gu_scales, activation_scale=scales,
                activate_left=True, quant_mode=1, swiglu_mode=0,
            )
        else:
            addresses = (product, gu_scales, scales, reference_codes,
                         reference_scales)
            reference.shared_swiglu_reference_launch(
                min(rows, 48), torch.npu.current_stream(args.device).npu_stream,
                *(tensor.data_ptr() for tensor in addresses), rows,
            )
            middle, middle_scale = reference_codes, reference_scales
        result = torch_npu.npu_quant_matmul(
            middle, down_nz, down_scales, pertoken_scale=middle_scale,
            output_dtype=torch.bfloat16,
        )
        return product, middle, middle_scale, result

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    native_graph, native_output = capture(torch, native, stream)
    vendor_graph, vendor_outputs = capture(torch, vendor, stream)
    checks = []
    for replay in range(args.replays):
        changed = source.clone()
        if replay % 4 == 1:
            changed.zero_()
        elif replay % 4 == 2:
            changed = (-source.to(torch.int16)).clamp(-128, 127).to(torch.int8)
        elif replay % 4 == 3:
            changed[:, ::2] = -128
            changed[:, 1::2] = 127
        if replay >= 4:
            changed = changed.roll(replay, dims=-1)
        with torch.npu.stream(stream):
            inputs.copy_(changed)
            input_scales.copy_(token_scale * (1 + replay * 0.125))
            order = (native_graph.replay, vendor_graph.replay)
            for call in order[::1 if replay % 2 == 0 else -1]:
                call()
        stream.synchronize()
        partial = scratch[layout[1]:layout[1] + 6 * 256 * capacity * 4]
        partial = partial.cpu().view(torch.int32).reshape(6, 4, capacity, 64)
        gate = partial.sum(dim=0).permute(1, 0, 2).reshape(capacity, 256)
        middle = scratch[layout[2]:layout[2] + capacity * 128]
        middle = middle.view(torch.int8).reshape(capacity, 128)[:rows]
        scales = scratch[layout[3]:layout[3] + capacity * 8 * 4]
        scales = scales.view(torch.float32).reshape(capacity, 8)[:rows, 0]
        boundaries = {
            "gate_up_int32": metrics(torch, gate[:rows].int(),
                                      vendor_outputs[0].cpu(), 0, 0),
            "middle_codes": metrics(torch, middle, vendor_outputs[1], 0, 0),
            "middle_scale": metrics(torch, scales, vendor_outputs[2], 0, 0),
            "output": metrics(torch, native_output, vendor_outputs[3], 0, 0),
        }
        passed = all(value["pass"] and value["bitwise_equal"]
                     for value in boundaries.values())
        checks.append({"replay": replay, "pass": passed,
                       "boundaries": boundaries})
        if not passed:
            diagnostic = args.output.with_name(
                f"{args.output.stem}-m{rows}-r{replay}.pt")
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "gate_up": gate[:rows].int(), "gate_scales": gu_scales.cpu(),
                "input_scales": input_scales.cpu(),
                "native_middle": middle.cpu(),
                "vendor_middle": vendor_outputs[1].cpu(),
                "native_scale": scales.cpu(),
                "vendor_scale": vendor_outputs[2].cpu(),
                "native_output": native_output.cpu(),
                "vendor_output": vendor_outputs[3].cpu(),
                "down_weight": down_cpu,
                "down_scale": down_scales.cpu(),
                "native_down_accumulators": scratch[
                    layout[4]:layout[4] + capacity * 6144 * 4
                ].cpu().view(torch.int32).reshape(24, capacity, 256)
                 .permute(1, 0, 2).reshape(capacity, 6144)[:rows],
            }, diagnostic)
            return {"rows": rows, "pass": False, "checks": checks}
    timing = None
    if not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            order = (("native", native_graph), ("vendor", vendor_graph))
            pairs.append({name: event_time_us(torch, graph, args.iterations, stream)
                          for name, graph in order[::1 if pair % 2 == 0 else -1]})
        timing = {"pairs_us": pairs,
                  "median_us": {name: statistics.median(pair[name] for pair in pairs)
                                for name in ("native", "vendor")}}
    return {"rows": rows, "capacity": capacity, "blocks": args.blocks,
            "scratch_bytes": layout[-1], "pass": True, "checks": checks,
            "timing": timing,
            "vendor_weight_formats": [torch_npu.get_npu_format(gu_nz),
                                      torch_npu.get_npu_format(down_nz)]}


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    library = load_library(args.library)
    reference = (load_source_reference(args.reference_source_library)
                 if args.reference_source_library else None)
    contract = getattr(library, "shared_w8_grouped_quant", None)
    grouped = bool(contract()) if contract else False
    if grouped != (reference is not None):
        raise ValueError("Candidate activation contract does not match reference")
    manifest = args.library.parent / "build-manifest.json"
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "shared W8A8 TP16 shard from quantized input through local down",
        "reference": (
            "stock CANN QuantMatmul INT32 -> unmodified vLLM source SwiGLU "
            "mode1 with fixed-shape tiling -> stock CANN QuantMatmul BF16"
            if reference is not None else
            "stock CANN QuantMatmul INT32 -> DequantSwigluQuant "
            "mode0 dynamic -> QuantMatmul BF16"
        ),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": str(torch.npu.get_device_properties(args.device)),
        "library_sha256": digest(args.library),
        "harness_sha256": digest(Path(__file__)),
        "helpers_sha256": digest(HERE.parent / "mla_tail/check.py"),
        "build_manifest": json.loads(manifest.read_text()),
        "cases": [], "pass": False,
    }
    if reference is not None:
        report["reference_library_sha256"] = digest(args.reference_source_library)
        reference_manifest = args.reference_source_library.parent / "build-manifest.json"
        report["reference_build_manifest"] = json.loads(reference_manifest.read_text())
    try:
        with torch.inference_mode():
            for rows in args.widths:
                result = run_case(torch, torch_npu, library, args, rows, reference)
                report["cases"].append(result)
                print(json.dumps({"rows": rows, "pass": result["pass"]}), flush=True)
                if not result["pass"]:
                    raise RuntimeError("shared W8A8 differential failed")
        report["pass"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
