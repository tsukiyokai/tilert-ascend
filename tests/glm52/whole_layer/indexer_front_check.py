"""Compare a native indexer front with CANN ops and the source Triton RoPE."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import statistics

from attention_check import capture, metric, provenance, sha256, time_graph


HERE = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=HERE /
                        "indexer_front_build-scale-order/whole_layer_indexer_front.so")
    parser.add_argument("--native-namespace", default="tilert")
    parser.add_argument("--rope-source", type=Path, default=HERE.parents[2] /
                        "indexer_front_reference/rope.py")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[1])
    parser.add_argument("--blocks", type=int, nargs="+", default=[0])
    parser.add_argument("--modes", nargs="+", choices=["int8", "bf16"],
                        default=["int8", "bf16"])
    parser.add_argument("--query-scale-dtype", choices=["float32", "bfloat16"],
                        default="float32",
                        help="original WQ channel scale dtype for both implementations")
    parser.add_argument("--convert-query-scale-to-fp32", action="store_true",
                        help="diagnose the old adapter's native-only FP32 conversion")
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--queued-producer", action="store_true")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-pairs", type=int, default=6)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(row < 1 or row > 32 for row in args.widths):
        parser.error("widths must lie in [1,32]")
    if any(block < 0 or block > 64 for block in args.blocks):
        parser.error("blocks must lie in [0,64]")
    if min(args.replays, args.iterations, args.timing_pairs) <= 0:
        parser.error("replay and timing counts must be positive")
    if not args.rope_source.is_file():
        parser.error("rope-source must name the recorded vLLM-Ascend rope.py")
    return args


def load_rope(source, destination):
    """Import the unmodified, self-contained source JIT function."""
    text = source.read_text()
    tree = ast.parse(text)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "_triton_rope_siso"]
    if len(functions) != 1:
        raise ValueError("source requires exactly one _triton_rope_siso")
    node = functions[0]
    first = min([node.lineno] + [item.lineno for item in node.decorator_list])
    body = "\n".join(text.splitlines()[first - 1:node.end_lineno]) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "# Source-replayed vLLM-Ascend Apache-2.0 RoPE function.\n"
        "import triton\nimport triton.language as tl\n\n" + body
    )
    spec = importlib.util.spec_from_file_location(
        "indexer_front_rope", destination,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._triton_rope_siso


def fixture(torch, args):
    if args.fixture:
        data = torch.load(args.fixture, map_location="cpu", weights_only=True)
    else:
        generator = torch.Generator().manual_seed(20261021)

        def normal(shape, scale=1.0):
            return (torch.randn(shape, generator=generator) * scale).bfloat16()

        hadamard = torch.ones(1, 1, dtype=torch.bfloat16)
        for _ in range(7):
            upper = torch.cat((hadamard, hadamard), dim=1)
            lower = torch.cat((hadamard, -hadamard), dim=1)
            hadamard = torch.cat((upper, lower), dim=0)
        data = {
            "hidden": normal((32, 6144)),
            "query_norm": torch.randint(-127, 128, (32, 2048),
                generator=generator, dtype=torch.int8),
            "query_norm_scale": (
                torch.rand(32, generator=generator) / 64 + .01
            ),
            "query_weight": torch.randint(-8, 8, (2048, 4096),
                generator=generator, dtype=torch.int8),
            "query_weight_scale":
                (torch.rand(4096, generator=generator) + .25) / 128,
            "key_weight": normal((6144, 160), 1 / 80),
            "gamma": torch.randn(128, generator=generator) / 8 + 1,
            "beta": torch.randn(128, generator=generator) / 8,
            "hadamard": hadamard / (128 ** .5),
        }
    expected = {
        "hidden": ((None, 6144), torch.bfloat16),
        "query_norm": ((None, 2048), torch.int8),
        "query_norm_scale": ((None,), torch.float32),
        "query_weight": ((2048, 4096), torch.int8),
        "query_weight_scale": ((4096,), torch.float32),
        "key_weight": ((6144, 160), torch.bfloat16),
        "gamma": ((128,), torch.float32),
        "beta": ((128,), torch.float32),
        "hadamard": ((128, 128), torch.bfloat16),
    }
    for name, (shape, dtype) in expected.items():
        value = data[name]
        if (not torch.is_tensor(value) or value.dtype != dtype
                or value.dim() != len(shape)
                or any(actual != wanted for actual, wanted in
                       zip(value.shape, shape) if wanted is not None)
                or shape[0] is None and value.shape[0] < max(args.widths)
                or not bool(torch.isfinite(value).all())):
            raise ValueError(
                f"invalid {name}: expected finite {dtype} {shape}"
            )
    return data


def layout(rows):
    def align(size):
        return (size + 511) // 512 * 512

    capacity = 16 if rows <= 16 else 32
    entries = (
        ("query_acc", capacity * 4096 * 4),
        ("key_acc", capacity * 160 * 4),
        ("query_projection", rows * 4096 * 2),
        ("key_projection", rows * 160 * 2),
        ("key_normalized", rows * 128 * 2),
        ("query_rope", rows * 4096 * 2),
        ("key_rope", rows * 128 * 2),
        ("norm_stats", rows * 2 * 4),
        ("query_hadamard", rows * 4096 * 4),
        ("key_hadamard", capacity * 128 * 4),
    )
    offset, result = 0, {}
    for name, size in entries:
        result[name] = (offset, size)
        offset = align(offset + size)
    result["system"] = (offset, 16 << 20)
    return result, offset + (16 << 20)


def run_case(torch, torch_npu, args, data, rope_kernel, rows, mode, requested):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    hidden_base = nd(data["hidden"][:rows])
    query_base = nd(data["query_norm"][:rows])
    scale_base = nd(data["query_norm_scale"][:rows])
    hidden, query_norm, query_scale = (nd(value.clone()) for value in
                                      (hidden_base, query_base, scale_base))
    qweight = nd(data["query_weight"])
    qtiles = nd(qweight.reshape(2048, 16, 256).permute(1, 0, 2))
    qnz = torch_npu.npu_format_cast(qweight, 29)
    vendor_qchannel = nd(data["query_weight_scale"].to(
        getattr(torch, args.query_scale_dtype)))
    qchannel = (nd(vendor_qchannel.float()) if args.convert_query_scale_to_fp32
                else vendor_qchannel)
    kweight = nd(data["key_weight"])
    ktiles = nd(kweight.reshape(6144, 5, 32).permute(1, 0, 2))
    knz = torch_npu.npu_format_cast(kweight.t().contiguous(), 29)
    gamma, beta, hadamard = (nd(data[name]) for name in
                              ("gamma", "beta", "hadamard"))
    cosine = nd(torch.ones(rows, 64, dtype=torch.bfloat16))
    sine = nd(torch.zeros(rows, 64, dtype=torch.bfloat16))
    quantized = mode == "int8"
    native_ops = getattr(torch.ops, args.native_namespace)
    prepared = native_ops.indexer_front_prepare(
        hidden, quantized, requested)
    scratch, tiling, outputs, blocks, sync = prepared
    offsets, expected_bytes = layout(rows)
    if scratch.nbytes != expected_bytes:
        raise ValueError("host and diagnostic scratch layouts disagree")

    def inputs():
        values = (hidden, query_norm, query_scale)
        return tuple(value.clone() for value in values) if (
            args.queued_producer
        ) else values

    def native():
        h, q, scale = inputs()
        return native_ops.indexer_front_out(
            h, q, scale, qtiles, qchannel, ktiles, gamma, beta, cosine, sine,
            hadamard, scratch, tiling, outputs, quantized, blocks, sync,
        )

    def rotate(value):
        value = value.clone().contiguous()
        heads = value.shape[1]
        rope_kernel[(min(rows, 2 * blocks),)](
            value, value.stride(0), cosine, cosine.stride(0),
            sine, sine.stride(0), None, None, None, rows,
            heads, 128, 64, heads, 64, BLOCK_SIZE=heads,
            IS_NEOX_STYLE=False, USE_COS_SIN=False,
        )
        return value

    def vendor():
        h, q, scale = inputs()
        qproj = torch_npu.npu_quant_matmul(
            q, qnz, vendor_qchannel, pertoken_scale=scale, bias=None,
            output_dtype=torch.bfloat16,
        ).reshape(rows, 32, 128)
        kw = torch.nn.functional.linear(h, knz)
        norm, mean, rstd = torch.native_layer_norm(
            kw[:, :128].float(), [128], gamma, beta, 1e-6,
        )
        norm = norm.bfloat16()
        qr = rotate(qproj)
        kr = rotate(norm.unsqueeze(1)).reshape(rows, 128)
        result = {"query_projection": qproj, "key_projection": kw,
                  "key_normalized": norm, "query_rope": qr, "key_rope": kr,
                  "norm_stats": torch.cat((mean, rstd), dim=1)}
        if quantized:
            qh, kh = qr @ hadamard, kr @ hadamard
            qwire, qscale = torch_npu.npu_dynamic_quant(qh.reshape(-1, 128))
            kwire, kscale = torch_npu.npu_dynamic_quant(kh)
            result.update(query_hadamard=qh, key_hadamard=kh,
                          query=qwire.reshape(rows, 32, 128), key=kwire,
                          query_scale=qscale.half().reshape(rows, 32),
                          key_scale=kscale.half(), weights=kw[:, 128:].half())
        else:
            result.update(query=qr, key=kr, weights=kw[:, 128:])
        return result

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))

    def update(replay):
        with torch.npu.stream(stream):
            factor = (1.0, 0.0, -1.0, .5)[replay % 4]
            hidden.copy_((hidden_base.float() * factor).bfloat16())
            query_norm.copy_((query_base.float() * factor).to(torch.int8))
            query_scale.copy_(scale_base * (1 + replay / 8))
            positions = torch.arange(rows, device=args.device) + 8192 + replay
            frequency = torch.arange(32, device=args.device).float() + 1
            angle = positions.float()[:, None] * frequency[None, :] / 1024
            cosine.copy_(angle.cos().repeat(1, 2).bfloat16())
            sine.copy_(angle.sin().repeat(1, 2).bfloat16())
        stream.synchronize()

    update(0)
    if args.capture:
        native_graph, _ = capture(torch, native, stream)
        vendor_graph, reference = capture(torch, vendor, stream)
    checks = []
    for replay in range(args.replays):
        update(replay)
        with torch.npu.stream(stream):
            if args.capture:
                order = (native_graph.replay, vendor_graph.replay)
                for call in order[::1 if replay % 2 == 0 else -1]:
                    call()
            else:
                native()
                reference = vendor()
        stream.synchronize()
        cpu = scratch.cpu()
        shapes = {
            "query_projection": (rows, 32, 128),
            "key_projection": (rows, 160), "key_normalized": (rows, 128),
            "query_rope": (rows, 32, 128), "key_rope": (rows, 128),
            "norm_stats": (rows, 2),
        }
        observed = dict(zip(("query", "key", "weights", "query_scale",
                             "key_scale"), outputs))
        for name, shape in shapes.items():
            start, size = offsets[name]
            dtype = torch.float32 if name == "norm_stats" else torch.bfloat16
            observed[name] = cpu[start:start + size].view(dtype).reshape(shape)
        if quantized:
            for name, shape in (("query_hadamard", (rows, 32, 128)),
                                ("key_hadamard", (rows, 128))):
                start, size = offsets[name]
                count = rows * (4096 if name.startswith("query") else 128)
                observed[name] = cpu[start:start + size].view(torch.float32)[
                    :count].reshape(shape).bfloat16()
        results = {name: metric(torch, observed[name], expected)
                   for name, expected in reference.items()}
        # Internal FP32 norm stats diagnose rounding; the production boundary
        # is the BF16 normalized key, followed by RoPE and cache encoding.
        passed = all(item["pass"] and item["bitwise_equal"]
                     for name, item in results.items() if name != "norm_stats")
        checks.append({"replay": replay, "pass": passed,
                       "boundaries": results})
        if not passed:
            print(json.dumps({"event": "boundary_failure", "rows": rows,
                              "mode": mode, "replay": replay,
                              "boundaries": results}), flush=True)
            dump = args.output.with_name(
                f"{args.output.stem}-{mode}-m{rows}-r{replay}.pt"
            )

            def retained(value):
                return value.detach().cpu().clone(
                    memory_format=torch.contiguous_format)

            failure_inputs = dict(data)
            failure_inputs.update(
                hidden=hidden, query_norm=query_norm,
                query_norm_scale=query_scale, cosine=cosine, sine=sine,
                query_weight_scale=vendor_qchannel,
                native_query_weight_scale=qchannel,
            )
            payload = {
                "scratch": retained(cpu),
                "observed": {name: retained(value)
                             for name, value in observed.items()},
                "reference": {name: retained(value)
                              for name, value in reference.items()},
                "inputs": {name: retained(value)
                           for name, value in failure_inputs.items()},
                "config": {"rows": rows, "blocks": blocks, "mode": mode,
                           "query_scale_dtype": args.query_scale_dtype,
                           "convert_query_scale_to_fp32":
                               args.convert_query_scale_to_fp32},
            }
            failure = {"rows": rows, "mode": mode, "blocks": blocks,
                       "pass": False, "checks": checks,
                       "failure_dump": str(dump)}
            try:
                torch.save(payload, dump)
            except Exception as error:
                failure["failure_dump_error"] = (
                    f"{type(error).__name__}: {error}"
                )
            return failure
    report = {"rows": rows, "mode": mode, "blocks": blocks, "pass": True,
              "checks": checks, "scratch_bytes": scratch.nbytes,
              "query_scale_dtypes": {"native": str(qchannel.dtype),
                                     "vendor": str(vendor_qchannel.dtype)},
              "vendor_formats": {"q": torch_npu.get_npu_format(qnz),
                                 "kw": torch_npu.get_npu_format(knz)}}
    if args.capture and not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            order = (("native", native_graph), ("vendor", vendor_graph))
            pairs.append({
                name: time_graph(torch, graph, stream, args.iterations)
                for name, graph in order[::1 if pair % 2 == 0 else -1]
            })
        report["timing"] = {"pairs_us": pairs, "median_us": {
            name: statistics.median(pair[name] for pair in pairs)
            for name in ("native", "vendor")}}
    return report


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    source_replay = args.output.with_name(args.output.stem + "-rope.py")
    rope_kernel = load_rope(args.rope_source, source_replay)
    data = fixture(torch, args)
    manifest = args.library.parent / "build-manifest.json"
    report = {
        "scope": "native indexer front with retained intermediate boundaries",
        "native_namespace": args.native_namespace,
        "provenance": provenance(torch, torch_npu, args),
        "front_harness_sha256": sha256(Path(__file__)),
        "rope_source": str(args.rope_source),
        "rope_source_sha256": sha256(args.rope_source),
        "rope_source_replay_sha256": sha256(source_replay),
        "build_manifest": json.loads(manifest.read_text())
            if manifest.exists() else None,
        "capture": args.capture, "queued_producer": args.queued_producer,
        "query_scale_contract": {
            "vendor_dtype": args.query_scale_dtype,
            "native_dtype": ("float32" if args.convert_query_scale_to_fp32
                             else args.query_scale_dtype),
            "values": "fixture rounded to the original vendor dtype",
            "native_fp32_conversion": args.convert_query_scale_to_fp32,
        },
        "cases": [], "pass": False,
    }
    try:
        with torch.inference_mode():
            for mode in args.modes:
                for rows in args.widths:
                    for blocks in args.blocks:
                        result = run_case(torch, torch_npu, args, data,
                                          rope_kernel, rows, mode, blocks)
                        report["cases"].append(result)
                        print(json.dumps({"rows": rows, "mode": mode,
                                          "pass": result["pass"]}), flush=True)
                        if not result["pass"]:
                            raise RuntimeError(
                                "indexer front differential failed"
                            )
        report["pass"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
