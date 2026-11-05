"""Compare native cache-write/selector with same-input CANN operators."""

import argparse
import json
from pathlib import Path
import statistics

from attention_check import capture, provenance, sha256, time_graph


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent /
                        "indexer_build/whole_layer_indexer.so")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--widths", type=int, nargs="+", default=[1])
    parser.add_argument("--blocks", type=int, nargs="+", default=[0])
    parser.add_argument("--modes", nargs="+", choices=["int8", "bf16"],
                        default=["int8", "bf16"])
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--queued-producer", action="store_true")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timing-pairs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(width < 1 or width > 32 for width in args.widths):
        parser.error("widths must lie in [1,32]")
    if any(blocks < 0 or blocks > 64 for blocks in args.blocks):
        parser.error("blocks must lie in [0,64]; zero selects the default")
    if min(args.replays, args.iterations, args.timing_pairs) <= 0:
        parser.error("replays, iterations and timing-pairs must be positive")
    if args.sequence < 0 or args.sequence > 131072:
        parser.error("sequence must lie in [0,131072]")
    args.fixture = None
    return args


def exact(torch, actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    return {
        "pass": bool(torch.equal(actual_bytes, expected_bytes)),
        "numerical_equal": bool(torch.equal(actual, expected)),
        "different_elements": int((actual != expected).sum()),
        "different_bytes": int((actual_bytes != expected_bytes).sum()),
        "elements": actual.numel(),
    }


def index_differences(torch, actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    positions = (actual != expected).nonzero()
    rows = positions[:, 0].unique().tolist()
    return {
        "coordinates": [
            {"position": position, "native": int(actual[tuple(position)]),
             "vendor": int(expected[tuple(position)])}
            for position in positions[:64].tolist()
        ],
        "truncated": positions.shape[0] > 64,
        "row_multiset_equal": {
            str(row): bool(torch.equal(actual[row].sort().values,
                                      expected[row].sort().values))
            for row in rows
        },
    }


def run_case(torch, torch_npu, args, mode, rows, requested_blocks):
    quantized = mode == "int8"
    dtype = torch.int8 if quantized else torch.bfloat16
    weight_dtype = torch.float16 if quantized else torch.bfloat16
    table_pages = (args.sequence + rows + args.replays + 127) // 128
    pages = table_pages + 3
    generator = torch.Generator().manual_seed(527 + rows)

    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    def empty(shape, tensor_dtype):
        return nd(torch.empty(shape, dtype=tensor_dtype))

    def random_keys(shape):
        if quantized:
            return torch.randint(-127, 128, shape, generator=generator,
                                 dtype=dtype)
        return torch.randn(shape, generator=generator).to(dtype)

    base = nd(random_keys((pages, 128, 1, 128)))
    native_cache, vendor_cache = nd(base.clone()), nd(base.clone())
    query, keys = empty((rows, 32, 128), dtype), empty((rows, 128), dtype)
    weights = empty((rows, 32), weight_dtype)
    slots = empty((rows,), torch.int64)
    q_lengths = nd(torch.tensor([rows], dtype=torch.int32))
    k_lengths = empty((1,), torch.int32)
    table = empty((1, table_pages), torch.int32)
    output = empty((rows, 1, 2048), torch.int32)
    q_scale = empty((rows, 32), torch.float16) if quantized else None
    k_scale = empty((rows,), torch.float16) if quantized else None
    scale_base = nd((torch.rand(pages, 128, generator=generator) / 64 +
                     1 / 64).to(torch.float16)) if quantized else None
    native_scale = nd(scale_base.clone()) if quantized else None
    vendor_scale = nd(scale_base.clone()) if quantized else None
    scratch, blocks, sync_base = torch.ops.tilert.indexer_prepare(
        query, pages, table_pages, requested_blocks,
    )

    def native():
        q = query.clone() if args.queued_producer else query
        return torch.ops.tilert.indexer_out(
            q, keys, weights, q_scale, k_scale, slots, native_cache,
            native_scale, q_lengths, k_lengths, table, scratch, output,
            blocks, sync_base,
        )

    def vendor():
        q = query.clone() if args.queued_producer else query
        vendor_cache.view(-1, 128).index_copy_(0, slots, keys)
        if quantized:
            vendor_scale.view(-1).index_copy_(0, slots, k_scale)
            return torch_npu.npu_quant_lightning_indexer(
                query=q, key=vendor_cache, weights=weights,
                query_dequant_scale=q_scale,
                key_dequant_scale=vendor_scale.view(pages, 128, 1),
                actual_seq_lengths_query=q_lengths,
                actual_seq_lengths_key=k_lengths, block_table=table,
                query_quant_mode=0, key_quant_mode=0, layout_query="TND",
                layout_key="PA_BSND", sparse_count=2048, sparse_mode=3,
                pre_tokens=2**63 - 1, next_tokens=2**63 - 1,
            )
        return torch_npu.npu_lightning_indexer(
            query=q, key=vendor_cache, weights=weights,
            actual_seq_lengths_query=q_lengths,
            actual_seq_lengths_key=k_lengths, block_table=table,
            layout_query="TND", layout_key="PA_BSND", sparse_count=2048,
            sparse_mode=3,
        )[0]

    def update(replay):
        prefix = args.sequence + replay
        physical = torch.randperm(pages, generator=generator)[:table_pages]
        positions = torch.arange(prefix, prefix + rows)
        physical_slots = physical[positions // 128] * 128 + positions % 128
        query.copy_(nd(random_keys(query.shape)))
        keys.copy_(nd(random_keys(keys.shape)))
        weights.copy_(nd(torch.randn(rows, 32, generator=generator).to(
            weight_dtype
        )))
        slots.copy_(nd(physical_slots))
        table.copy_(nd(physical.to(torch.int32).unsqueeze(0)))
        k_lengths.copy_(nd(torch.tensor([prefix + rows], dtype=torch.int32)))
        native_cache.copy_(base)
        vendor_cache.copy_(base)
        if quantized:
            q_scale.copy_(nd((torch.rand(rows, 32, generator=generator) / 64 +
                             1 / 64).to(torch.float16)))
            k_scale.copy_(nd((torch.rand(rows, generator=generator) / 64 +
                             1 / 64).to(torch.float16)))
            native_scale.copy_(scale_base)
            vendor_scale.copy_(scale_base)
        torch.npu.synchronize()

    update(0)
    stream = torch.npu.Stream(device=args.device)
    native_graph = vendor_graph = None
    if args.capture:
        native_graph, native_result = capture(torch, native, stream)
        vendor_graph, vendor_result = capture(torch, vendor, stream)
    results = []
    for replay in range(args.replays):
        update(replay)
        with torch.npu.stream(stream):
            if args.capture:
                native_graph.replay()
                vendor_graph.replay()
            else:
                native_result, vendor_result = native(), vendor()
        stream.synchronize()
        checks = {
            "indices": exact(torch, native_result, vendor_result),
            "key_cache": exact(torch, native_cache, vendor_cache),
        }
        if quantized:
            checks["scale_cache"] = exact(torch, native_scale, vendor_scale)
        result = {"replay": replay, "checks": checks,
                  "pass": all(check["pass"] for check in checks.values())}
        if not result["pass"]:
            result["index_differences"] = index_differences(
                torch, native_result, vendor_result,
            )
            if args.output:
                failure_path = args.output.with_name(
                    f"{args.output.stem}-{mode}-m{rows}-r{replay}.pt"
                )
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                tensors = {
                    "query": query, "weights": weights,
                    "key_cache": vendor_cache, "query_scale": q_scale,
                    "scale_cache": vendor_scale, "block_table": table,
                    "query_lengths": q_lengths, "key_lengths": k_lengths,
                    "native_indices": native_result,
                    "vendor_indices": vendor_result,
                    "native_scratch": scratch,
                }
                saved = {name: value.detach().cpu() if value is not None
                         else None for name, value in tensors.items()}
                saved["blocks"] = blocks
                torch.save(saved, failure_path)
                result["failure_tensors"] = str(failure_path)
            result["first_native_indices"] = (
                native_result[0, 0, :16].cpu().tolist()
            )
            result["first_vendor_indices"] = (
                vendor_result[0, 0, :16].cpu().tolist()
            )
        results.append(result)
        if not result["pass"]:
            break
    passed = len(results) == args.replays and all(x["pass"] for x in results)
    timing = None
    if passed and args.capture and not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            order = [("native", native_graph), ("vendor", vendor_graph)]
            if pair % 2:
                order.reverse()
            times = {name: time_graph(torch, graph, stream, args.iterations)
                     for name, graph in order}
            pairs.append(times)
        timing = {
            "unit": "us_per_replay", "pairs": pairs,
            "median_native": statistics.median(x["native"] for x in pairs),
            "median_vendor": statistics.median(x["vendor"] for x in pairs),
            "iterations": args.iterations,
        }
    return {"mode": mode, "rows": rows, "blocks": blocks,
            "prefix": args.sequence, "workspace_bytes": scratch.numel(),
            "pass": passed, "replays": results, "timing": timing}


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    report = {
        "scope": "same-wire cache-write plus selector; no Q/K preprocessing",
        "capture": args.capture, "queued_producer": args.queued_producer,
        "provenance": provenance(torch, torch_npu, args), "cases": [],
    }
    report["provenance"]["harness_sha256"] = sha256(Path(__file__))
    try:
        for mode in args.modes:
            for rows in args.widths:
                for blocks in args.blocks:
                    case = run_case(
                        torch, torch_npu, args, mode, rows, blocks,
                    )
                    report["cases"].append(case)
                    print(json.dumps(case), flush=True)
                    if not case["pass"]:
                        raise RuntimeError("indexer differential failed")
        report["pass"] = True
    except Exception as error:
        report["pass"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
