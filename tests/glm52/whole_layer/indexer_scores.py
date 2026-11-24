"""Inspect C8 scores from a saved wire fixture and live merge-list metadata."""

import argparse
import hashlib
import json
from pathlib import Path


TOPK = 2048
S1_BASE = 4


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_scores(torch, fixture, row, indices):
    ids = torch.tensor(indices, dtype=torch.int64)
    pages = fixture["block_table"][0, ids // 128].long()
    keys = fixture["key_cache"][pages, ids % 128, 0].long()
    key_scale = fixture["scale_cache"][pages, ids % 128].float()
    dots = fixture["query"][row].long() @ keys.T
    weights = (fixture["weights"][row].float() *
               fixture["query_scale"][row].float()).half()
    scores_half = (dots.float() / 1024).clamp_min(0).half()
    products = weights.double()[:, None] * scores_half.double()
    sequential = torch.zeros(len(indices))
    for head in range(32):
        sequential = sequential + products[head].float()
    sums = {
        "exact_products_sum_then_fp32": products.sum(0).float(),
        "torch_sum_fp32": products.float().sum(0),
        "torch_matmul_fp32": weights.float().reshape(1, 32) @
            scores_half.float(),
        "sequential_fp32": sequential,
    }
    methods = {}
    for name, summed in sums.items():
        scores = summed.reshape(-1) * key_scale
        methods[name] = {"head_sum": summed.reshape(-1).tolist(),
                         "scores": scores.tolist(),
                         "score_bits": scores.view(torch.int32).tolist()}
    return {
        "scope": "CPU arithmetic with wire FP16 boundaries; no device oracle",
        "physical_pages": pages.tolist(), "key_scale": key_scale.tolist(),
        "dot_i32": dots.tolist(),
        "weight_times_query_scale_f16_bits": (
            weights.view(torch.int16).tolist()
        ),
        "relu_dot_div_1024_f16_bits": scores_half.view(torch.int16).tolist(),
        "subnormal_weight_products": int(
            ((weights.abs() > 0) & (weights.abs() < 2**-14)).sum()
        ),
        "methods": methods,
    }


def retained_scores(torch, fixture, scratch, blocks, row, indices):
    if scratch.dtype != torch.uint8 or not scratch.is_contiguous():
        raise ValueError("native scratch must be contiguous UINT8")
    scratch = scratch.reshape(-1)
    mm_bytes = blocks * 2 * S1_BASE * 2048 * 4
    merge_bytes = blocks * S1_BASE * 2 * 2 * TOPK * 4
    params_offset = mm_bytes + merge_bytes
    params_bytes = blocks * S1_BASE * 2 * 16 * 8
    weights_bytes = blocks * 16 * S1_BASE * 32 * 2 * 2
    if scratch.numel() != params_offset + params_bytes + weights_bytes:
        raise ValueError("scratch length differs from C8 geometry")
    lists_raw = scratch[mm_bytes:params_offset]
    values = lists_raw.view(torch.float32).reshape(blocks, S1_BASE, 2, TOPK, 2)
    list_indices = lists_raw.view(torch.int32).reshape(
        blocks, S1_BASE, 2, TOPK, 2,
    )[..., 1]
    params = scratch[params_offset:params_offset + params_bytes]
    params = params.view(torch.int64).reshape(blocks, S1_BASE, 2, 16)
    rows = fixture["query"].shape[0]
    causal_length = int(fixture["key_lengths"][0]) - rows + row + 1
    records = []
    for core in range(blocks):
        for lane in range(S1_BASE):
            for side in range(2):
                meta = params[core, lane, side].tolist()
                if meta[0] != 1 or meta[6] != row:
                    continue
                start, end = meta[2] * 2048, meta[3]
                if (meta[1] != causal_length or meta[5] != 0 or
                        meta[8] != row * TOPK or meta[4] not in (0, 1) or
                        not 1 <= meta[7] <= S1_BASE or
                        not 0 <= start < end <= causal_length):
                    raise ValueError("inconsistent retained-list metadata")
                entries = []
                stored = list_indices[core, lane, side]
                for index in indices:
                    positions = (stored == index).nonzero().reshape(-1)
                    if positions.numel() > 1:
                        raise ValueError("duplicate retained index")
                    if positions.numel() == 0:
                        continue
                    if not start <= index < end:
                        raise ValueError("retained index outside its range")
                    position = int(positions[0])
                    value = values[core, lane, side, position, 0]
                    if not bool(torch.isfinite(value)):
                        raise ValueError("nonfinite retained score")
                    entries.append({
                        "index": index, "position": position,
                        "score": float(value),
                        "score_bits": int(value.view(torch.int32)),
                    })
                records.append({"core": core, "lane": lane, "side": side,
                                "params": meta[:9], "entries": entries})
    recovered = {entry["index"] for record in records
                 for entry in record["entries"]}
    return {"scope": "device FP32 scores in metadata-qualified partial lists",
            "records": records, "all_requested_recovered":
                recovered == set(indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if fixture["query"].dtype != torch.int8:
        raise ValueError("this diagnostic requires C8 INT8 wire inputs")
    rows = fixture["query"].shape[0]
    causal_length = int(fixture["key_lengths"][0]) - rows + args.row + 1
    if (args.row < 0 or args.row >= rows or args.blocks < 1 or
            any(index < 0 or index >= causal_length
                for index in args.indices)):
        raise ValueError("invalid row, blocks or causal index")
    sources = {}
    if args.vendor:
        source = args.vendor / "attention/quant_lightning_indexer/op_kernel"
        for name in (
            "quant_lightning_indexer_kernel_arch22.h",
            "quant_lightning_indexer_service_cube_arch22.h",
            "quant_lightning_indexer_service_vector_arch22.h",
            "quant_lightning_indexer_vector.h",
        ):
            path = source / "arch22" / name
            sources[str(path)] = sha256(path)
    report = {
        "fixture_sha256": sha256(args.fixture),
        "harness_sha256": sha256(Path(__file__)),
        "source_sha256": sources, "row": args.row, "indices": args.indices,
        "cpu": cpu_scores(torch, fixture, args.row, args.indices),
        "device_retained": None,
    }
    if "native_scratch" in fixture:
        report["device_retained"] = retained_scores(
            torch, fixture, fixture["native_scratch"], args.blocks,
            args.row, args.indices,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
