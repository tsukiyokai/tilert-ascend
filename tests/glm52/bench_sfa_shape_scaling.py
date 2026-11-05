"""Time one attention path at one query-head and top-k shape.

MLA keeps a single KV head, so holding top-k fixed while raising the head
count leaves HBM traffic unchanged and isolates device occupancy; holding
heads fixed while raising top-k isolates bandwidth. The gather kernel times
the scattered read alone, which separates the read rate from the fused
kernel's internal loop. One shape per process so an unsupported shape faults
alone. One device, one rank, no model.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

import torch

LATENT_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 128
SEQUENCE_LENGTH = 8192
SCALE = 1 / 16


def sparse_attention():
    """Return the vLLM-Ascend CANN binding, registering it on first use."""
    namespace = torch.ops._C_ascend
    if not hasattr(namespace, "npu_sparse_flash_attention"):
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("the A3 sparse attention binding is absent")
    return namespace.npu_sparse_flash_attention


def selected_positions(top_k, width, generator):
    rows = [torch.randperm(SEQUENCE_LENGTH, generator=generator)[:top_k]
            for _ in range(width)]
    return torch.stack([row.sort().values for row in rows]).to(torch.int32)


def build_case(heads, top_k, width, generator):
    pages = SEQUENCE_LENGTH // PAGE_SIZE

    def paged(dim):
        flat = torch.randn(pages * PAGE_SIZE, dim, generator=generator)
        return flat.to(torch.bfloat16).view(pages, PAGE_SIZE, 1, dim)

    def query(dim):
        return torch.randn(width, heads, dim,
                           generator=generator).to(torch.bfloat16)

    key = paged(LATENT_DIM)
    return {
        "query": query(LATENT_DIM),
        "query_rope": query(ROPE_DIM),
        "key": key,
        "value": key,
        "key_rope": paged(ROPE_DIM),
        "sparse_indices": selected_positions(top_k, width,
                                             generator).unsqueeze(1),
        "block_table": torch.arange(pages, dtype=torch.int32).unsqueeze(0),
        "actual_seq_lengths_query": torch.tensor([width], dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor([SEQUENCE_LENGTH],
                                              dtype=torch.int32),
    }


def sfa_call(case, heads):
    """Time the fused CANN sparse kernel, whose gather is internal."""
    operation = sparse_attention()

    def call():
        return operation(
            **case, scale_value=SCALE, sparse_block_size=1,
            layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=3, attention_mode=2, return_softmax_lse=False,
        )

    return call


def physical_rows(case):
    """Flatten the selected positions; the synthetic block table is identity."""
    return case["sparse_indices"].reshape(-1).long()


def gather_call(case, heads):
    """Time the scattered read alone, without any attention arithmetic."""
    rows = physical_rows(case)
    latent = case["key"].view(-1, LATENT_DIM)
    rope = case["key_rope"].view(-1, ROPE_DIM)

    def call():
        return latent.index_select(0, rows), rope.index_select(0, rows)

    return call


def gather_fia_call(case, heads):
    """Time an explicit gather feeding the dense fused attention kernel."""
    import torch_npu

    rows = physical_rows(case)
    width, top_k = case["query"].shape[0], rows.numel() // case[
        "query"].shape[0]
    latent = case["key"].view(-1, LATENT_DIM)
    rope = case["key_rope"].view(-1, ROPE_DIM)
    query = case["query"].view(width, heads, 1, LATENT_DIM)
    query_rope = case["query_rope"].view(width, heads, 1, ROPE_DIM)

    def call():
        key = latent.index_select(0, rows).view(width, 1, top_k, LATENT_DIM)
        key_rope = rope.index_select(0, rows).view(width, 1, top_k, ROPE_DIM)
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, key, query_rope=query_rope, key_rope=key_rope,
            num_query_heads=heads, num_key_value_heads=1, softmax_scale=SCALE,
            input_layout="BNSD", sparse_mode=0, block_table=None, block_size=0,
            actual_seq_qlen=[1] * width, actual_seq_kvlen=[top_k] * width,
        )
        return output

    return call


def shared_fia_call(case, heads):
    """Time one shared read of the whole cache under a per-row mask.

    The gather kernels read `top_k` rows for every query, so their bytes grow
    with the verify width; this reads the cache once and excludes the
    unselected columns instead, so they do not. The mask is built outside the
    timed call because the engine refreshes it on index producers only and the
    consumer layers reuse it (`cann_attention.py:SharedKVAttention`).
    """
    import torch_npu

    width = case["query"].shape[0]
    latent = case["key"].view(1, SEQUENCE_LENGTH, 1, LATENT_DIM)
    rope = case["key_rope"].view(1, SEQUENCE_LENGTH, 1, ROPE_DIM)
    query = case["query"].view(1, width, heads, LATENT_DIM)
    query_rope = case["query_rope"].view(1, width, heads, ROPE_DIM)
    bitmap = torch.ones((width, SEQUENCE_LENGTH), dtype=torch.uint8,
                        device=latent.device)
    bitmap.scatter_(1, case["sparse_indices"].reshape(width, -1).long(), 0)
    mask = bitmap.view(torch.bool).view(1, 1, width, SEQUENCE_LENGTH)

    def call():
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query, latent, latent, query_rope=query_rope, key_rope=rope,
            atten_mask=mask, num_query_heads=heads, num_key_value_heads=1,
            softmax_scale=SCALE, input_layout="BSND", sparse_mode=0,
            block_table=None, block_size=0,
            actual_seq_qlen=[width], actual_seq_kvlen=[SEQUENCE_LENGTH],
        )
        return output

    return call


def per_row_read(width, top_k):
    """Every query reads its own selected rows."""
    return width * top_k * (LATENT_DIM + ROPE_DIM) * 2


def whole_cache_read(width, top_k):
    """One shared read, so the bytes do not grow with the width."""
    return SEQUENCE_LENGTH * (LATENT_DIM + ROPE_DIM) * 2


KERNELS = {"sfa": (sfa_call, per_row_read),
           "gather": (gather_call, per_row_read),
           "gather_fia": (gather_fia_call, per_row_read),
           "shared_fia": (shared_fia_call, whole_cache_read)}


def batch_mean_ms(call, calls):
    torch.npu.synchronize()
    start = perf_counter_ns()
    for _ in range(calls):
        call()
    torch.npu.synchronize()
    return (perf_counter_ns() - start) / calls / 1e6


def measure(args, report):
    import torch_npu

    torch_npu.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    generator = torch.Generator().manual_seed(args.seed)
    case = build_case(args.heads, args.top_k, args.width, generator)
    case = {key: value.to(device) for key, value in case.items()}
    factory, read_model = KERNELS[args.kernel]
    call = factory(case, args.heads)

    for _ in range(args.warmup):
        call()
    samples = sorted(batch_mean_ms(call, args.calls)
                     for _ in range(args.batches))

    median = statistics.median(samples)
    read = read_model(args.width, args.top_k)
    report.update({
        "status": "pass", "kv_bytes": read,
        "median_us": median * 1e3,
        "min_us": samples[0] * 1e3,
        "p90_us": samples[-1 - len(samples) // 10] * 1e3,
        "gb_per_s": read / (median / 1e3) / 1e9,
    })
    return True


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--kernel", choices=tuple(KERNELS), default="sfa")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = arguments()
    report = {
        "status": "error", "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "hostname": platform.node(),
        "kernel": args.kernel, "heads": args.heads,
        "top_k": args.top_k, "width": args.width,
        "device": args.device, "calls": args.calls, "warmup": args.warmup,
        "batches": args.batches, "seed": args.seed,
        "sequence_length": SEQUENCE_LENGTH,
        "scope": "one die, one rank, shared with another tenant",
        "kv_note": "MLA keeps one KV head, so heads do not change kv_bytes",
        "index_note": "sorted distinct random positions; the model shares "
                      "indices across MTP rows",
    }
    try:
        passed = measure(args, report)
    except Exception as error:
        report["error"] = repr(error)[:2000]
        passed = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"kernel": args.kernel, "heads": args.heads,
                      "top_k": args.top_k, "passed": passed,
                      "median_us": report.get("median_us"),
                      "gb_per_s": report.get("gb_per_s")}))
    return int(not passed)


if __name__ == "__main__":
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    raise SystemExit(main())
