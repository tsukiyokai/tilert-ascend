"""Time sparse attention at the capture's own shape, on one die.

`SparseFlashAttention` is 87.01 us a call and 73.43 of that is above the floor
any MIX kernel pays -- the largest unexplained term in the verify. Rows,
bytes, structure and arithmetic have each been excluded by measurement, so
what remains is to vary the one quantity nothing has varied: the number of
selected positions.

If the call is flat in picks, the cost is a fixed setup and the 2048 are
free; if it is linear, the slope says what a pick costs and the intercept
what the call costs before any. Either answer names the next change.

The shape comes from `kernel_details.csv`: query 8 rows x 4 heads x 512
latent plus 64 rope, a paged cache of 73 blocks x 128 slots, 2048 picks a
row, a block table of 73 entries, all bfloat16.

One device, one rank -- attention is per-rank over that rank's heads, so a
single die reproduces the production call exactly.
"""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter_ns

HEADS = 4
LATENT = 512
ROPE = 64
BLOCKS = 73
BLOCK_SIZE = 128


def attention_call(rows, picks, device):
    """Return a thunk that runs one sparse attention call.

    The operator and its layouts come from the production call site,
    `vllm_ascend/attention/sfa_v1.py:224`: `npu_sparse_flash_attention` with
    a TND query against a PA_BSND paged cache, `sparse_block_size=1`,
    `sparse_mode=3`, `attention_mode=2`. Feeding the captured shapes to
    `npu_fused_infer_attention_score_v2` instead fails with 561103 -- page
    attention there wants the key's N axis to be 1 -- which is how it became
    clear that production uses a different operator.
    """
    import torch

    # The tiling accepts a qk head dim of 512 only, so the 64 rope values go
    # through the separate rope inputs rather than concatenated -- which is
    # also what the capture shows: query 8,4,512 beside 8,4,64.
    options = {"dtype": torch.bfloat16, "device": device}
    query = torch.randn(rows, HEADS, LATENT, **options)
    query_rope = torch.randn(rows, HEADS, ROPE, **options)
    cache = torch.randn(BLOCKS, BLOCK_SIZE, 1, LATENT, **options)
    cache_rope = torch.randn(BLOCKS, BLOCK_SIZE, 1, ROPE, **options)
    slots = BLOCKS * BLOCK_SIZE
    indices = torch.randint(0, min(picks, slots), (rows, 1, picks),
                            dtype=torch.int32, device=device)
    # One request, `rows` tokens: under TND the rows are a sequence's tokens
    # and the batch is 1, so the block table carries one row and the query
    # lengths are a single cumulative count. Giving one length per row made
    # the tiling read a batch of eight and reject a table of one.
    table = torch.arange(BLOCKS, dtype=torch.int32,
                         device=device).unsqueeze(0)
    lengths_q = torch.tensor([rows], dtype=torch.int32, device=device)
    lengths_kv = torch.tensor([picks], dtype=torch.int32, device=device)

    def run():
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            query=query.contiguous(), key=cache, value=cache,
            sparse_indices=indices, scale_value=LATENT ** -0.5,
            sparse_block_size=1, block_table=table,
            actual_seq_lengths_query=lengths_q,
            actual_seq_lengths_kv=lengths_kv,
            query_rope=query_rope, key_rope=cache_rope,
            layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=3, attention_mode=2, return_softmax_lse=False,
        )

    return run, indices, table


def time_call(run, calls, warmup):
    import torch

    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    samples = []
    for _ in range(calls):
        begin = perf_counter_ns()
        run()
        torch.npu.synchronize()
        samples.append((perf_counter_ns() - begin) / 1e3)
    samples.sort()
    return {"p10": samples[len(samples) // 10],
            "median": statistics.median(samples),
            "p90": samples[9 * len(samples) // 10]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 8))
    parser.add_argument("--picks", type=int, nargs="+",
                        default=(128, 512, 1024, 2048))
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401
    # Importing vllm_ascend leaves one op in the `_C_ascend` namespace and
    # loading its kernel library by hand adds none. The vendor's own
    # single-card test for this operator says what registers them.
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    results = []
    for rows in args.rows:
        for picks in args.picks:
            run, _, _ = attention_call(rows, picks, device)
            timing = time_call(run, args.calls, args.warmup)
            results.append({"rows": rows, "picks": picks, **timing})
            print(f"rows {rows:2} picks {picks:5}  p10 {timing['p10']:8.2f}  "
                  f"median {timing['median']:8.2f}  p90 {timing['p90']:8.2f} us",
                  flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "sparse attention at the capture's shape, one die",
        "heads": HEADS, "latent": LATENT, "rope": ROPE,
        "blocks": BLOCKS, "block_size": BLOCK_SIZE,
        "calls": args.calls, "warmup": args.warmup,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
