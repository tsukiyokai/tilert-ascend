# Sparse attention is 59% fixed and 41% picks

`tests/glm52/bench_sfa_shape.py`, one die, the capture's own shape and the
production operator — `npu_sparse_flash_attention` with a TND query against a
PA_BSND paged cache, `sparse_block_size=1`, `sparse_mode=3`,
`attention_mode=2`, taken from `sfa_v1.py:224`.

| picks | 1 row | 8 rows |
| --- | --- | --- |
| 128 | 92.69 µs | 96.34 |
| 512 | 112.68 | 113.44 |
| 1024 | 125.44 | 129.47 |
| 2048 | **153.44** | **157.55** |

A line fits both: `88.64 + 31.64 µs a thousand picks` at one row,
`92.26 + 31.88` at eight. The slopes agree to 0.8%, and eight rows cost 1.027×
one row — the same row-independence the capture shows.

At 2048 picks and eight rows that splits as **92.3 µs fixed and 65.3 µs for
the picks**, so 59% of the call does not depend on how many positions it
attends to.

## The bench reads 1.81× the capture

157.55 against 87.01. The bench calls the operator eagerly through aclnn while
the capture replays it inside a graph, so the bench carries a dispatch the
capture does not. What transfers is the curve's shape; scaling it onto 87.01
gives roughly 51 µs fixed and 36 µs for the picks.

## Two routes, with prices

**Fewer picks.** The slope is 31.88 µs a thousand, so halving 2048 to 1024
saves 32.6 µs eagerly, about 18 µs in the graph, and over 64 calls **1.15 ms a
verify**. The pick count is the indexer's `topk_tokens`, a hyperparameter; what
it costs in acceptance is directly measurable, since every width tested so far
emits the digest `68f2bc184553` and a change in acceptance would show as a
change in round count.

**The 92 µs that is neither picks nor rows.** It is the per-call setup, and it
is what the kernel's own source has to explain. Four explanations have already
been excluded by measurement — rows, bytes, the MIX structure, arithmetic —
and this isolates a fifth quantity rather than guessing one.

## Five wrong calls before the operator ran

Worth recording because each was a different kind of wrong, and the vendor's
own test held the answer to the fourth:

1. `npu_fused_infer_attention_score_v2` with the captured shapes: 561103, page
   attention wants the key's N axis to be 1. Production uses a different
   operator entirely.
2. `torch.ops._C_ascend.npu_sparse_flash_attention`: the namespace is absent
   until `vllm_ascend` is imported.
3. After importing it, the namespace holds one op and not this one.
4. Loading `libvllm_ascend_kernels.so` by hand adds none either. The vendor's
   single-card test for this operator says what does:
   `from vllm_ascend.utils import enable_custom_op`.
5. Then two tiling rejections, each naming its own fix: `qk_head_dim only
   support 512, but got 576` — the rope goes through its own input, as the
   capture's `8,4,512` beside `8,4,64` already showed — and `block_table's
   first dimension(1) should be equal to batch size(8)` — under TND the rows
   are one request's tokens, so the query lengths are one cumulative count and
   not one per row.
