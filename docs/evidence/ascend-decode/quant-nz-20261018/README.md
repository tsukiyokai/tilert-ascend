# The quantised matmul is fixed-cost bound, and NZ does not rescue it

`a3-84`, one idle die, 24 rotating weight banks,
`torch_npu.npu.config.allow_internal_format = True` so
`npu_format_cast(w, 29)` really produces FRACTAL_NZ rather than silently
falling back. Die ceiling 1190.77 GB/s.
[rates](rates.json), [benchmark](../../../../tests/glm52/bench_quant_matmul_nz.py).

| shape | rows | layout | eager µs | GB/s | share of ceiling |
| --- | --- | --- | --- | --- | --- |
| o_proj `[2048, 6144]` | 4 | ND | 23.94 | 525.5 | 44.1% |
| o_proj | 4 | **NZ** | **22.67** | **555.0** | 46.6% |
| q_b_proj `[1536, 6144]` | 4 | NZ | 22.74 | 415.0 | 34.9% |
| q_a_proj `[6144, 1536]` | 4 | ND | 33.90 | 278.3 | 23.4% |
| q_a_proj | 4 | **NZ** | **27.18** | 347.2 | 29.2% |

## This retracts my retraction

I had recorded that the quantised matmul's deficit was an artefact of
benchmarking an ND weight, since production casts to FRACTAL_NZ. Measured,
NZ buys 5% on o_proj (23.94 → 22.67 µs) and 20% on q_a_proj (33.90 → 27.18).
The deficit survives: the earlier claim is withdrawn and the original
observation reinstated with its cause corrected.

## The cause is per-call fixed cost, not bytes

Two comparisons settle it. At the same shape bfloat16 moves 25.17 MiB in
22.52 µs (1117.3 GB/s, 94% of ceiling) while int8 moves 12.58 MiB in
22.67 µs — **half the bytes in the same time**. And across shapes the time
moves against the bytes: `q_a_proj` carries 9.44 MiB and costs 27.18 µs while
`o_proj` carries 12.58 MiB and costs 22.67. Fewer bytes, more time. Nothing
bandwidth-bound behaves like that.

What does vary with the cost is N, the output width: 6144 columns give 22.7 µs
and 1536 give 27.2, consistent with N supplying the parallelism across cube
cores. So a batch-one projection costs roughly 22–33 µs almost regardless of
its weight size, and that is prologue.

The arithmetic then closes on the profile. Dense projections at about five
calls per layer and ~25 µs each give ~9.4 ms per round; the expert GMMs at
160–172 µs per layer over 75 MoE layers give 12–13 ms. The sum, ~22 ms,
matches the measured `SuperKernel` MIX_AIC block of **22.80 ms per round**
([round composition](../profile-rope-20261018/README.md)). The cube block is
almost entirely per-call fixed cost, with the ~6 ms streaming floor buried
inside it.

## Which measurement is the honest one

In-graph numbers are mostly *worse* than eager here — o_proj NZ 29.10 against
22.67 — because a single-operator graph pays its own replay floor of about
6 µs, while eager back-to-back calls already hide dispatch behind the previous
kernel. [The chain sweep](../matmul-streaming-20261018/README.md) established
that directly: one call and thirty-two chained calls cost the same per call.
So eager back-to-back is the right instrument for a single operator, and the
graph instrument belongs to whole-round measurements.

## What it means for the fused kernel

This is the sharpest argument yet, and it is not the one I started with.
Fusing does not buy bandwidth — the vendor already streams bfloat16 at 94% of
peak. It buys the elimination of per-call prologue, and at batch one that
prologue is most of the cube block. A hand-written kernel that does a norm, a
projection, a quantisation and a reduction in one launch replaces four
prologues with one, and the prologue is worth 22–33 µs each.
