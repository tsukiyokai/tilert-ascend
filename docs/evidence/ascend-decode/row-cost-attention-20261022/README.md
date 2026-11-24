# Attention carries 0.356 ms a row, and one shared read removes all of it

2026-10-22, a3-180, one die, TP16 per-rank shapes: 4 query heads, top-k 2048,
8192 context, bf16, MLA's single KV head. 20 batches of 20 calls a case, one
shape a process. The fused sparse kernel needs
`vllm_ascend.vllm_ascend_C`, which no tree on this host carries, so the three
native kernels stand in -- `gather` is the scattered read alone, `gather_fia`
is that read feeding the dense fused attention, and `shared_fia` reads the
whole cache once under a per-row mask, which is what the engine's
`cann_attention.py:shared_kv_fia` does.

Median microseconds a layer:

| width | gather | gather_fia | shared_fia |
| --- | --- | --- | --- |
| 1 | 30.811 | 131.583 | 107.084 |
| 4 | 29.491 | 147.598 | 102.490 |
| 8 | 43.604 | 158.767 | 93.970 |
| 12 | 47.095 | 139.360 | 106.495 |
| 13 | 60.798 | 145.851 | 105.060 |
| 16 | 85.351 | 157.045 | 85.456 |
| 32 | 125.853 | 230.763 | 87.628 |
| 64 | 169.549 | 330.601 | 112.706 |

Slope over 12 to 32, the span the round model was fitted on, in ms a row of a
78-layer round: **gather 0.307, gather_fia 0.356, shared_fia -0.074**.

So attention supplies 38 to 44 percent of the round model's 0.814 ms a row,
and the shared read supplies none of it. Its 85 to 113 microseconds over
widths 1 to 64 have no trend; the variation is the kernel's tiling regimes.

## The row cost is the read, not the arithmetic

`gather` alone is 3.938 of `gather_fia`'s 4.570 microseconds a row a layer --
86 percent. Every extra verify row reads its own 2048 selected positions, and
2048 x 576 x 2 bytes is 2.36 MB a row a layer. The attention arithmetic over
those rows adds 0.63.

That is why the shared read wins by more as the width grows: its bytes are the
cache, 9.44 MB, whatever the width.

| width | per-row read | shared read | gather_fia / shared_fia |
| --- | --- | --- | --- |
| 12 | 28.3 MB | 9.44 MB | 1.31x |
| 32 | 75.5 MB | 9.44 MB | 2.63x |
| 64 | 151.0 MB | 9.44 MB | 2.93x |

## The step is in the gather's tiling, not in any graph

12 to 13 costs 13.703 microseconds for one row, 13 to 16 costs 8.18 a row, and
16 to 32 costs 2.53. The marginal row is five times dearer at 12 than at 32,
so the round model's single linear term is a smoothing of a curve with
regimes. A per-row figure quoted without its width is not comparable.

## Scope

Synthetic queries and sorted distinct random positions per row, which is the
shape of the engine's `topk_indices`; the engine's real selections are more
local, so the per-row gather here is an upper bound on its read. These are
operator medians on one die, not a round, and `gather_fia` is already known to
be about 19 percent dearer than the fused sparse kernel at width 12, so the
level is not the engine's. The slopes and the ratio are the claim.

The mask build is outside the timed call, as it is in the engine, where an
index producer refreshes it and the consumer layers reuse it. It is
`width x 8192` bytes -- 96 KB at width 12, 512 KB at 64 -- and one scatter.

`sfasweep` is the same sweep run sixteen ways in parallel, one case a die. Its
12-to-32 slopes are 3.95 and 4.03 against the serial arm's 3.938 and 4.570, so
A3's two-dies-a-module pairing did not distort it.
