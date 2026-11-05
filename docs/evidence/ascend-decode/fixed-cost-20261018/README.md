# What a batch-one projection actually costs, eager and in a graph

`a3-84`, one idle die, FRACTAL_NZ weights with
`allow_internal_format = True`, rotating banks, `npu_quant_matmul` at four
rows. Five fast experiments, each a few minutes.
[benchmark](../../../../tests/glm52/bench_fixed_cost.py).

## Eager, the cost is a constant

| K | weight | µs | GB/s |
| --- | --- | --- | --- |
| 256 | 1.5 MiB | 29.97 | 52.5 |
| 512 | 3.0 | 26.61 | 118.2 |
| 1024 | 6.0 | 26.00 | 241.9 |
| 2048 | 12.0 | 27.06 | 465.0 |
| 4096 | 24.0 | 28.91 | 870.6 |

Sixteen-fold more bytes, the same time. Shrinking both axes confirms it:
`K=32, N=32` — one kibibyte of weight — still costs 21.68 µs, and
`K=2048, N=6144` costs 22.00. Rows do not matter either: 1, 4, 16 and 64 rows
cost 26.90, 26.07, 25.81 and 26.47 µs. At decode widths the cube is idle.

`torch.matmul` in bfloat16 separates the floor from the operator: `[4,64] ×
[64,64]` costs 9.64 µs, which is the eager dispatch floor this project
measured earlier at ~10 µs, while `[4,2048] × [2048,6144]` costs 24.51. So
`npu_quant_matmul`'s 21.5 µs is roughly the 10 µs dispatch floor plus 11 µs
of its own per-call work.

## In a graph the two separate cleanly

Capturing a chain of N identical calls and reading the marginal cost removes
dispatch, which is what production pays
([the round runs inside an ACL graph](../profile-rope-20261018/README.md)).

| weight | eager marginal | **graph marginal** |
| --- | --- | --- |
| 1 KiB | 21.54 µs | **7.92 µs** |
| 12 MiB | 21.63 µs | **19.22 µs** |

So in a graph the operator keeps a floor of about **8 µs per call**, and the
remaining 11 µs carries 12.58 MB — **1144 GB/s, 96% of the die's measured
ceiling**. Two conclusions at once: the streaming is already at peak so there
is no bandwidth to win, and the 8 µs floor is real rather than a dispatch
artefact.

This corrects the claim that a projection is 95% prologue. Eagerly it is;
in a graph the prologue is 8 of 19 µs, about 42%. The earlier statement was
measured with the wrong instrument — the same mistake this project made with
TileLang earlier in the day.

## Fusing calls is worth 1.6 to 2.2×, up to a knee

Same 48 MiB of weight either way, all captured in one graph:

| calls | each N | µs | GB/s | vs one call |
| --- | --- | --- | --- | --- |
| 1 | 24576 | 54.78 | 918.7 | 1.00× |
| 2 | 12288 | 56.72 | 887.4 | 1.04× |
| 4 | 6144 | 87.43 | 575.7 | **1.60×** |
| 8 | 3072 | 121.82 | 413.2 | **2.22×** |

Merging four calls into one saves 37%, eight into one saves 55%, and merging
two into one saves nothing. The knee is a single call of roughly 24 MiB: past
it the call is bandwidth-bound at 890–920 GB/s and there is nothing left to
amortise. Every per-layer dense projection is far below that, so they sit
squarely in the profitable region.

The eager version of this comparison read 46.9% for two-into-one, which was
half dispatch. The in-graph numbers are the ones to plan with.

## What it means for 2 ms

The fused kernel's case survives, resized. Per round the recoverable
per-call floor is roughly 8 µs times the number of cube calls — a few
milliseconds for the dense projections, plus the expert GMM's own 62–63 µs
per call over ~150 calls ([clean GMM](../gmm-clean-20261018/README.md)). That
is the largest single item in the 22.80 ms cube block and fusion is how to
take it.

But it is not the path to 2 ms by itself, and the arithmetic says why. At
acceptance 3.433 a round delivers 3.433 tokens, so 2 ms per token allows
6.87 ms per round, while reading each rank's weights once costs 5.7–6.0 ms at
the measured ceiling — **84% of the budget is unavoidable**. Clearing every
structural tax leaves about 1.1 ms of room.

So the sufficient condition is more tokens per round or fewer bytes per
round, and today's row sweep points at the first: a projection costs the same
at 1, 4, 16 and 64 rows. Widening MTP is nearly free on the projection side,
and a round that delivers 6 tokens may cost 12 ms. That is the next
experiment.
