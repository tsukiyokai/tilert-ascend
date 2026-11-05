# The KV gather is bandwidth-bound, so run structure buys nothing

Sparse attention selects 2048 of 8192 positions a row. The hypothesis was that
the gather's cost follows the number of descriptors rather than the bytes, and
that a copying workload -- whose attended positions should fall in contiguous
spans -- could coalesce them. It does not. The cost is flat in run length.

## The measurement

One idle die of a3-194, seven repeats of 30 iterations, NPU events. Every arm
gathers the same 2048 positions a row over twelve rows from an `[8192, 576]`
bf16 cache, which is 28.31 MB; only the run length differs.

| run length | median us | GB/s |
| ---: | ---: | ---: |
| 1 | 25.80 | 1097 |
| 2 | 25.88 | 1094 |
| 8 | 25.91 | 1093 |
| 64 | 26.01 | 1088 |
| 256 | 25.84 | 1096 |
| 2048 | 25.89 | 1094 |

The spread across a 2048-fold change in run length is 0.21 microseconds, under
one percent. Fully scattered costs what one contiguous block costs.

## Two things this retires

The road's fourth step was to make the latent's 512 columns and the rope's 64
contiguous, estimated at 23 microseconds a layer. The figure it rested on was
294 GB/s, which was measured at **one** row where the read is latency-bound. At
the engine's twelve rows the gather already runs at 1094 GB/s, and the layout
cannot be the constraint.

It also explains a result from the other direction. The packed-KV candidate
combined each position's latent and rope in one cache and gathered each pair
with a single command; it regressed by 14.87 to 21.65 microseconds at M=12 with
zero wins in 240 paired comparisons, and the packed-workspace variant regressed
further. Both were paying for a reorganization that had nothing to win.

## What it leaves inside SFA

SFA's measured window is 76.97 microseconds a layer. The gather of its selected
positions is 26 of that, and the attention arithmetic -- 12 rows, four local
heads, 2048 positions, roughly 107 million multiply-accumulates -- is about 2
microseconds at the chip's cube rate. So some 49 microseconds is neither the
bytes nor the arithmetic, and itemizing it needs a profile of the kernel rather
than another shape probe.

[`probe.py`](probe.py) is the measurement and [`result.json`](result.json) the
seven repeats.
