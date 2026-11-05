# The all-gather tail is arrival spread, and `--draft-projection-tp` buys it

## Scope limit, added the same afternoon

This capture was taken on a shared machine while another tenant's residency
rose from 12.7 to 40.4 GiB a die and began computing. HBM contention produces
exactly the shape measured here — balanced compute, 8% spread, against waiting
that differs by 107% — so the attribution to our own structure is not settled
until the same capture is taken on an idle host.

Two hosts were tried that afternoon and both filled: a3-84 left 21.8 GiB free
against the 24.20 a rank needs for weights plus KV, and a3-187 went from 3.0
to 60.5 GiB a die within twenty minutes of a container being built on it. The
move itself turned out to be nearly free — the 378 GiB of weights already sit
on `8.5.255.11:/mnt/sfs_turbo`, which both machines mount, and 187 still held
the CANN 9.2.0 image from 2026-10-16, so only 13 GiB had to travel.


From the width-8 capture, seven ranks analysed.

## Every collective is in the draft graph, none in the verify

| `Model ID` | collectives |
| --- | --- |
| 49 (draft graph) | 216 |
| 47 | 36 |
| 4294967295 (eager) | 20 |
| **48 (target graph)** | **0** |

So the verify's TP reductions are fused into the MC2 matmuls and never appear
as collectives. Which means the "157 reductions a round" the one-sided
reduction was priced against is not what the verify does, and the explicit
collectives all belong to the draft side.

## The tail is 96% of the all-gather time

| | `hcom_allGather_` | `hcom_allReduce_` |
| --- | --- | --- |
| calls | 140 | 132 |
| p10 | 22.44 µs | 11.90 |
| median | 25.40 | 22.92 |
| p90 | **788.48** | 32.04 |
| max | **13240.94** | 2349.82 |
| over 10× the median | 50 calls, 61.096 ms, **96%** | 6 calls, 9.936 ms, 78% |

The `Wait Time` column has a median of 0.36 µs and totals 0.349 ms, so the
61 ms is the kernel's own duration, not a profiler-reported wait.

## Not a slow rank: arrival spread

| rank | gather total | gmm1 total | gmm2 total |
| --- | --- | --- | --- |
| 7 | 22.321 ms | 22.953 | 19.072 |
| 8 | 34.686 | 22.740 | 20.142 |
| 15 | 42.117 | 24.190 | 20.710 |
| 1 | 42.268 | 22.507 | 19.359 |
| 2 | 56.636 | 23.465 | 20.442 |
| 0 | 63.554 | 22.419 | 18.515 |
| 3 | 73.676 | 23.910 | 20.406 |

Gather totals span 22.3 to 73.7 ms — a spread of 107% of the mean — while
gmm1 totals span 8% and every rank's **median** gather is between 23.6 and
26.0 µs. So the compute is balanced, no rank is systematically slow, and the
waiting is randomly distributed: at each of 16 ranks' collectives everyone
waits for whoever happens to arrive last.

That rules out both explanations this page was written to test — a straggler
card, and uneven expert routing making one rank's MoE heavier.

## Where they come from

`cann_mtp_projection.py` rewrites the draft's `rot` and `eh_proj` as
`AscendColumnParallelLinear(..., gather_output=True)`. Column-parallel with
gathered output is an all-gather per forward, and a round runs seven draft
forwards, so `--draft-projection-tp` costs about fourteen 16-rank collectives
a round. Every measurement on this board has had it on.

The option's benefit is spreading two projections across sixteen ranks. At
batch one and eight rows those projections are one small matmul each, so the
arithmetic saved is small beside a full barrier's exposure to arrival jitter.
The A/B is running; the prediction is that turning it off wins, and three of
my predictions today have been refuted by the measurement that followed.

## What it does to the one-sided reduction's value

That work replaces `_all_reduce_out_place`, so it reaches all-reduces and not
all-gathers — and the all-gathers carry the tail. Its value was priced per
call at 16.571 µs against 32.340, 1.95×, over a reduction count that this page
shows was counted on the wrong side. It remains the only mechanism on the
board that removes a full barrier rather than making one faster, which is what
arrival spread needs, but it would have to be applied to the gather.
