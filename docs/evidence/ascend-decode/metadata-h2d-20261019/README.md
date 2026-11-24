# The chain does not pipeline, because one copy a step drains the queue

`phase-attribution-20261019` found 5.53 ms a round of wall the device is not
computing in, and ruled out launch count. This is the mechanism.

## One pageable copy costs what is queued behind it

`tests/glm52/bench_metadata_h2d.py`, one die of a3-187, 300 repeats, the exact
copies `CannModel._metadata` makes. The queue depths are the real ones: 73
kernels is one draft graph, 365 is five of them, 2489 is the target graph.

| queue depth | width | pageable | pinned | removed |
| --- | --- | --- | --- | --- |
| 0 | 1 | 108.17 | 42.39 | 65.8 |
| 73 | 1 | 139.36 | 47.79 | 91.6 |
| 365 | 1 | 345.88 | 50.66 | 295.2 |
| 2489 | 1 | 1141.18 | 71.18 | 1070.0 |
| 0 | 8 | 87.85 | 34.40 | 53.5 |
| 73 | 8 | 158.77 | 40.86 | 117.9 |
| 365 | 8 | 269.35 | 43.53 | 225.8 |
| 2489 | 8 | 2936.31 | 64.29 | 2872.0 |

Median host microseconds, [bench.json](bench.json). **Pageable scales with
the queue; pinned does not.** A copy from pageable host memory has to stage,
and the runtime drains the stream to do it, so the caller pays for every
kernel already enqueued. Pinned memory is addressable by the engine, so
`non_blocking=True` is actually non-blocking: 34 to 71 µs across a 2489-fold
range of queue depth.

## Why it lands on the chain and not elsewhere

A round deliberately issues every device call before reading anything — that
is `MtpRoundExecutor.advance`'s whole design, and `host_tokens` is its only
synchronisation. So the queue is deep by construction, and each forward's
copy pays for the forwards before it. In the chain that compounds: step *k*'s
copy drains steps 1 through *k-1*, so no draft step is ever enqueued while
another runs. The chain's wall is 10.27 ms against 6.72 of device compute,
and the 3.55 ms difference is this.

Summing the measured curve over the chain's own depths — step *k* sees about
73*k* queued — gives 1.6 ms of drain against 0.3 ms pinned. The rest of the
3.55 is the pipelining those drains prevent, which the bench cannot show and
the round will.

The target's own copy, at width 8 behind the chain's 438 kernels, is about
270 µs against 44. The conditioned draft's copy sits behind the target's
whole forward and blocks for it — but the host was going to wait at
`host_tokens` anyway, and the conditioned pass depends on the target's
output, so nothing is lost there.

## The change, narrowed to the chain's one copy

A width-1 forward makes exactly one host-to-device copy — `seq_lens` — and
fills `slots` and `positions` with a scalar. Six of a round's nine forwards
are width 1, and they are the chain. So `seq_lens` is the whole target, and
the width > 1 paths are left alone: no new device operation anywhere.

`_metadata` takes its length from a pinned ring and copies `non_blocking`.
The ring is not decoration. The host enqueues every forward of a round before
it reads anything, so with one shared buffer the host would overwrite the
next length before the engine had read this one — a silent wrong `seq_lens`,
not a slowdown. One slot per forward a round can have in flight.

This is also why `hoist-20261019` read null at 313 → 91 launches and −0.14%
TPOT: the three allocations and copies sit **above** the `hoisted is not None`
early return, so the hoist removed the attention-metadata builder and kept
the entire cost. Launch count was never the variable.

## This is not the change `29ada2c` reverted

`cc794c2` touched the same lines and was reverted the same morning on a paired
width-8 chain with complete three-against-three separation: the round floor
rose 0.278 ms. Its revert message concludes that "a copy_ from a CPU tensor to
a device tensor does not synchronise the device; it was never a host round
trip, which is what the premise assumed."

That conclusion is wider than its experiment. Both of its arms kept
`self.seq_lens.copy_(length_cpu)` with a pageable source, so neither arm ever
varied the thing in question; what varied was two copies against three device
kernels — a subtraction, an add and a widening copy — nine forwards a round,
twenty-seven more launches. The patch failed on launch count, which is what
the revert measured and what the bench above does not dispute.

The variable this change moves is the source buffer's residency, and it adds
no device operation: the same copies, the same fills, the same counts, from
pinned memory instead of a fresh allocation. So the downside is bounded at
nothing, and the device-side `_offsets` arithmetic `cc794c2` introduced stays
reverted.

`positions-20261019/tree` still carries that patch, and `topk-run3` is running
from it. Its arms both carry the 0.278 ms, so their comparison holds, but its
absolute TPOT of 12.19 is not the project's floor.

## Prediction, to be judged on the round

3.5 to 3.9 ms a round, 0.6 to 0.9 ms of TPOT at acceptance 4.507. Per-call
figures have failed to predict rounds in both directions on this project, so
this is a prediction and not a result until a paired alternating chain says
otherwise.
