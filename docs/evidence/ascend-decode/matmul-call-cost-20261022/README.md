# A matmul call costs 18.5 microseconds before it reads a byte

This is the mechanism the megakernel is for, and it is not the one the earlier
reasoning rejected. Merging kernel launches was priced at 1.064 microseconds
each and dismissed. Merging matmul *invocations* is worth 18.5 microseconds
each, it survives graph capture, and a layer makes about a dozen of them.

## The measurement

One idle die of a3-194, five repeats of 30 iterations, NPU events. Every arm
computes the same products from the same `[12, 6144]` activation against the
same `[6144, 2816]` weight, 34.6 MB; only the number of matmul calls differs,
because the weight is split along its output axis.

| calls | eager us | graph us | us an extra call, eager | graph |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.57 | 17.21 | — | — |
| 2 | 31.40 | 32.18 | 14.83 | 14.97 |
| 4 | 71.97 | 74.09 | 18.47 | 18.96 |
| 8 | 147.58 | 152.76 | 18.72 | 19.36 |

## It is device time, not host dispatch

The graph column is the control that matters. A captured graph holding the same
repetitions, replayed once, charges the same 18.96 to 19.36 microseconds an
extra call -- marginally more than eager, inside the noise. Host dispatch is
gone under replay, so the cost is on the device and a kernel can address it.

That control was necessary. An earlier reading in this project attributed
12.62 ms a round to eager dispatch from a profiler trace that cannot tell
dispatch from replay, and had to be withdrawn.

## It is exposed only when the weight is small

| weight | 1 call | 8 calls | us an extra call |
| --- | ---: | ---: | ---: |
| 34.6 MB, L2-resident | 16.20 | 147.36 | 18.74 |
| 302 MB, exceeds L2 | 231.94 | 305.25 | 10.47 |

At 302 MB the matmul streams from HBM at 1302 GB/s and the fixed cost hides
under the traffic. At 34.6 MB there is no traffic to hide behind. A layer's
per-rank weights are about 50 MB -- 11.9 of attention and 37.8 of the five
touched experts -- so every projection in the decode path sits in the exposed
regime.

The projections confirm it at their own shapes. Per-rank `q_b` at 4.19 MB takes
13.61 microseconds at one row and 12.92 at thirty-two; `o` at 12.58 MB takes
32.78 and 34.64; `kv_b` at 1.84 MB takes 9.44 and 9.44. All are flat in rows and
all are an order above their bytes, which is what a fixed cost looks like.

## Why this closes the gap

The round is 650.4 microseconds a layer. Attention's five profiled windows sum
to 301.11 at twelve rows, and its two projection phases alone are 99 -- Prolog
42.38 and the projection tail 56.55 -- against 11.9 MB of weights that need 8.3
microseconds at bandwidth. Counting the layer's matmul invocations across
Prolog, the indexer front, SFA, the projection tail, the router, the shared
expert and the two grouped calls gives roughly a dozen, and a dozen times 18.5
is about 220 microseconds. The unidentified gap was 289.

So the megakernel's object is not the launch count and not the memory round
trip. It is the matmul prologue, and the way to remove it is one cube pass whose
output axis spans every projection that shares an activation, rather than one
call per projection inside one launch -- which is what the native stages do
today.

[`call_cost_graph.py`](call_cost_graph.py) is the controlled measurement,
[`call_cost.py`](call_cost.py) the two-size sweep, and the JSON files record
both along with the projection and weight-size sweeps.

## Where the lever is still untaken

The 18.5 microseconds is the cost of a separate vendor operator invocation. It
is not the cost of an `IterateAll` inside an AscendC kernel, which is a loop
rather than a launch, and the native sources already hoist `Matmul::Init` out of
their tile loops. Partitioning the indexer front's two projections across core
groups so each core pays one `Init` instead of two saved 7.16 microseconds at
M=12 with C8 caches, so the in-kernel figure is about 7, not 18.5.

That resolves an apparent contradiction. The native FFN beats the stock CANN
chain 161 against 310 microseconds, which is roughly eight operator prologues,
and yet the native whole layer at 574.15 matches the vendor whole layer at
575.22. The reference the native FFN beats is not what the engine runs: the
engine's own SuperKernel already fuses GatherV3, QuantBatchMatmulV3,
GroupedMatmul, ForeachAddList and DynamicQuant. A kernel census of a round found
about eight `sk_*` a layer and **zero** whose name mentions attention.

So the FFN's prologue cost is taken twice over -- by the engine's SuperKernel and
by the native body -- while attention's is taken by neither. Attention's two
projection phases measure 99 microseconds against 8.3 of weight bytes, and the
composed native attention entry beats separate stages by only 11.6. The
remaining 91 microseconds a layer, 7.1 milliseconds a round, is the one place
where a megakernel has something that neither path has already claimed.
