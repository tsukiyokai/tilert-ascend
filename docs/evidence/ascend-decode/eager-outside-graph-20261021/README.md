# A round makes 1108 eager calls outside its graph, worth 11.4 ms of dispatch

The census splits the capture's kernels by name family. `sk_*` is the captured
super kernel; anything beginning `aclnn` is an eager aclnn call, which pays host
dispatch -- 11.386 us measured in this project against 1.064 us under graph
replay.

Per round, from `profile-rope-20261018` rank 1 over its three rounds:

| family | calls a round | device ms a round |
| --- | --- | --- |
| collective | 43 | 36.62 |
| super kernel (captured) | 637 | 33.58 |
| aclnn (eager) | **1108** | 7.40 |
| other | 81 | 0.42 |

    1108 eager calls at 11.386 us dispatch = 12.62 ms
    the same under replay at 1.064 us      =  1.18 ms
    difference                               11.44 ms a round

Against the round's unaccounted 12 ms. The match is close enough to name the
residual, and the remedy is the one the engine already applies to the rest of
the model: capture.

## It also corrects the layer figure

The super kernel is not 78 calls of 210 us. It is **637 calls a round totalling
33.58 ms** -- so the layer's own device time is twice what the 210 us reading
suggested, and the earlier 16.4 ms was the cost of 78 of them rather than all
637.

Device time now sums above the measured round (36.62 + 33.58 + 7.40 = 77.6 ms
against 50.710), which means the streams overlap. That is consistent with the
earlier finding that a single stream already recovers part of a serial sum, and
it means device totals cannot be added into a round budget without knowing the
overlap.

## What is actionable

The eager calls are the residual and they are host dispatch, not device work --
7.40 ms of device time against 12.62 of dispatch. Which of the 1108 are
capturable is the next question, and the names say where to look: the census's
largest eager families were `aclnnInplaceCopy_CastAiCore_Cast` at 530 a round
and `aclnnMuls` at 75.

## Where the 530 casts a round come from

`refresh_hoisted_rope` -> `refresh_rope`, in `cann_model.py`:

    torch.index_select(rope._cos_cache, 0, positions, out=rope._cos_mla[:width])
    torch.index_select(rope._sin_cache, 0, positions, out=rope._sin_mla[:width])
    for cos, sin in destinations:
        cos.copy_(rope._cos_mla[:width])
        sin.copy_(rope._sin_mla[:width])

`aclnnInplaceCopy_CastAiCore_Cast` is `copy_`'s kernel, and 530 a round over
two copies an entry puts about 265 entries in `destinations`.

They are not redundant. `rope_destinations` already deduplicates by
`data_ptr()`, and its comment names the reason: a builder configured for
speculative decoding keeps independent RoPE storage and copies the shared
gather into it, so every distinct buffer has to be written every round.

## But the addresses are fixed, so the copies are capturable

The destinations come from hoisted metadata -- the addresses a replay captured
-- so they do not change between rounds. `index_select`'s output and every
`copy_`'s target are the same pointers every time, which is exactly the
condition a graph needs.

So the 11.44 ms is not work to remove but dispatch to capture: 1108 eager calls
at 11.386 us against 1.064 under replay. The engine already captures the model
forward; this is the per-round metadata refresh that sits outside it.

Two ways in, both untried: capture the refresh in its own small graph replayed
before the model's, or make the speculative builders share the MLA buffers so
`destinations` is empty and the loop disappears. The first is ours to do; the
second is a vLLM-Ascend builder configuration.

## The pattern is capturable, 22.5x, and bit-exact

One die, the refresh's shape replicated with 265 destinations -- two
`index_select` calls writing into fixed outputs, then 530 `copy_` calls:

    eager          7588.89 us
    graph replay     336.63 us   22.54x
    saved            7252.26 us a round
    bit-exact after a replay at new positions: True

The correctness check is the one that matters: `positions` is moved to a
different range, the graph is replayed, the result is kept, and then the eager
path is run at those same positions. They agree exactly -- so the capture holds
the addresses and reads whatever the buffer contains at replay, which is the
premise `RopeGraph` rests on.

And the eager figure is higher than the estimate: 7.59 ms measured against 6.06
ms from 532 calls at 11.386 us. So the dispatch estimate was conservative.

## Which makes this the session's largest validated lever

7.25 ms a round on a 50.710 ms round, mechanism measured on a die and
bit-exact, with the engine-side implementation committed behind `--rope-graph`.
What remains is running it in the engine, which needs the sixteen ranks the
tenant holds.

## The full list, and attention is on it

| kernel | a round | dispatch ms |
| --- | --- | --- |
| `aclnnInplaceCopy_CastAiCore_Cast` | 530 | 6.03 |
| `aclnnFusedInferAttentionScoreV4` | **81** | 0.93 |
| `aclnnAbs_AbsAiCore_Abs` | 78 | 0.89 |
| `aclnnMuls_Muls_Muls` | 75 | 0.85 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 63 | 0.72 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 38 | 0.43 |
| `aclnnLayerNormWithImplMode` | 24 | 0.28 |
| `aclnnIndex_IndexCheck_IndexCheck` | 19 | 0.22 |
| total | 1108 | 12.62 |

    the rope graph captures 580 of them, 6.61 ms
    the remainder is 528 calls, 6.01 ms

## Attention runs eager, once a layer

Eighty-one FusedInferAttentionScore calls a round is 78 layers plus three, so
**every layer's attention is an eager call** rather than part of a super kernel.
Each pays 11.386 us of dispatch and, from today's measurement, about 120 us of
the call's own argument handling -- seventeen parameters including Python lists
rebuilt every time.

`Abs` at 78 and `Muls` at 75 are also once a layer: the dynamic quant's absmax
and scale. `Matmul` at 63 is most of the projections.

So of the 528 calls the rope graph leaves, 297 are the once-a-layer items --
attention, the quant's two parts, and the projections -- and all of them are
outside the captured graph.

## Which is a larger lever than the rope refresh

The rope graph removes 6.61 ms of dispatch. The once-a-layer items are 3.39 ms
of dispatch by the same count, but attention alone carries about 120 us a call
of argument handling that the dispatch figure does not include: 78 calls times
120 us is **9.4 ms a round**, which no lever has touched and which capture
removes by construction.

That is the largest single item left in the host account, and it follows from
the same mechanism already implemented for the rope refresh.

## Confirmed: no super kernel contains attention

The same capture shows 637 super kernel calls a round and 81 eager attention
calls, which is only consistent if the super kernel covers part of the model.
Grouping its 1911 calls by the operator each name ends in:

    GatherV3               234
    QuantBatchMatmulV3     234
    GroupedMatmul          225
    QuantBatchMatmulV3     225
    ForeachAddList         225
    QuantBatchMatmulV3      63
    DynamicQuant            48 + 48
    aiv_all_reduce           3
    super kernels naming attention: 0

Each group is 225 to 234 calls over three rounds, so 75 to 78 a round -- once a
layer. The super kernel fuses the MoE side: the routing gather, the quantised
matmuls, the grouped matmul, the expert combine, the quantisation. **Attention
is not in any of them.**

So the eager attention call is structural rather than an artefact of this
capture's width. Its 120 us of argument handling a call, 78 times a round, is
9.4 ms that capture would remove -- and it is the largest item in the host
account that no lever touches.

## And it explains the 637

Eight or so super kernels a layer rather than one, grouped by operator type,
which is why the earlier reading of "78 super kernels at 210 us" was wrong in
both factors: there are 637 calls a round at 52.7 us average, and the 210 us
names were the few heavy ones.

## Withdrawn: the profiler cannot tell eager from replay

The profiled run carries `target_graph: true`, `captures: 1` and `replays: 18`
and `36`. So its forward was replayed from a graph.

A profiler reports device kernels, and a graph replay is a recorded sequence of
exactly those kernels. So `aclnnFusedInferAttentionScoreV4` appearing 81 times
says nothing about how it was dispatched -- and during a replay no aclnn host
path runs at all.

Which withdraws two readings from this page:

- The 1108 "eager calls a round" and their 12.62 ms of dispatch. The count is
  real; the word eager was an inference from the name prefix, and the prefix is
  the operator's name whether it was launched or replayed.
- The 9.4 ms of attention argument handling. 78 calls times 120 us holds only
  if each one runs the host path, and in a replay none does.

And `sk_*` is a compiler-fused kernel, not "the captured graph" -- so attention
being absent from every super kernel means it was not fused with its
neighbours, not that it was outside the graph.

## What survives

The rope refresh lever, because it was measured directly rather than inferred:
on one die, the same 265-destination refresh costs 7588.89 us eager and 336.63
replayed, bit-exact. That is a measurement of two paths, not a reading of a
profile.

And the per-item in-graph ratios from the same kind of direct comparison. What
does not survive is any claim about which path the engine took, from a profile
that cannot distinguish them.

## The rope lever depends on hoisting, which the baseline does not use

`_hoisted[width]` is populated only under `graphed and self.hoist_metadata`,
and `refresh_hoisted_rope` returns immediately when the entry is absent. The
best arm carries `hoist_metadata: false`, so in the baseline the refresh never
runs and the 530 copies it would issue do not exist.

So `RopeGraph` is not an independent lever. It captures work that only happens
when hoisting is on, and an arm passing `--rope-graph` alone measures nothing.
The watcher's arms were rewritten accordingly:

    baseline
    hoist-rope   --hoist-metadata --rope-graph
    narrow       --narrow-bank
    hoist        --hoist-metadata --hoist-forward-context
    all          all four
    aicpu        --ai-cpu-collectives
    steady       --profile-last

Which also means the capture's 530 `aclnnInplaceCopy_CastAiCore_Cast` calls a
round came from somewhere other than this refresh, or from a run that did
hoist. Either way they are not attributable to the baseline's rope path.

## What that leaves of the two host-side levers

The hoist pair removes a per-round metadata rebuild measured at a floor of
130.62 us. The rope graph removes the copies that hoisting then introduces,
measured at 22.54x on one die. They are one lever in two parts, and the arms
now measure them that way: `hoist` alone against `hoist-rope`.
