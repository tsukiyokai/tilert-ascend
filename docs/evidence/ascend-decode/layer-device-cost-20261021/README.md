# A layer by device time: the grouped matmul is 56 percent, attention 22

One die, the engine's shapes, twelve rows, profiled rather than timed around a
synchronise.

| kernel | a layer | us | share |
| --- | --- | --- | --- |
| `aclnnGroupedMatmulV5` | 1 | **127.17** | 56% |
| `aclnnFusedInferAttentionScoreV4` | 1 | 50.28 | 22% |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 4 | 44.63 | 20% |
| `RmsNorm` | 1 | 3.33 | 1% |
| `aclnnDynamicQuantV2` | 1 | 3.27 | 1% |
| **a layer** | | **228.67** | |
| **a round of 78** | | **17.84 ms** | |

## Which agrees with the engine and overturns today's attribution

17.84 ms a round against the census's 16.4 ms of super kernel time -- within 8
percent, from two unrelated measurements. So this is what a captured layer
spends.

And it inverts the compute ordering. The host-timed TP16 probe put attention
and its projections at 308.97 us against the MoE's 198.88, and that was the
host tax: attention's seventeen-parameter call pays about 120 us of argument
handling that a graph removes. On the device the MoE's grouped matmul is 127.17
us and attention is 50.28.

## So the bank narrowing is aimed at the largest item after all

The grouped matmul is 56 percent of a layer's device time. At 7.53 us an expert
over a 55.57 intercept, cutting sixteen experts to the two a round touches takes
127.17 us to about 71 -- **56 us a layer, 4.4 ms a round against a device-side
round of 17.84**, a quarter of all compute.

And attention, which this session spent three experiments trying to attack, is
50.28 us a layer -- 3.9 ms a round, and its two candidate levers were measured
and excluded for good reasons that no longer matter much.

## The lesson the numbers carry

Every absolute figure taken with `perf_counter_ns` around a synchronise was a
host round trip. The ratios survived because both arms paid it; the ordering did
not. The profiler was available all along.

## Correction: a layer's MoE is two calls, 258.17 us, not one at 127.17

The layer account used a single grouped matmul at `inner=1536`, which is
neither the checkpoint's width nor the call count. A layer's MoE runs the fused
gate-and-up bank and then the down bank:

    gate_up  (16, 6144, 2048) int8
    down     (16, 2048, 3072) int8
    302.0 MB, 2.0 kernels, 258.17 us, 1170 GB/s
    a round of 78: 20.14 ms

So the layer's device time is 228.67 - 127.17 + 258.17 = **359.67 us**, and a
round is **28.05 ms** rather than 17.84.

## Which makes the round's account consistent

    device compute          28.05 ms
    156 collectives at 190  29.64
    sum                     57.69
    the measured round      50.73

The sum now exceeds the round, which is what overlap looks like -- and overlap
is what this session measured three times, where a serial pair already costs
less than its parts. The earlier account had compute at 17.84 and summed to
47.48 against 50.73, which looked tidy and was short a call.

## And it reprices the narrowing consistently

Sixteen experts cost 258.17 us a layer and two cost about 35, so the narrowing
saves **223 us a layer, 17.4 ms a round** -- 62 percent of the 28.05 ms compute
term it acts on.

The previous figure of 22.2 ms claimed more saving than the compute term
contained, which was the tell: a lever cannot remove more than the item it acts
on. That arithmetic check is cheaper than any measurement and was available the
moment both numbers existed.

## The MoE's plumbing, which the synthetic layer omitted

The capture counts three operators once a layer that the eight-operator layer
did not hold: `GatherV3` at 78 a round, `ForeachAddList` at 75, and a second
`DynamicQuant`. Profiled at the decode shape -- twelve rows, eight picks each,
ninety-six sorted rows:

| operator | us | ms a round |
| --- | --- | --- |
| the routing gather, 96 of 12 rows | 8.45 | 0.66 |
| the expert combine, 96 into 12 | **26.98** | **2.10** |
| a dynamic quant, 12 rows | 3.37 | 0.26 |
| together | 38.80 | 3.03 |

The combine is the largest: a weighted scatter-add of ninety-six rows back into
twelve, which reads 96 x 6144 bf16 and writes 12 x 6144 with a permutation
between. Its 26.98 us against the gather's 8.45 for the same traffic in the
other direction is the scatter's cost.

## The round's account, as far as it goes

    measured device compute, eight operators   28.05 ms
    the MoE's plumbing, three more              3.03
    collectives in a graph at a round's scale   5.14
    sum                                        36.22
    the measured round                         50.73
    unattributed                               14.51   186 us a layer

Down from 17.54 and still the largest unexplained term. What is left can be
bubbles between kernels inside the graph, operators still not enumerated, or
synchronisation points between device and host -- and only the queued
`--round-phases` and `steady` arms separate those.

## The graph's own bubbles: 53.10 us a layer

The same eight-kernel layer chain, profiled for device time and then timed as a
graph replay:

    device time, profiled     350.98 us
    graph replay, wall        404.09 us
    the gap                    53.10 us    4.14 ms a round

So a replay costs 15 percent more than the sum of its kernels' device time --
about 6.6 us a kernel of gap. That is the dependency wait between consecutive
kernels: each waits for its predecessor's output, and the device idles between
them.

Note the device sum here is 350.98 us rather than the 228.67 the earlier
account used, because this chain runs both grouped matmuls at the checkpoint's
widths. The two figures differ by exactly that correction.

## The round, with the bubbles counted

    device compute, eight kernels at real widths   27.38 ms
    the MoE's plumbing, three more                  3.03
    the graph's bubbles                             4.14
    collectives in a graph at a round's scale       5.14
    sum                                            39.69
    the measured round                             50.73
    unattributed                                   11.04

Down from 14.51. And the bubbles are attackable by the mechanism already in the
engine: fusing adjacent operators removes the wait between them, which is what
the super kernel does on the MoE side and does not do for attention -- the
census found zero super kernels naming attention.

## Withdrawn: the gap is not a dependency wait, so fusion does not recover it

Dependent against independent kernels in one graph, same shapes:

| count | dependent | independent | gap each |
| --- | --- | --- | --- |
| 2 | 42.63 | 27.74 | 7.45 |
| 4 | 27.23 | 27.69 | -0.11 |
| 8 | 32.07 | 32.10 | -0.00 |
| 16 | 41.56 | 41.65 | -0.01 |
| 32 | 54.92 | 55.12 | -0.01 |

From four kernels up they cost the same to within a hundredth of a microsecond.
The two-kernel row is one noisy measurement, not a trend. So **dependent
kernels do not wait** at these shapes, and fusing adjacent operators would not
recover the gap.

The marginal cost a kernel from four to thirty-two is (54.92 - 27.23) / 28 =
**0.99 us**, which matches the 1.064 us this project measured for a trivial
kernel's replay issue.

## Which leaves the 53.10 us unexplained rather than attributed

Eight kernels at 0.99 us of issue is 7.9 us, not 53.10. So the layer chain's
gap between 350.98 us of profiled device time and 404.09 us of replay wall is
not the issue and not a dependency wait.

What remains: the profiler's Duration may not count a kernel's full occupancy,
or one operator in the chain may have a wall time well above its reported
Duration. Either way the 4.14 ms a round attributed to bubbles is withdrawn,
and the round's unattributed term returns to 15.18 ms.

The mechanism claimed with it -- that fusing attention with its neighbours
would remove the wait -- is refuted, and the census's finding that no super
kernel names attention stands as a fact without a lever attached to it.

## The harness has a 27 us floor, which explains half the gap

Timing a replay plus a synchronise, with the graph holding almost nothing:

    one 8-element multiply        27.50 us
    one 2048-element multiply     27.13
    one 1M-element multiply       32.39
    an empty graph                26.58

So `perf_counter_ns` around `graph.replay()` and `torch.npu.synchronize()`
costs **26.58 us** before the graph does anything. Of the layer chain's 53.10 us
gap, 27 is the harness and about 26 is left -- 2.0 ms a round rather than 4.14,
and at the scale of a single measurement's noise.

## And it corrects an earlier reading of overlap

The two-graph test read 69.46 us for the small graph, 59.51 for the large, and
106.03 for the pair, and this page concluded the pair overlapped. Subtracting
the floor from each: 42, 33 and 79 -- and 42 plus 33 is 75 against 79. There
was no overlap; the apparent saving was the floor counted once in the pair and
twice in the parts.

Which is a general correction to today's method. Every replay figure measured
this way carries 27 us, so differences between two replays are sound and sums
of separate replays are not.

## The collective count was assumed, and the capture counts 43 a round

Every collective estimate today used 156 calls -- two a layer over 78. The
capture's own records hold 128 collective entries over three rounds, which is
**43 a round, 0.55 a layer**. So most layers issue no standalone collective and
their reductions sit inside a super kernel; the census found one `sk_*` named
`aiv_all_reduce`.

| basis | calls | us each | ms a round |
| --- | --- | --- | --- |
| two a layer, assumed | 156 | 32.77 | 5.11 |
| the capture's own count | 43 | 32.77 | **1.41** |

## Which exposes the structural flaw in the whole account

    compute, eight separate kernels   27.38 ms
    plumbing, three more               3.03
    collectives, counted               1.41
    sum                               31.81
    the measured round                50.73

The gap is 18.92 ms, and the account cannot close because its terms are not
commensurable: the engine fuses a layer into about eight super kernels and the
sum of eight separately-launched kernels is not what it spends. Every attempt
today to attribute the remainder -- to collectives, to bubbles, to eager
dispatch, to host paths -- compared a measurement of separate operators against
a fused execution.

So the steady-state capture is not one arm among seven. It is the only
measurement that can produce commensurable terms, because it reports the
engine's own fused kernels at the engine's own width. Everything else today is
a bound or a mechanism, and the mechanisms are what the levers rest on -- which
is why the bank narrowing stands at 17.41 ms while the round's composition does
not.

## A whole layer as one graph replay, which is commensurable

Fifteen kernels -- the eight operators plus the three plumbing items plus the
second quant -- captured as one graph and replayed:

    device time, profiled        417.86 us
    graph replay, wall           487.60 us
    minus the 26.58 us floor     461.02 us
    a round of 78 layers          35.96 ms
    plus 43 collectives at 32.77  37.37 ms
    the measured round            50.73 ms

**Unattributed: 13.36 ms, 171 us a layer** -- and this time both sides are the
same kind of thing, one replay of a whole layer against an engine that replays
its layer as about eight super kernels.

## The two candidates left

The engine's layer holds more than fifteen kernels. Not counted here: the
gating matmul and its top-k, the residual adds, the output projection's bias,
the router's correction. Each is small but there are several.

Or the engine's fusion costs more than separate kernels. The super kernel names
show what it fuses -- `GatherV3`, `QuantBatchMatmulV3`, `GroupedMatmul`,
`ForeachAddList` in one kernel -- and a sequence inside one kernel cannot
spread across cores the way separate kernels can. This project measured the
same shape at the cross-layer scale: eight layers nested in one launch cost
6.15 times one.

Both are testable only against the engine's own fused kernels, which is the
steady capture. But the gap is now 13.36 ms rather than 18.92, and it is a gap
between two commensurable measurements rather than between a sum of parts and a
whole.

## With the router and the residual adds, the gap closes to 10.55 ms

Six more kernels -- the gating matmul over the hidden size, its correction
bias, the top-k that picks eight of 256, two residual adds, and the output
bias:

    21 kernels
    device time, profiled        472.57 us
    graph replay, wall           523.61 us
    minus the 26.58 us floor     497.03 us
    a round of 78 layers          38.77 ms
    plus 43 collectives at 32.77  40.18 ms
    the measured round            50.73 ms

**Unattributed: 10.55 ms, 135 us a layer.** The six additions cost 36 us a
layer, 2.81 ms a round, so the first candidate holds in part: the engine's
layer does hold more than fifteen kernels and counting them narrows the gap.

But the rate says it will not close this way. Six small kernels bought 2.81 ms;
10.55 remains, which would need twenty more of the same. So the second
candidate -- that the engine's fusion costs more than separate kernels -- is
where the rest has to be, and only its own fused kernels can show it.

## The account as the day ends

    a layer, 21 kernels, one graph replay   497.03 us   38.77 ms a round
    collectives, the capture's own 43        32.77 us    1.41
    sum                                                 40.18
    the measured round                                  50.73
    unattributed                                        10.55   79 percent attributed

From 34 ms unattributed in the morning to 10.55, by aligning the measurement
with its object four times: the checkpoint's real widths, the omitted
operators, the harness's floor, and one replay of a whole layer rather than a
sum of parts.
