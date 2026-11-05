# A complete layer on sixteen ranks, at a decode round's width

Every per-layer figure this session used came from one die with no collective in
it; every collective figure came from a standalone benchmark or a
prefill-scoped capture. This is the number both halves lacked. One layer's
weights a rank -- 12 MB of attention shards and a 151 MB expert bank -- fit what
the tenant leaves.

Sixteen ranks, twelve rows, AIV with determinism:

    compute only, no collective     493.22 us
    with two all_reduce calls       870.63 us
    the collectives cost            377.42 us

    a round of 78 layers             67.91 ms
      of which compute               38.47 ms
      of which collectives           29.44 ms

## Which vindicates a figure this session withdrew

The 374 us a layer of collective cost was derived as 588 minus 214 -- the
engine's fitted term over 78 against a synthetic layer in a graph -- and then
withdrawn as an artefact of dividing a round-wide fit by the layer count.

Measured directly, a layer's two reductions cost **377.42 us**. The derivation
was right and the retraction was wrong: the quantity exists and the arithmetic
found it.

## What the total does and does not say

67.91 ms a round against a measured 50.710 means this synthetic layer is an
upper bound, not a prediction. It runs eight separate operators where the engine
fuses them into super kernels and overlaps its streams, which is exactly the 637
super kernel calls a round the census counted.

So the useful readings are the parts: collectives are 29.44 ms a round **after
AIV is already enabled**, which is 43 percent of this bound, and the naive
unfused compute is 38.47.

## And it sets the ceiling on the day's levers

The rope graph, the bank narrowing and AIV act on compute and dispatch. The
29.44 ms of collective time remains, and the five exits measured today --
smaller messages, explicit overlap, fewer calls, fewer ranks, a hand-rolled
primitive -- are all closed. At 5.441 accepted, 29.44 ms of collectives alone is
5.41 ms of TPOT, above the 2.9 target before a single layer computes anything.

That is the session's hardest finding and it is now measured rather than
inferred: the target is unreachable while every layer reduces twice.

## Pricing the reduce count, with the compute kept in

The collective floor is per call and the message is free, so the reduce count
is the only collective lever left. At 5.441 accepted:

| reduces a layer | collectives ms | + compute | TPOT ms |
| --- | --- | --- | --- |
| 2 | 29.64 | 68.11 | 12.52 |
| 1 | 14.82 | 53.29 | 9.79 |
| 0 | 0.00 | 38.47 | 7.07 |

Compute is the synthetic layer's 38.47 ms, which is an upper bound -- the
engine's own round is 50.710 against this arrangement's 67.91, because it fuses
into 637 super kernels and overlaps streams. On the census's 16.4 ms of super
kernel time instead, one reduce a layer gives 5.74 ms.

## Which corrects the previous section's arithmetic

That section divided only the collective term by the acceptance and read 2.72
ms. With the compute kept in, removing every reduce still leaves 7.07 ms on the
naive layer and the target needs both halves down.

So the honest statement: the reduce count is worth 2.73 ms of TPOT on this
bound, which is real and not sufficient. The target needs the reduce count
halved **and** the compute at the engine's fused figure **and** the drafting at
its measured ceiling, all three at once -- which is the first time this session
has priced them together rather than one at a time.

## The bank narrowing, priced inside the layer

Same arrangement, with the MoE's bank cut from sixteen experts to the two a
round touches:

    attention and its projections   308.97 us
    the MoE, 16 experts             198.88
    the MoE, 2 experts              108.66
    compute only, no collective     485.01
    with two all_reduce calls       852.05
    the same, bank narrowed         760.68
    the collectives cost            367.04
    narrowing saves a layer          91.37
    a round of 78 layers             66.46 ms, narrowed 59.33

**91.37 us a layer, 7.13 ms a round** -- half again what the standalone
measurement priced at 4.63, because the layer's MoE runs the full bank at
198.88 us and two experts cost 108.66.

## And the compute splits two to one toward attention

    attention and its projections   308.97 us   64% of compute
    the MoE                         198.88      41%

(They overlap slightly, so the shares exceed one.) The compute half's weight is
on the attention side, not the experts -- which is where the attention
redundancy dial points, though that one trades memory for a collective rather
than for compute.

So the day's two compute levers act on different halves: the bank narrowing
takes 91 us from the MoE's 199, and nothing yet takes anything from attention's
309.
