> Retracted 2026-10-22. The lever does not exist. The grouped matmul
> charges nothing for an expert whose group is empty, so the engine
> already pays only for what a round touches: a sixteen-expert tensor
> with five nonzero groups costs 87.23 us against 87.10 for a tensor of
> the five. This page varied the tensor size and the active group count
> together and attributed the whole effect to the tensor.
> See [`bank-skip-20261022`](../bank-skip-20261022/README.md).

# The grouped matmul's cost follows the bank, not the rows

2026-10-21, a3-84, one die, 96 sorted rows held fixed, INT8 into INT32 in a
graph.

| experts | bank MB | us | GB/s |
| --- | --- | --- | --- |
| 1 | 9.4 | 55.57 | 170 |
| 2 | 18.9 | 54.24 | 348 |
| 4 | 37.7 | 73.85 | 511 |
| 8 | 75.5 | 105.51 | 716 |
| 16 | 151.0 | 168.58 | 896 |

Linear in the bank: 7.53 us an expert over an intercept of 55.57, which
predicts 168.6 at sixteen against 168.58 measured. Bandwidth climbs to 896 GB/s
at the full bank, so the operator is reading the whole thing at close to HBM
speed whatever it was asked to compute. That is why it is flat in rows: 2.8
percent from 16 rows to 256.

## What a batch-one round actually touches

Each row picks eight of two hundred and fifty-six experts, and a rank owns
sixteen. Under uniform routing the expected number of a rank's own experts
touched, and what the bank would cost if only those were handed over:

| rows | picks | touched | us | saved | ms a round |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 0.49 | 55.6 | 113.0 | 8.81 |
| 5 | 40 | 2.32 | 65.5 | 103.1 | 8.04 |
| 12 | 96 | 5.01 | 85.8 | 82.8 | 6.46 |
| 24 | 192 | 8.45 | 111.7 | 56.9 | 4.43 |

At the acceptance ceiling of 4.289 the engine proposes a handful of rows, so a
rank sees two to five of its sixteen experts. Narrowing the bank to those is
worth **82.8 to 103.1 us a layer, 6.5 to 8.0 ms a round** -- against the 4.25
ms the target needs.

## First lever this session whose size exceeds the gap

And the mechanism is calibrated rather than assumed: the cost is linear in the
bank with a measured slope, so the saving follows directly from how many
experts a round touches.

## Three premises to check before believing it

Routing may not be uniform. A real stream's topk distribution is on this
machine and can be counted rather than modelled -- and a copy-heavy workload
repeating its own tokens may well route them to the same experts, which would
*help*, since fewer distinct experts means a narrower bank.

The narrowing must not cost a copy. `npu_grouped_matmul` takes a list of
weights, so handing it a subset should be a matter of which tensors are in the
list rather than moving any bytes -- but the vendor's path builds that list
once at load, and whether a per-round list costs anything is unmeasured.

And this is distinct from a lever already closed. "Narrowing the draft to touch
fewer experts" changed the drafter to route to fewer; this changes nothing about
routing and only declines to read experts no row selected.

## The saving depends entirely on routing concentration, which is unmeasured

| rows | routing | touched | us | saved | ms a round |
| --- | --- | --- | --- | --- | --- |
| 5 | uniform over 256 | 2.43 | 66.4 | 102.2 | 7.97 |
| 5 | favouring 64 | 7.87 | 107.3 | 61.2 | 4.77 |
| 5 | favouring 32 | 12.14 | 139.5 | 29.1 | 2.27 |
| 5 | favouring 16 | 15.52 | 164.9 | 3.6 | 0.28 |
| 12 | uniform over 256 | 5.02 | 85.9 | 82.7 | 6.45 |
| 12 | favouring 32 | 15.48 | 164.6 | 3.9 | 0.30 |
| 12 | favouring 16 | 16.00 | 168.5 | 0.0 | 0.00 |

So the lever is worth between 0 and 8 ms a round depending on a quantity this
project has never measured. At twelve rows, routing concentrated onto 32 of the
256 experts already touches a rank's full sixteen and the saving is nil.

## And the argument offered for it was wrong

This page said a copy-heavy workload repeating its tokens would route them to
the same experts, "which would help, since fewer distinct experts means a
narrower bank". That holds globally and not per rank. Concentration means the
touched experts cluster, and if the cluster falls inside a rank's own sixteen
that rank touches all of them. Fewer distinct experts globally does not imply
fewer on any given rank.

So the premise cannot be argued either way. It has to be counted: the topk
indices of a real decode round, per rank, which no saved artifact on this
machine holds -- `moe_capture.py` exists to record exactly that and has not been
run against a decode.

## Where that leaves the lever

Mechanism calibrated: cost is linear in the bank at 7.53 us an expert over a
55.57 intercept, 896 GB/s at the full bank, flat in rows. That part is
measured and stands.

Size unknown: 0 to 8 ms a round, set by the routing concentration. One capture
of a decode round's topk indices settles it, and that capture is the same one
the queued arms need the machine for.

## The router's own bias says routing is not clustered

`mlp.gate.e_score_correction_bias` is 256 trained values added to the gate's
scores before the top eight are taken -- the model's own load-balancing term,
readable from the checkpoint with no machine at all.

| layer | min | max | std | top8 - median |
| --- | --- | --- | --- | --- |
| 10 | 6.8174 | 7.0632 | 0.0403 | 0.0460 |
| 11 | 6.8044 | 7.0193 | 0.0343 | 0.0460 |
| 13 | 6.0188 | 6.2417 | 0.0243 | 0.0200 |
| 14 | 5.9079 | 6.1098 | 0.0271 | 0.0210 |
| 17 | 7.5610 | 7.7459 | 0.0264 | 0.0200 |

The range is 0.02 to 0.03 across 256 experts, so the bias barely distinguishes
them: it is a near-constant term, which is what a well-trained balancing
correction should be. Counting the experts it leaves within one standard
deviation of the eighth-best gives 89 to 162 of 256 by layer.

Priced over that contention set, five rows of eight picks each:

| contending | touched | us | saved | ms a round |
| --- | --- | --- | --- | --- |
| 89 | 2.02 | 63.3 | 105.2 | 8.21 |
| 130 | 2.16 | 64.3 | 104.2 | 8.13 |
| 162 | 2.22 | 64.8 | 103.8 | 8.09 |

Insensitive: 8.09 to 8.21 ms across the whole range. Forty picks spread over at
least 89 experts touch about two of a rank's sixteen however the 89 are chosen.

## So the lever's size narrows from 0-to-8 to about 8

To zero it, routing would have to concentrate below 32 experts, and the
narrowest contention set the bias permits is 89.

What this does not cover: the bias is one of two terms. The score itself is
`hidden @ gate.weight`, and a token's scores across layers may correlate in a
way the bias cannot show. So concentration is excluded on the bias side and
unmeasured on the score side -- which needs hidden states, and the one saved
tensor on this machine holds a single row.

## A probe bug worth recording

The first pricing printed 8.04 ms for every layer because its touched-expert
expression contained `contending / GLOBAL * (1 / contending)`, which cancels
the quantity being varied. A constant column across a swept parameter is the
signal, and it was visible before the numbers were believed.

## Narrowing without a copy works, and the obvious way does not

One die, 96 rows, 16 experts narrowed to 6, eager:

    full bank, 16 experts                   220.36 us
    gather of 6 experts, alone              703.35
    grouped on the already-gathered 6       137.07
    gather then grouped, together           779.40
    contiguous slice of 6, no copy          153.17
    a list of 6 per-expert tensors          149.40

`bank[touched]` costs 703 us on its own -- three times the whole saving, and
together with the matmul it is 779 against the full bank's 220. Advanced
indexing is dead as a mechanism, and by a wide margin rather than a near miss.

But `npu_grouped_matmul` accepts **a list of one tensor per expert**, at 149.40
us against the full bank's 220.36. That is the narrowing with no bytes moved:
the weights are held as sixteen tensors and a round passes only the ones its
rows touched. **70.96 us a layer, 5.5 ms a round.**

A contiguous slice also works, 153.17, but it needs the touched experts to be
consecutive, which they are not.

## What is settled and what is not

Settled: the mechanism is linear in the bank, the bias excludes clustered
routing, and a zero-copy narrowing exists and is accepted by the operator.

Not settled: these figures are eager, so the in-graph saving is unmeasured --
and the in-graph table put the grouped matmul's capture ratio at 0.877, so
replay changes little for this operator specifically. And the engine holds its
expert weights as one fused tensor per layer after
`process_weights_after_loading` casts them to fractal NZ, so handing the
operator sixteen separate tensors means intervening where
`routed_expert.install_before_conversion` already does.

## Bit-exact on one die, across the shapes that would break it

Narrowing drops the slots with no rows and rebuilds the group list from the
survivors. The group list is also what tells the operator where each slot's
rows begin, so a wrong narrowing shifts every row after the first gap -- which
reads as a model that still runs.

| case | rows | kept | verdict |
| --- | --- | --- | --- |
| every expert holds rows | 96 | 16 | bit-exact |
| six hold rows, scattered | 72 | 6 | bit-exact |
| two hold rows | 96 | 2 | bit-exact |
| the first is empty | 96 | 8 | bit-exact |
| the last is empty | 96 | 8 | bit-exact |

The first case takes `narrow`'s early return and so tests the bypass rather
than the narrowing; the other four are the real ones, and they cover a
scattered pattern of gaps, a leading gap and a trailing gap -- the three ways
an off-by-one in the group list shows up.

So the lever is semantics-preserving, not merely survivable.

## The engine already knows which experts were touched, then discards it

`vllm_ascend/ops/fused_moe/moe_utils.py:cumsum_group_list` converts between
three group-list conventions, and type 2 is a **sparse** one: pairs of expert
id and token count, listing only the experts that have rows. Converting it to
type 0 expands that into a dense array of `expert_num` entries, filling the
slots nobody mentioned with `active_num`:

    cumsum_group_list = torch.full(size=(expert_num,), fill_value=active_num)
    for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
        if end > start:
            cumsum_group_list[start:end] = tokens[i]

So an untouched slot ends up equal to its neighbour, whose first difference is
zero -- which is what `expert_counts` reads. All three conventions make the
empty slots visible: type 1 holds zeros directly, type 0 holds them as equal
neighbours, type 2 omits them.

## Correction: the sparse path is not ours

Every call site passes only the two conventions and omits `active_num` and
`expert_num`, so the type 2 branch would build a zero-length tensor and cannot
run at all on these quantisation methods. The sparse form belongs to another
backend.

So the claim that the engine receives the touched set in sparse form and throws
it away does not hold for the W4A8 path. What holds is narrower and sufficient:
the group list arrives as cumulative counts or as counts, and an empty slot is a
zero or a repeat of its neighbour -- never a negative step, which is now an
assertion in `expert_counts` rather than a case silently read as empty.

## The device-side price: 7.79 ms a round, and the curve is superlinear

Profiled rather than timed around a synchronise, 96 sorted rows, one kernel a
call:

| experts | device us | ms a round |
| --- | --- | --- |
| 1 | 19.97 | 1.56 |
| 2 | 18.26 | 1.42 |
| 4 | 19.27 | 1.50 |
| 8 | 49.64 | 3.87 |
| 16 | 118.09 | 9.21 |

**Narrowing sixteen to two saves 99.83 us a layer, 7.79 ms a round** -- against
a device-side round of 17.84 ms of compute, which is 44 percent of it.

The curve is not the line the host timings fitted. One, two and four experts
cost the same 18 to 20 us; eight jumps to 49.64 and sixteen to 118.09. That is a
cache effect rather than a bandwidth one: a small bank fits on chip and a large
one is re-read from HBM, which also explains why the earlier linear fit at 7.53
us an expert worked over the range it was measured on and misses the shape.

So a round touching two to five of a rank's experts sits in the flat part of the
curve, where the bank is nearly free.

## Four prices for one lever, and which to use

    standalone, host-timed, in a graph     59.4 us a layer
    standalone, host-timed, eager          70.96
    inside a host-timed TP16 layer         91.37
    profiled device time                   99.83

The device figure is what a captured layer pays, so 7.79 ms a round is the one
to carry. The others were all measured around a synchronise and each carried
part of the host tax that a graph removes.

## At the row count a rank actually receives

Twelve draft rows pick eight of 256 experts each and a rank owns sixteen, so a
rank receives about `12 * 8 * 16/256 = 6` rows, not the 96 the earlier
measurement used. Profiled at both:

| experts | 6 rows | 96 rows |
| --- | --- | --- |
| 2 | 17.02 us | 18.15 us |
| 16 | 114.05 | 116.95 |
| saved | 97.03 us, **7.57 ms a round** | 98.80 us, 7.71 ms |

Sixteen times the rows changes the cost by two to three percent, so the
operator is flat in rows at the shapes that matter as well as the ones first
measured, and the lever's price is 7.57 to 7.71 ms a round either way.

That is the session's best-evidenced lever: device time rather than host round
trip, the row count a rank really sees, bit-exact across five gap patterns, a
superlinear curve whose flat part is where the workload sits, and the
zero-copy form the operator accepts.

## The curve is a floor plus bandwidth, not a cache effect

Implied bandwidth if every listed expert's weights are read, at six rows:

| experts | MB | us | GB/s |
| --- | --- | --- | --- |
| 1 | 9.4 | 19.98 | 472 |
| 2 | 18.9 | 18.15 | 1040 |
| 4 | 37.7 | 19.82 | **1905** |
| 8 | 75.5 | 49.20 | 1535 |
| 16 | 151.0 | 116.95 | 1291 |

    the 8-to-16 marginal slope is 8.47 us an expert, 1114 GB/s

A3's HBM is roughly 1600 GB/s a die, so eight and sixteen experts are at
bandwidth and the marginal expert costs what reading it costs. The four-expert
point implies 1905 GB/s, above what exists -- so its four banks are not all
read, most likely because at six rows the group list gives each of them one row
and the operator shortcuts a single-row group.

So the shape is a floor of about 18 us plus bandwidth above it, not the cache
effect an earlier section claimed. What the narrowing saves is real HBM traffic,
which is why the 7.57 ms stands.

## And the first bandwidth reading was a units error

The first pass divided MB per microsecond by a thousand once too often and read
every point as 0.00 TB/s, then concluded from the pattern that small banks were
cached. MB per microsecond is GB per millisecond; the correction is one factor
of a thousand and it changes the mechanism, not the lever's price.

## At the checkpoint's real widths, the lever is three times larger

The probe used `(experts, 6144, 1536)` int8, 9.4 MB an expert. The checkpoint
holds `gate_proj` and `up_proj` at `[1024, 6144]` and `down_proj` at
`[3072, 2048]`, all int8 with the output axis nibble-packed -- so
`moe_intermediate` is 2048, not 1536, and an expert is 18.9 MB rather than 9.4.

Profiled at six rows and the real widths:

| experts | inner | MB | us | GB/s | ms a round |
| --- | --- | --- | --- | --- | --- |
| 2 | 1024 | 12.6 | 16.65 | 756 | 1.30 |
| 2 | 2048 | 25.2 | 17.69 | 1423 | 1.38 |
| 16 | 1024 | 100.7 | 80.21 | 1255 | 6.26 |
| 16 | 2048 | 201.3 | 160.53 | 1254 | 12.52 |

Sixteen experts hold 1254 to 1255 GB/s at both widths, which confirms the
bandwidth bound, and two experts cost 16.65 to 17.69 us regardless -- a floor
of about 17 us independent of the bank.

## Which reprices the round

A layer runs the grouped matmul twice: once for the gate and up projections
together and once for the down projection. At sixteen experts that is roughly
160 us each, so about 320 us a layer; at the two a round touches it is about 35.

    narrowing saves about 285 us a layer, 22.2 ms a round

Three times the 7.57 ms this page carried, and the reason is the probe's inner
width: 1536 where the checkpoint has 1024 and 2048, and one call where a layer
makes two.

So the bank is the round's largest device item by a wide margin, and the
narrowing is the session's largest lever -- on a 50.710 ms round, 22.2 ms.

## The experts are on dim 0 of all four values

`splitting` slices each of the four values `_get_mlp_weights` returns along dim
0, which is right only if the experts are there. The checkpoint stores per
expert -- `model.layers.10.mlp.experts.0.gate_proj.weight` at `[1024, 6144]`,
`weight_scale` at `[2048, 1]`, `down_proj.weight` at `[3072, 2048]` -- so the
engine fuses them at load and the experts' axis is whatever that fusion puts
first.

This project's own `routed_expert.py` already names it: it reads
`method.w13_weight.data` and asserts `wide.shape[1] == GATE_UP`, so the layout
is `[expert][N/2][K]`, and it takes the scale as
`method.w13_weight_scale.data.squeeze(-1)`, so that is `[expert][N][1]`.
Experts first in both.

So slicing dim 0 is correct for the weights and the scales alike, which was the
last silent-failure risk in the lever: a wrong axis would have produced a
running model with the wrong experts' scales, which no shape check would catch.

## Through the installer, at real shapes, with the real operator

The earlier checks used a stand-in whose weights were a 5x2x2 arange. This one
puts the engine's own `npu_grouped_matmul` at the checkpoint's widths behind the
installed wrapper, with twelve rows scattered over sixteen slots so that three
are touched:

    the operator saw 16 experts without the install
    the operator saw  3 experts with it
    the round's group list was restored: True
    bit-exact against the full bank: True

So the wrapper reaches the operator, hands it the subset, preserves the result
exactly, and leaves the round's own group list for whatever reads it next. That
closes the two questions a stand-in built from the same assumption could not:
whether the install takes effect, and whether it changes the answer.

What remains untested is only the engine's own quant method rather than a
stand-in with its interface -- which is what the queued arms do.

## The engine's own consumer already takes a list

`apply_gmm1_act_quant` hands `w1` to
`_as_gmm_dequant_situ_quant_expert_weights`, which reads:

    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) == 1 and tensor_or_list[0].dim() >= 3:
            return list(tensor_or_list[0].unbind(0))
        return tensor_or_list
    if tensor_or_list.dim() >= 3:
        return list(tensor_or_list.unbind(0))
    return [tensor_or_list]

So a multi-element list passes straight through, which is exactly what the
narrowing produces. And the unqualified path tells us more: the engine's own
`w1` is a three-dimensional tensor it unbinds along dim 0 into one tensor an
expert. The engine already splits per expert -- the narrowing only hands it
fewer.

`_as_gmm_dequant_situ_quant_expert_scales` takes `num_experts` explicitly, and
the narrowing shortens the scale list to the same length, so the two agree.

That is the last link: the lever's form is what the engine's hot path already
expects, confirmed by reading the consumer rather than inferring from the
producer.
