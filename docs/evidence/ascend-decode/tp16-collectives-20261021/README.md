# Sixteen-rank collectives are pure latency, and they are the missing 374 us

2026-10-21, a3-84, sixteen ranks on one node, bf16 over hidden 6144. Needs
dies but almost no memory, which is what a tenant holding 55 GB of each leaves.

| rows | KB | all_reduce | reduce_scatter | all_gather |
| --- | --- | --- | --- | --- |
| 1 | 12.0 | 289.62 | 292.96 | 332.53 |
| 4 | 48.0 | 292.16 | 314.89 | 309.25 |
| 12 | 144.0 | 304.12 | 277.27 | 282.01 |
| 24 | 288.0 | 290.53 | 266.67 | 286.32 |
| 48 | 576.0 | 306.08 | 268.72 | 423.56 |

Flat. Twelve kilobytes and 576 cost the same to five percent, so at decode
sizes a sixteen-rank collective is **latency, not bandwidth**: 270 to 330 us a
call whatever it carries.

## Which closes the 2.7x

The engine fits 588 us a layer; the same operators replay in 214 on one die
with no collective in them. The difference is 374 us a layer, and one
all_reduce plus one all_gather at these sizes is 575 us. The engine's
SuperKernel absorbs most of its collectives -- the standalone count is 0.26
calls a layer -- so a layer carries a fraction of that pair, and 374 sits
inside it.

So the unattributed four fifths of the round has a name and a measurement, and
it is the one thing the synthetic layer omitted all along.

## What it rules out, and what it leaves

Ruled out: narrowing the draft, reducing the width, shrinking any message. The
cost does not follow the bytes -- 48x the payload costs 6 percent more.

Left: fewer calls, or overlap. And the overlap measurement from earlier today
says vector work hides under cube work but cube does not hide under cube; a
collective is neither, so whether it overlaps compute is a separate question
this does not answer.

## A port collision worth recording

The run failed three times with `EI0020`: the tenant's job holds the NPU
socket port 16666 on every die. `HCCL_IF_BASE_PORT` does not move it --
`HCCL_NPU_SOCKET_PORT_RANGE=27000-27200` does. Sixteen-rank collective work is
therefore possible on a machine whose dies are memory-held, which is worth
knowing for every future day the machine is busy.

## Explicit overlap with compute is slower

Sixteen ranks, twelve rows, an all_reduce against a matmul:

    compute alone               89.14 us
    all_reduce alone           300.46
    sum                        389.61
    larger of the two          300.46
    serial, one stream         343.01
    concurrent, two streams    376.70
    concurrency gains          -33.69 us

Serial on one stream is already 46 us below the sum, so the hardware overlaps
part of it unasked. Putting the collective on a side stream behind an event
costs 33.69 us more than that -- the third time today the same shape appears:
making a dependency explicit serialises what was already overlapping and adds
the event's cost. The other two were vector-under-cube and the MoE fork.

## So the collective's three exits are all measured

    smaller messages   flat to five percent over a 48x payload range -- no
    explicit overlap   33.69 us worse than leaving it alone
    fewer calls        the only one left

And fewer calls is already mostly spent: the SuperKernel absorbs them, leaving
0.26 standalone calls a layer. What a round pays is largely inside that kernel,
where the count is not a host-side choice.

## Which is the honest shape of the remaining gap

The round is compute plus collectives, both now measured: 214 us a layer in a
graph and 374 us a layer of collective latency. The bank narrowing takes 59.4
from the first. Nothing measured today takes anything from the second, and its
three mechanisms are each closed by a measurement rather than by argument.

## The fourth exit: 193 us of it is independent of the rank count

One all_reduce of 144 KB, one world size a run:

| ranks | us |
| --- | --- |
| 2 | 193.60 |
| 4 | 193.11 |
| 8 | 238.27 |
| 16 | 296.21 |

Two ranks already cost 193 us. So the rank count explains 103 of the 296, and
**193 us is a fixed per-call cost** -- not bandwidth, not topology, not the
number of participants.

## Which explains why capture cannot touch it

A kernel's issue under graph replay is 1.064 us, measured earlier in this
project. A collective's fixed cost is 193, a hundred and eighty times that. So
the collective's overhead does not sit on the path a graph captures, and the
0.877 capture ratio of the operators around it is the same story: what capture
removes is host dispatch, and this is not host dispatch.

## And it prices the one structural alternative

Dropping to eight ranks would save 58 us a call and to four 103 us, at the cost
of doubling or quadrupling the per-rank compute. At batch one that trade is
less lopsided than it looks, because every operator is flat in rows and the
grouped matmul's cost follows its bank -- a rank holding twice the experts pays
7.53 us each for twice as many. Sixteen experts at 7.53 is 120 us a layer; at
TP8 a rank would hold thirty-two.

So the trade is 58 us of collective against roughly 120 us of bank, and it
loses. TP16 is the right rank count for this workload, which is now measured
rather than assumed -- and the 193 us floor is what no rank count avoids.

## The 193 us is synchronisation, not transport

144 KB out of die 0, one process, two devices, no HCCL:

| destination | us | GB/s |
| --- | --- | --- |
| die 1 | 219.64 | 0.66 |
| die 2 | 217.98 | 0.66 |
| die 4 | 244.03 | 0.59 |
| die 8 | 218.50 | 0.66 |
| die 15 | 218.51 | 0.66 |
| pinned host memory | **37.09** | |

Two readings, both useful. A die-to-die copy through torch is 218 us -- worse
than HCCL's 193 us floor, and flat across every destination, so building a
reduction from peer copies is not the way out. And the same payload reaches
pinned host memory in 37.09 us.

So moving 144 KB off a die costs 37 us. HCCL's 193 us per call is therefore
**synchronisation and setup, not transport**: five times the cost of the data
movement it performs, and independent of the rank count, the payload and the
destination.

## Which is the honest end of this thread

Four exits closed by measurement -- smaller messages, explicit overlap, fewer
calls, fewer ranks -- and now the fifth, a hand-rolled primitive, closed too:
the peer copy is more expensive than the collective it would replace.

What remains is the gap between 37 us of transport and 193 us of call: a cost
that is neither bandwidth nor participants nor host dispatch. Whatever it is
lives inside HCCL, and 78 layers of it is the largest single item in this
project's round.

## The device figure, profiled: 452 us a call, and it includes waiting

Sixteen ranks, twelve rows, one all_reduce and one all_gather a step, under the
profiler with AIV and determinism:

    AivKernel   2.0 calls a step   452.06 us each
    the pair on the device          904.12 us a step
    78 layers                        70.52 ms
    one-off kernels                  18.08 ms, excluded as the domain build

Forty of the forty-one kernel names appear once or twice across twenty steps --
the domain build -- and dividing those into a per-step figure is the error a
previous page caught in the by-type average. Only `AivKernel` recurs.

## Three figures for one quantity, and they measure different things

    native HCCL device unit          10.74 us   the link action alone
    torch, timed around a barrier   177-296 us  host dispatch and synchronise
    profiler kernel Duration         452.06 us  device occupancy including wait

None of them is wrong and none of them is the others. A collective's Duration
in a profile counts the kernel as resident until the operation completes, which
includes time blocked on peers -- the same thing that made the engine's own
profile read 858 us a call.

So the question this page has circled all day -- what a collective costs the
engine -- is not answerable by timing a collective. It needs the engine's own
rounds, where the collectives sit between compute that determines when each rank
arrives.

## What stands

The ratios: AIV against AI_CPU, 1.5 to 1.85x, measured in one method with both
arms paying the same tax. The flatness in message size and the per-call floor's
independence from the rank count, same reasoning. And the exclusions -- smaller
messages, explicit overlap, fewer ranks, a hand-rolled primitive -- each a
comparison rather than an absolute.

What does not stand is any absolute per-round collective total. The queued arms
measure the round directly, which is the only scope that answers it.

## Collectives can be captured, and a graph amortises them

Sixteen ranks, twelve rows, AIV with determinism:

    eager                       183.83 us
    captured and replayed       134.69 us   1.36x
    four in one graph, each      68.50 us   2.68x

So a collective is capturable, a single one in its own graph saves only 1.36x
because the replay keeps its own fixed cost, and four in one graph fall to 68.50
us each. The host overhead amortises with the number of operations a graph
holds.

Over a round's 156 calls:

    eager                    28.7 ms
    one a graph              21.0
    four a graph             10.7

## Which discounts what AIV should show

The engine's graph holds about 1848 kernels a round, so its collectives are
already inside a graph and already amortised -- which makes the 29.64 ms
estimate, built from the torch-measured 190 us a call, an upper bound rather
than the engine's cost.

And it discounts the AIV arm's expected effect. AIV's 1.5 to 1.85x was measured
on eager, barrier-aligned calls where the host path is fully exposed. Inside a
graph that path is amortised, so the ratio should compress -- by how much is
what the `aicpu` arm measures, and saying so before it runs is the point of
having written the prediction down.

## A round's worth in one graph: 32.94 us each, 5.14 ms a round

Sixteen ranks, twelve rows, AIV with determinism, N all_reduce calls captured
in one graph and replayed:

| in a graph | total us | us each | a round ms |
| --- | --- | --- | --- |
| 1 | 150.57 | 150.57 | 23.49 |
| 4 | 265.62 | 66.41 | 10.36 |
| 16 | 654.75 | 40.92 | 6.38 |
| 64 | 2195.52 | 34.31 | 5.35 |
| **156** | 5139.34 | **32.94** | **5.14** |

The curve converges to about 33 us a call, which is the order of the 37 us this
session measured for moving the same payload off a die. So inside a graph, at a
round's scale, a collective costs close to its transport and nothing else.

## Which overturns the round's accounting a third time

The collective term was estimated at 29.64 ms from the torch per-call figure of
190 us, then discounted to 10.7 from four-in-a-graph, and is now **5.14 ms** at
a round's own scale. Each figure was measured; each was at a different scale;
only the last matches what the engine does, since its graph holds about 1848
kernels.

So the round's account becomes

    traffic, full bank      21.01 ms
    collectives in a graph   5.14
    sum                     26.15
    the measured round      50.73
    unattributed            24.58

That 24.58 ms is neither bytes nor collectives, and it is now the largest
unexplained term in the project. The day's earlier attributions put it in the
collectives, which this measurement removes.

## And it reprices AIV to near nothing

AIV's 1.5 to 1.85x was measured on exposed eager calls. At 32.94 us a call in a
graph, the host path it improves is already gone, so the arm should show little
-- which is what the discount written before the arm ran predicted, and this
sharpens it from "compressed" to "almost nothing".

## Outside a graph, 156 sequential collectives cost 142.84 us each

Sixteen ranks, 156 points with a matmul between each, issued eagerly:

    the work alone                 4454.18 us
    work plus 156 collectives     26737.92
    the collectives cost          22283.74
    each                            142.84
    a round of 156                   22.28 ms

Against 32.77 us each inside one graph at the same count -- **4.4 times**. So
the amortisation a graph gives is not a small effect, and whether the engine
gets it decides the round's largest term.

## Which closes the round's account, under one assumption

    compute, plumbing, as measured          30.41 ms
    collectives outside a graph             22.28
    sum                                     52.69
    the measured round                       50.73

Four percent apart. With the collectives inside a graph instead, the sum is
35.52 against 50.73 and 15.18 ms is unattributed.

So the account closes if the engine's collectives are issued outside its graph,
and does not if they are inside. That is a single testable proposition, and it
decides two things at once: where the 15.18 ms went, and whether AIV's 1.5 to
1.85x applies in full rather than nearly not at all.

The `aicpu` arm measures exactly this. If it is 1.5x slower than the baseline,
the collectives are exposed and AIV is the round's second-largest lever. If it
is within noise, they are amortised and the 15.18 ms is something else again.

Written before the arm runs, as the fourth repricing of this term today.

## The proposition is answerable by reading, and the answer is "inside"

`vllm_ascend/compilation/acl_graph.py` contains no mention of hccl,
all_reduce, collective or communication. The wrapper does not special-case
them, so whatever the captured region issues is recorded -- including the model
forward's reductions.

And the capture's own kernel list carries both: 637 super kernel calls, 1108
aclnn calls and 43 `AivKernel` collectives a round, all in one
`kernel_details.csv` produced from a run whose record says `target_graph: true`
with replays. Compute and collectives appear in the same list from the same
replayed region.

So **the engine's collectives are inside its graph**, the round's collective
term is 5.11 ms rather than 22.28, and the previous section's claim that the
account closes under the exposed assumption is withdrawn: the assumption is
false.

Which puts the 15.18 ms back as unattributed and reprices AIV to near nothing
for the second time -- the host path it improves is amortised away inside the
graph, as the four-in-a-graph and 156-in-a-graph measurements showed.

The `aicpu` arm now has a prediction rather than a question: it should land
within noise of the baseline. If it is 1.5x slower, this reading is wrong and
something in the engine takes the collectives out of the graph.

## In a chain, a reduction a layer costs 44.05 us -- more than measured alone

Eight layers on sixteen ranks, each a matmul and a grouped matmul, captured as
one graph and replayed, with and without a reduction a layer:

    no reduction a layer      37.34 us a layer
    one reduction a layer     81.40 us a layer
    the reduction costs       44.05 us a layer    3.44 ms a round

Against 32.77 us measured for 156 collectives alone in one graph. The
difference is the alignment: in a chain every reduction makes each rank wait
for the slowest arrival, and the compute between them lets the arrivals drift.
Measured in isolation they all arrive together.

## Which closes the round to 83 or 90 percent

    a layer, 21 kernels, one replay          38.77 ms
    one reduction a layer, in a chain         3.44
    sum                                      42.21
    the measured round                       50.73
    unattributed                              8.52   83 percent

With two reductions a layer -- after the attention and after the MoE, which is
what TP16 needs -- it is 6.88 ms, a sum of 45.65 and 5.08 ms unattributed, 90
percent. The capture's 43 standalone calls a round are the ones outside a super
kernel; the super kernel absorbs the rest, so two a layer is the right count
and 43 is the visible remainder.

So the day's final account attributes 83 to 90 percent of the round, and the
collective term is 3.44 to 6.88 ms rather than the 29.64 it began as -- with
the correction coming from measuring in a chain rather than alone.

## Two reductions a layer, measured directly: 6.10 ms a round

Eight layers on sixteen ranks in one graph, with the reductions placed where
TP16 needs them -- after the attention's projection and after the MoE:

| reductions a layer | us a layer | added | ms a round |
| --- | --- | --- | --- |
| 0 | 38.30 | 0.00 | 0.00 |
| 1 | 82.92 | 44.62 | 3.48 |
| 2 | 116.48 | 78.19 | **6.10** |

The second reduction adds 33.57 us against the first's 44.62, so it is cheaper:
the first aligns the ranks and the second does not pay that jitter again.
Doubling the one-reduction figure would have overstated it by 13 percent.

## The day's final account

    a layer, 21 kernels, one graph replay    38.77 ms
    two reductions a layer, in a chain        6.10
    sum                                      44.87
    the measured round                       50.73
    unattributed                              5.86   88 percent attributed

From 34 ms unattributed in the morning to 5.86, and the collective term from
29.64 ms to 6.10 -- both by moving the measurement to where the cost sits
rather than where it was convenient.

## A failure worth naming

The probe first died with `NPUGraph: ERROR -- capture failed: NameError: name
'reduce' is not defined` on eleven ranks at once, which reads as a capture or
communication fault. It was a stale parameter name after a rename. A Python
error inside a capture is reported as a capture failure, so the first place to
look is the traceback rather than the collective.
