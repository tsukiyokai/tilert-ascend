# An adversarial audit of the one-sided route, and what it broke

Every claim below was attacked by an independent agent with its own
measurements. Three survived, four did not, and one of the casualties was a
latent production defect rather than a mis-measurement.

## The ratio was 1.44×, not 2.84×

The numerator holds. Sweeping the round count on the same binaries that
produced it gives 130.265, 26.954, 23.373, 22.482 and 22.235 µs at 1, 25, 100,
400 and 1600 rounds — a convergence, not a constant product, so the 400-round
figure is not one launch divided by the count. Transfer is real too: the push
half scales monotonically with bytes (6.576 → 96.332 µs as hidden goes 384 →
49152) and the three parts add (17.691 + 3.407 = 21.098 against 22.482 whole).

The denominator does not. Measured directly — 156 interdependent in-place
`all_reduce` calls in one graph, TP16, float32 98304 B, AIV expansion,
production's `HCCL_DETERMINISTIC` and `HCCL_BUFFSIZE` — HCCL costs
5044.98 / 156 = **32.340 µs** per reduction, marginal 31.291. AI_CPU expansion
gives 70.054. The figure is size-insensitive in-graph as well (32.149 at
bf16 49152 B, 32.098 at fp32 24576 B).

The 63.5 µs came from a mean that one outlier group inflated: `sk_1277` alone
is 4.356 ms per round while the other 156 groups average about 36 µs, and
those groups begin with a compute kernel whose time is inside the group.
A third line agrees: the draft phase's 158 collectives cost 33.8 µs each.

So the honest ratio is 32.340 / 22.482 = **1.44×**, or 1.33× once in-graph
dispatch is charged to both sides.

**This demolishes the pricing that reordered the programme.** The one-sided
collective is worth 1.4 to 2.0 ms per round, not the 8 to 10 ms claimed. The
estimate was four to seven times too high.

## The correctness check could not fail

With each rank contributing its own index, every subset sum is an integer no
greater than 120, far below 2²⁴, so float32 never rounds and the check is
**constant on the entire symmetric group**: 200000 random summation orders
give exactly one distinct result. Reverse rank order, pairwise tree reduction
and arbitrary peer permutations all return 120.0. Inputs that can see order
split immediately — `[1e8, −1e8, 1e-3 × 14]` gives 15 distinct results over
200000 orders, and 24912 of those orders return 0.0 where rank order returns
0.014.

"Zero elements differ from HCCL" has the same defect: the compared values are
small integers exactly representable in float32, so any two implementations
agree bit for bit.

And the two claims are mutually exclusive on any order-sensitive input. The
kernel accumulates slot by slot in ascending rank; HCCL's RING-RING-NHR
accumulates each element in a rotation beginning at that shard's owner. If
agreement with HCCL survived a cancellation construction, that would
*disprove* rank-ordered bit-reproducibility.

The blind spot nearest the code: in every measurement the payload is written
once and never changes across 400 rounds, so the check cannot see plane
staleness, round crosstalk, or an implementation that communicates only in
round 0 and re-reads the same plane 399 times — which would report 120.0 and a
*lower* per-round cost.

## Graph capture proved a different mechanism than the one in use

`aclgraph_demo`'s per-round variable travels through device memory, written by
`aclrtMemcpy` each round and read through a pointer baked into the task args,
while its payload is identical across all 400 replays. So it establishes that
a round token *in memory* survives capture — which is precisely what the torch
`AllReduce` does not do. That binding passes `round_` as a kernel immediate
and increments it on the host, and **no host code runs during replay**, so the
plane parity is frozen at whatever it was at capture.

Worse, the one-sync-per-round argument needs temporally adjacent rounds to
land on different planes, and `PLANES` is 2. The draft graph captures once and
replays twice per round while the draft phase contains three reductions.
Three is odd: back-to-back replays put the same plane on consecutive rounds
with a single `sync_vec_all` between them, and the push precedes that sync.
The target graph's 157 reductions are also odd, and every graph shares one
`AllReduce` instance, one window and one counter, so the interleaving at run
time bears no relation to the parity assigned at capture.

This is a race that would have corrupted production silently, and the
worthless checksum above is why it would have stayed silent.

## The sync floor holds; the extrapolation from it did not

5.078 µs survives the fixed-overhead test in the way that matters: it is
insensitive to message size (5.016 at 96 KiB, 5.006 at 12 KiB) but *sensitive
to rank count* — 2.171, 2.891, 3.514, 5.016 µs at 2, 4, 8 and 16 PE. A cost
that grows with ranks and ignores bytes is rendezvous.

The linearity premise was wrong. At one eighth the bytes the transfer term is
3.029 µs rather than 1.58, which is 24% and not 12.5%; the fit is
0.85 + 0.1175 µs/KiB, so there is a fixed µs in the transfer itself.

The ~16.5 µs conclusion nevertheless stands, by direct proxy rather than by
that arithmetic: at hidden 384 the push half is 6.576 µs and the whole is
9.879, summing to 16.455 for the two stages a reduce-scatter plus all-gather
would run. The proxy omits reassembling 96 KiB after the gather, so the true
figure is nearer 17 µs.

## The weight floor: my retraction was the error

An independent recount gives 6.622 GB per rank per round, or 5.985 with
`--draft-projection-tp`. The 4.24 GB figure undercounted because it priced one
pass over 27.121 active experts when a round performs four forwards
(29.729 + 8 + 8 expert-layer visits), divided 2.669 GB per forward of
replicated weights by sixteen although every die holds its own copy, counted
an `embed_tokens` table that is never loaded, and missed three of the four
head reads.

The bandwidth is contested between two independent measurements and the
disagreement is real rather than an error: `1190.77` is the maximum of 1000
single-read samples whose median is 1154.58 and whose wall median is 1019.10,
and the source file says in its own scope line that it is not a measured model
bandwidth — while batching thirty reads inside one event pair gives marginal
rates of 1527–1564 GB/s. Both are right about different quantities: the
marginal rate is what the die delivers, the wall median is what a kernel
achieved. So the floor is **3.880 ms at the die's rate and 6.50 ms at the
wall median**, and which one applies is a statement about how good a kernel
can be rather than about the hardware.

Either way, "3.31 ms remain to hide everything else in" is dead. The remainder
is between 0.1 and 1.8 ms, and about zero on the wall measure.

## What the arithmetic now says

| term | ms per round | note |
| --- | --- | --- |
| budget at acceptance 3.433 | 6.866 | |
| weight stream | 3.880 … 6.50 | die rate against wall median |
| the 157 reductions, one-sided | 3.70 | against 5.08 with HCCL |
| host launch path | 5.861 | graph capture only |

Any two of these already exceed the budget, and all three sum to 13.4 ms at
the optimistic end. **No incremental combination reaches 6.866 ms.** The route
must remove the host path entirely, hide the collective inside compute rather
than merely shortening it, and run the weight stream at the die's marginal
rate — which is one change, not three, and it is the megakernel.

That is a sharper argument for the same programme than the inflated collective
pricing ever was.

## Two facts about the measurement environment

The sync floor grows with rank count roughly as its logarithm — 2.171, 2.891,
3.514, 5.016 µs at 2, 4, 8, 16 PE — and the transfer term fits
0.85 + 0.1175 µs/KiB across five sizes from 6 KiB to 768 KiB within 1.6%.

The repository's committed copy of `tp_push_kernel.cpp` was stale: it still
held `my_pe` as the peer and the deadlocking flag poll, the version already
established as never computing an all-reduce. The audit built from it and lost
three 300-second timeouts. It has been replaced with the source that produced
the measurements. A committed kernel that deadlocks is worse than none.
