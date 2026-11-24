> Superseded 2026-10-22. The narrowing is 164.39 us a layer and 12.82 ms a
> round, not 223 and 17.41: the 258.17 it was built on was profiled at
> `inner` 1536 rather than the checkpoint's 1024 and 2048, and a round
> touches 5.07 of a rank's sixteen experts rather than two. Width 32 is
> null rather than the last step, because more rows touch more experts.
> See [`road-to-2ms-20261022`](../road-to-2ms-20261022/README.md).

# The road to 2.x ms, from today's measured quantities

## The budget is arithmetic, not a target

    2.9 ms x 5.441 accepted / 78 layers = 202 us a layer, including its reductions

Seventy-eight layers is the model's and lossless verification runs all of them,
so the per-layer budget follows from the acceptance and nothing else.

Today: a layer is 497.03 us as one graph replay of 21 kernels with the full
bank, plus 78.19 us of two reductions measured in a chain -- 575 us, a round of
44.87 ms, a TPOT of 8.25 at 5.441 accepted. The measured TPOT is 9.3794, the
difference being the round's 5.86 ms still unattributed.

## Seven steps

| step | saves us | layer | round ms | TPOT | basis |
| --- | --- | --- | --- | --- | --- |
| as measured today | | 575 | 44.87 | 8.25 | |
| narrow the bank to the touched experts | 223 | 352 | 27.46 | 5.05 | **measured** |
| fuse the router's six kernels into one | 25 | 327 | 25.51 | 4.69 | estimated |
| fold the combine into the matmul's epilogue | 20 | 307 | 23.95 | 4.40 | estimated |
| make the KV latent and rope contiguous | 23 | 284 | 22.16 | 4.07 | estimated |
| fuse the four projections | 35 | 249 | 19.43 | 3.57 | estimated |
| attention redundancy: one reduction a layer | 34 | 215 | 16.81 | 3.09 | **measured** |
| width 32 for 6.872 accepted, costing 13 us a layer | | 228 | 17.81 | **2.60** | measured, one stream |

## What each step rests on

**The bank, 223 us of the 360.** Measured: sixteen experts are 302 MB a rank
and cost 258.17 us across two grouped matmul calls at 1254 GB/s, which is the
die's bandwidth; the two a round touches cost about 35, a floor independent of
the bank. The checkpoint's router bias spans 0.02 to 0.03 across 256 experts,
so routing is not clustered and the count holds. Implemented, bit-exact on five
gap patterns, verified through the installer at real shapes.

**The four estimated steps, 103 us together, are all one thing: removing
per-operator fixed cost.** Each has measured bounds rather than a guess.

- The router's six kernels cost 36 us for arithmetic that is negligible -- a
  6144x256 matmul, a bias, a top-k, two adds.
- The combine is 26.98 us of weighted scatter-add, three times the gather that
  moves the same bytes the other way; an epilogue would not round-trip.
- The attention's 31 us floor is a strided read: the latent is 512 columns and
  the rope 64, two separate tensors, reaching 294 GB/s. Contiguous, the same
  9.4 MB is 8 us at 1200.
- The four projections read 11.9 MB, which is 10 us at bandwidth, and measure
  44.63.

So the four steps aim at the gap between a narrowed layer's 49 us of traffic
and its 215 us of time -- a factor of 4.4, of which they take about half.

**The last two steps have measured costs.** Attention redundancy removes one
reduction, and the second reduction was measured in a chain at 33.57 us (the
first costs 44.62 because it aligns the ranks and the second does not pay that
jitter again). Its price is 13.91 GB a rank of replicated attention weights,
30 percent of the 47 GiB budget, so it is a dial: 178 MB a layer buys 0.43 us.

Width 32 buys 6.872 accepted, measured on one stream with the automaton, and
costs 0.66 us a row of attention. This is the weakest link: the twelve-stream
saturated figure is 4.289 and the engine's own is 5.441, so if 6.872 does not
hold the path ends at 3.09 rather than 2.60.

## What the path does not need

Not more bandwidth: the narrowed layer's traffic is 49 us against 215 of time.
Not fewer collectives than one a layer: TP16 must sum partial sums. Not a
better drafter: four independent routes to the copy oracle's 13.474 failed and
its advantage is knowing the answer. Not fewer ranks: a rank owns 256/ranks
experts so narrowing does not make TP8 affordable.

## The honest summary

Two steps are measured and together reach 3.09 ms. Four are estimated from
measured bounds and are all the same kind of work -- fusing small operators
whose cost is their boundary rather than their arithmetic. One is a drafting
change whose acceptance figure rests on a single stream.

So 2.x is reachable on paper from quantities measured today, with the largest
single step already implemented and the remaining engineering being the removal
of per-operator fixed costs rather than any new algorithm.
