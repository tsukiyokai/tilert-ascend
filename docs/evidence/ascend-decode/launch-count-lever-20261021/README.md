# Priced by launches, the MoE kernel saves 8.48 ms a round

2026-10-21, a3-84, one die, one process, twelve tokens, 96 sorted rows.

The routed experts as the vendor issues them -- init_routing, the fused
grouped matmul, the second grouped matmul, token_unpermute -- against the same
block in one launch:

    vendor chain, four launches   181.172 us   14.13 ms a round
    mine, one launch               72.463 us    5.65 ms a round
    saving                                      8.48 ms a round

## Why this reverses today's earlier figure

Priced as a replacement for the grouped matmul alone, this kernel measured
365.887 us against the vendor's 157.176 at 256 declared experts and the lever
was recorded at -16.3 ms a round. Both measurements are right; they price
different things. What sets the round is the launch count times the 28.8 us
per-kernel floor, so the comparison that matters is four launches against one,
not one operator against one operator.

Three launches saved is 3 x 28.8 x 78 = 6.7 ms by the floor model; the
measured 8.48 includes the operators' own differences on top.

## Where that puts the target

    round 55.424 - 8.48 = 46.9 ms      TPOT about 7.9 ms

at the acceptance already measured. Reaching 2.x ms needs the remaining twenty
or so launches a layer brought down the same way -- the attention projections,
the MLA preprocess, the norms, the residuals, the shared expert.

## The methodological point

The kernel was written, then measured three ways and withdrawn twice, because
the unit of comparison kept changing. The unit is the launch, and it was
established only after the per-kernel floor was measured at 28.824 us for a
twelve-element multiply.
