# The round, split into what the width costs and what it does not

2026-10-20. Four widths, one drafter, one machine, so a line through them
separates the two terms without a profiler.

    round(width) = 45.88 ms + 0.814 ms a row

      w 8  measured 52.787  fit 52.392  residual +0.395  tpot 11.609
      w12  measured 55.424  fit 55.647  residual -0.223  tpot 10.186
      w16  measured 58.588  fit 58.902  residual -0.314  tpot 10.481
      w32  measured 72.066  fit 71.924  residual +0.142  tpot 11.906

Residuals of a few hundred microseconds on values of fifty to seventy
milliseconds. The 45.88 ms is per-round fixed cost -- seventy-eight layers of
kernel issue, norms, router, residuals, TP reductions, synchronisation -- and
it does not care how many rows are verified. This replaces the 36.2 ms
residual the roofline page arrived at by subtraction.

## What the fixed cost permits, even granting perfect acceptance

    w12  tpot floor 4.637 ms
    w26  tpot floor 2.578 ms
    w32  tpot floor 2.248 ms
    w64  tpot floor 1.531 ms

At 45.88 ms fixed, a 2.x ms TPOT needs width 26 or more *and* every drafted
token accepted every round. So the target is unreachable without moving the
fixed cost, whatever is done to the drafter.

## What a 2.5 ms TPOT needs of the fixed cost

    accepted  5.44 at w12   fixed <= 3.84 ms   12.0x below today
    accepted  8.00 at w16   fixed <= 6.98 ms    6.6x below today
    accepted 12.00 at w24   fixed <= 10.47 ms   4.4x below today

Two legs, and the order between them is derived rather than chosen. Cut the
fixed cost first: at 45.88 ms the optimal width is small, because fixed
dominates and acceptance saturates, which is exactly why width 12 beat 16 and
32. Once fixed is near 10 ms the binding constraint becomes the 0.814 ms a
row, the optimal width moves up, and only then does a tree have the budget to
pay for the attention-mask work it needs.

## Which corrects why the tree is a later lever

I argued the tree was not worth it because the optimum width is 12 and a tree
needs budget to branch. The observation was right and the reason was wrong:
width 12 is a consequence of the 45.88 ms fixed cost, not a property of the
workload. Reduce the fixed cost and the optimal width rises with it, and the
tree's budget arrives.

## The marginal term is not bandwidth

0.814 ms a row over seventy-eight layers is 10.4 µs a layer a row. At batch
one the weights are already read, so another row ought to be nearly free in
bytes; that it costs this much says the per-row term is scalar and issue
bound, consistent with the routed kernel's own `aic_scalar_ratio` of 0.625
against an `aic_mac_ratio` of 0.109.
