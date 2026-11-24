# A complete layer, item by item, against the sum of its parts

2026-10-21, a3-84, one die, twelve tokens, TP16 shards, 96 sorted rows.

| item | us | share |
| --- | --- | --- |
| attention, dense stand-in | 96.959 | 30.6% |
| grouped matmul | 61.910 | 19.5% |
| fused input projection | 44.783 | 14.1% |
| rms_norm | 28.660 | 9.0% |
| q_b projection | 22.735 | 7.2% |
| o projection | 21.070 | 6.6% |
| shared down projection | 21.244 | 6.7% |
| dynamic quant | 20.114 | 6.3% |
| **sum of the parts** | **317.476** | |
| run as one sequence | 304.965 | ratio 0.961 |

## What this settles

The accounting was assembled item by item and had never been checked against a
layer run as one sequence. It holds: the whole is 3.9 percent faster than the
sum, so cross-operator overlap is a few percent rather than a factor.

And it closes the round's budget. 317.476 us over 78 layers is 24.76 ms
against the 24.84 ms the fixed term becomes after the five boundary merges --
the two agree to a third of a percent, so there is no unattributed remainder
left to find.

## Which makes the conclusion about the target an accounted one

    attention        6.63 ms a round, near its own floor: a dense torch
                     implementation of the same work is 11 percent slower, and
                     fewer picks, pre-gathering and lower precision are closed
    grouped matmul   4.83 ms, flat in rows and in expert count, and CANN's own
                     is the fastest of four implementations
    six small items  13.30 ms, each 20 to 45 us on twelve rows -- operator
                     fixed cost, which is what the merges already collected

2.9 ms at a captured width of 24 needs the fixed term at 15.80 ms. The 9 ms
would have to come from the six small items, whose own work is negligible and
whose cost is the boundary -- and merging boundaries is the mechanism already
spent, at 21.04 ms.

So the target is outside the mechanisms identified, and this is now an
accounted statement rather than an inference: every millisecond of the round
is attributed to an operator measured at its shard and shape.
