# The round time has almost no variance

From the best arm's own `intervals_ns`: 1023 per-token intervals, 835 of them
under a millisecond because a round delivers its tokens together, and 188
carrying a round each.

    min    47.70 ms
    p10    50.04
    p25    50.39
    p50    50.73
    p75    51.22
    p90    52.88
    max    79.00
    mean   51.15
    p90 over p10   1.06x

## What the tightness rules out

A spread of 1.06x means there is no slow class of round to eliminate -- no
occasional domain rebuild, no straggler that lands every few rounds, no
bimodality from a cache that sometimes misses. Every round does the same work
and pays the same cost, and the single 79 ms outlier is one round in 188.

So optimisation has to take something every round pays, which is what the four
levers aim at. There is no tail to cut.

## And the acceptance is exact rather than approximate

1023 tokens over 188 rounds is **5.441 a round** -- identical to what the
offline corpus measured for `LookupDrafter` at width 12. The earlier figure of
5.407, derived by dividing the round time by the TPOT, was the rounding of two
reported means; the record's own counts give the drafting measurement exactly.

Which is worth more than it looks: it means the offline drafting work this
session did -- the saturation sweep, the four failed attacks on the oracle gap,
the twelve-stream ordering -- transfers to the engine without a correction
factor.
