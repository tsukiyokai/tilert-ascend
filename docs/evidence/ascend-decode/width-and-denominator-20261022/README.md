# The denominator is capped, so the target is the layer

TPOT is a round divided by the tokens it accepts, and both terms saturate in
the verification width. This prices the width and, with the bank lever gone,
says where the remaining work has to come from.

## Width is worth 0.45 ms, not a structural win

Accepted tokens follow `(1 - a**(W+1)) / (1 - a)`, with `a` of 0.86 fitted to
the measured 7.1 at width 32 and scaled to the engine's 5.441 at width 12. A
rank's touched experts follow `16 * (1 - (248/256)**W)`, and the bank is now
priced by that count rather than by the tensor, because the kernel skips an
empty group. Attention adds its measured 0.66 microseconds a row.

| width | touched experts | layer us | round ms | accepted | TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 5.07 | 650.4 | 50.73 | 5.441 | 9.324 |
| 16 | 6.37 | 670.1 | 52.27 | 5.845 | 8.942 |
| 24 | 8.53 | 703.6 | 54.88 | 6.186 | **8.871** |
| 32 | 10.21 | 730.8 | 57.00 | 6.289 | 9.065 |
| 64 | 13.90 | 800.3 | 62.42 | 6.332 | 9.858 |

The minimum sits at width 24 and the curve is flat from 16 to 32. Moving the
engine off width 12 is worth 0.45 milliseconds, which is a tuning result rather
than a mechanism. Accepted tokens approach 6.332 and stop; no width buys more.

## Why no drafter closes the gap either

The earlier comparison put a copy oracle at 13.474 accepted against the
drafter's 5.441 and read the difference as room. At width W no drafter can
accept more than W+1 tokens, so 13.474 is the ceiling of a width near 13 and
not a better algorithm. That is why four independent routes to it failed: they
were chasing a bound, not a method. The acceptance asymptote of 6.332 is the
real cap on the denominator.

## What that leaves

With the denominator capped at 6.332, a 2.9 millisecond TPOT needs a round of
18.4 milliseconds, which is 235 microseconds a layer against today's 650 -- a
factor of 2.8. A rank reads 59.1 MB a layer, or 41 microseconds at the measured
1442 GB/s, so 235 is 5.7 times the traffic floor and the factor is available in
principle.

None of it is in the bank, which already runs at bandwidth, and none is in the
drafter or the width. The native path's own itemization puts attention at about
210 microseconds a layer averaged over the 21 layers that compute an indexer
and the 57 that reuse one, against 14.8 microseconds of traffic -- a factor of
fourteen, and the largest single inefficiency on the path.

[`sweep.py`](sweep.py) computes the table and [`sweep.json`](sweep.json) records
it.
