# The expert sweep cannot give a fixed/slope split, and its rates are too high

`gmmsweep-20261019` on a3-187, one die, 20 calls a point after 20 warmups.
Four active-expert counts a projection, `--rows 8`.

| | e8 | e16 | e32 | e64 |
| --- | --- | --- | --- | --- |
| gate_up MB | 6.00 | 12.00 | 24.00 | 48.00 |
| gate_up µs | 64.295 | 67.854 | 75.792 | 112.340 |
| down MB | 3.00 | 6.00 | 12.00 | 24.00 |
| down µs | 68.080 | 81.495 | 84.194 | 107.709 |

## The two candidate terms are not separable here

The bench doubles the weight bytes every time it doubles the active experts,
so bytes and table occupancy move together and a three-parameter fit —
fixed, per-byte, per-empty-slot — is degenerate on four points. The test:
the coefficients of `e32 − e16` are exactly twice those of `e16 − e8`, so the
model requires 7.118 µs where 7.938 was measured. The 0.820 µs residual is
the size of one point's own min-to-p90 spread.

Separating them needs the other sweep: hold the bytes fixed and declare the
table at 8 slots against 256, which is what the 95.301-against-55.387 pair in
`gmm-fastpath-20261019` already does.

## The implied rates exceed the die

| segment | added | added | implied |
| --- | --- | --- | --- |
| gate_up e8→16 | 6.00 MB | 3.559 µs | 1768 GB/s |
| gate_up e16→32 | 12.00 MB | 7.938 µs | 1585 GB/s |
| gate_up e32→64 | 24.00 MB | 36.548 µs | 689 GB/s |

Die 0's streaming read is 1.019 TB/s by wall and 1.155 by event
([a3-hbm-20261017](../a3-hbm-20261017.json)). Two of the three segments are
above it, so the bench's `bank_note` — "the rotation puts 4 times L2 between
two reads of one bank, so a call streams from HBM" — does not hold at small
expert counts. **No absolute bandwidth should be read off this sweep.**

`down` is worse than inconsistent with HBM: its segments imply 234, 2331 and
535 GB/s in that order. A bandwidth curve cannot be non-monotonic like that,
so either its e8 or its e16 point is wrong, and byte-matched comparisons
against `gate_up` inherit the problem — they give 17.20, 16.34 and 31.92 µs
for the same three byte counts, which is not one number.

## What survives

A regime change between 32 and 64 active experts: the last doubling costs
4.6 times the one before it, at 689 GB/s against 1585 and 1768. The measured
operating point is below it — target union mean 27.121, MTP 29.729
([expert unions](execution-gap-20261018/README.md#observed-expert-unions)) —
so this is a constraint on raising the width past where the union crosses 32,
not an opportunity at width 8.

That is the whole of it. The queued "fixed/slope split" is not extractable
from this data and the sweep would have to be redesigned to vary one term at
a time.
