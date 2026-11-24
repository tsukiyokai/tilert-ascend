# The kernel is 2.33x slower than the vendor, not 1.98x faster

2026-10-21, a3-84, one die. First measurement of this kernel with its matmuls
actually running.

    16 experts    49.137 us   absmax 1.507812
    256 experts  365.887 us   absmax 5.062500
    vendor, 256  157.176 us

So 365.887 against 157.176: the hand-written kernel costs 2.33x the vendor's
fused operator at the engine's shape.

## What this replaces

The recorded 79.206 us at 256 experts, "1.98x the vendor", was measured while
the cube half skipped every expert -- it read the group counts from unified
buffer, which the cube core cannot access, got zero for all of them and ran no
matmul at all. A kernel that does no work is fast.

The derived figure, 5.8 ms a round, inverts: 78 layers of
(365.887 - 157.176) us is about 16.3 ms a round worse than the vendor.

## Why the vendor wins here

It takes fractal-NZ weights and dequantises inside L0C through the fixpipe.
This kernel reads ND and dequantises on the vector side, so every expert's
product makes a round trip through unified buffer that the vendor never makes.
That is a structural disadvantage of the current design, not a tuning gap.

## The rule

A criterion that cannot fail also corrupts the timings taken beside it: the
fastest possible kernel is one that skips the work. Correctness has to be
established before a speed number means anything, and both have to come from
the same binary.
