# The vendor as the judge: the kernel is wrong, not imprecise

2026-10-21, a3-84, one die, sixteen experts, one row each, identity routing.

Same inputs to both sides -- random INT4 codes, random scales, random
activations:

    vendor final (16, 6144) bfloat16 absmax 26.25000
    mine         (16, 6144) bfloat16 absmax  0.00000
    max abs difference 26.250000  mean 1.419976  unequal 98304/98304

Every element disagrees. The output is not imprecise, it is absent.

## Why this replaces the host reference

The hand-written reference disagreed with the kernel about where the scale
bias goes -- the kernel computes `(product + bias) * scale`, the reference
`product * scale + bias` -- so it could not settle anything: either side
might have been the wrong one. The vendor chain takes the checkpoint's own
weights through the vendor's own packing and needs none of my conventions to
be right.

## The second broken criterion

The one numerics check that passed, against the vendor under a constant
weight, gave absmax 0.45117 on both sides, a bfloat16 ULP apart. Under a
constant weight every column's product is the same, so a column offset, a
K-axis mismatch and an index displacement all leave the result unchanged.
That mode has almost no resolving power for structural error, and it was
being read as numerical validation.

## What the pair of criteria says

Constant weights pass, random weights produce nothing. The errors that
survive a constant weight and die on a random one are the misalignments --
which axis is paired, which column is read, where a row starts -- not the
arithmetic. That is a direction, and it comes from the difference between two
criteria rather than from a guess.
