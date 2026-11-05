# The dispatch's gather and the combine's scatter, inside the expert kernel

2026-10-20, 187, one free die.

    counts 1   out (32, 6144)   nonzero  98304/196608   44.467 us
    counts 2   out (32, 6144)   nonzero 196608/196608   51.190 us

    identity run absmax 27.00000
    weighted rows: max abs difference 0.000000  unequal 0/98304
    gather moved the rows: plain != shifted True

## What it replaces, and what that is worth

`init_routing` 45.772 µs a layer and `finalize_routing` 55.979 -- 101.751
together. Absorbing them costs 6.6 µs on the kernel, 37.868 to 44.467, so the
layer saves about 95 µs and the round about 7.1 ms over seventy-five MoE
layers. Measured at both ends rather than projected.

## How the three lines read

Half the output rows are zero at one row an expert, and that is right: sixteen
experts take one row each and land on tokens 0 to 15, while the rest keep the
zeros the output was allocated with. A contiguous write could not produce
that, so the shape itself says the scatter is going through the index.

The weighted run at 0.5 is bit-identical to half the unit-weight run, 0 of
98304 unequal, so the weight multiplies in the right place and the atomic add
does not lose anything.

And the negative control matters most: changing `source_row` changes the
output. Without it, an indexing path that silently ignored its index would
pass both other checks.

## What is new and what was already settled

New is the indexing: each slot reads the row the dispatch chose, and each
result is added into the row its token owns, weighted, with
`SetAtomicAdd<bfloat16_t>` -- because the eight experts a token visits write
the same output row from different blocks, which is the sum `finalize_routing`
used to perform.

The arithmetic was settled earlier against the vendor's own operators under a
constant weight: both chains returned absmax 0.45117 with a maximum difference
of one bfloat16 ULP.

## The line on this project's whole MoE path

    routing absorbed       7.1 ms   measured both ends
    MoE matmuls            9.0 ms   157.2 to 37.2 us a layer
    one-sided reduce       6.5 ms   at the docstring's in-graph rate
    ----------------------------------------------------------
    together              22.6 ms   off a 55.4 ms round

That would leave about 32.8 ms and, at the measured acceptance of 5.441, a
TPOT near 6.0 ms. Every term is a single-die or few-die measurement scaled by
a layer count; none has run inside the engine, which is where the number that
counts comes from.
