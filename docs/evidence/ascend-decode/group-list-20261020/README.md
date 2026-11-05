# The expert kernel takes a group list, and pays nothing for it

2026-10-20, 187, one free die.

    counts 1   out (16, 6144)   nonzero 98304/98304     37.868 us
    counts 2   out (32, 6144)   nonzero 196608/196608    42.887 us
    counts 4   out (64, 6144)   nonzero 393216/393216    53.517 us
    counts 8   out (128, 6144)  nonzero 786432/786432    72.534 us

At one row an expert, 37.868 µs against the 37.193 the fixed-rows kernel
measured -- 1.8%, inside the noise. That is the regression this change had to
pass before a varying list means anything, and it passes: a variable count
costs nothing.

## The blocker it clears

The engine decides each expert's row count: twelve tokens at top eight scatter
ninety-six assignments over 256 experts, so most get none or one and some get
two or more. The kernel was compiled for a fixed `ROWS`, so it was correct only
for the experts holding exactly that many. This, not packaging, is why the
INT4 binding had never run inside the engine.

## Why the change was small

`ROWS` carried two meanings. One is the loop bound -- how many rows an expert
has. The other is the offset to the low INT4 half in the M-stacked activation,
which the comment at the top of the file states as
`C[r] = 16 * C'[r] + C'[ROWS + r]` and which the cube's tiling depends on.
Separating them is the whole change: the stride stays a compile-time
`M_TILE / 2`, so one tiling serves every count, and only the loop bounds
become runtime. The cube reads a whole sixteen-row tile whatever M says, so
the rows an expert does not have cost nothing to compute and the vector stages
simply do not read them.

## The curve says the kernel is not row-sensitive

Doubling the rows costs 13 to 35%, not 100%: the weight movement that
dominates -- `aic_mte2_ratio` 0.733 -- does not depend on the row count, and
only the vector stages scale. So the experts holding two or four rows in a
real round are not a problem.

## One detail that would have deadlocked it

An expert with no rows cannot be skipped. The cross-core flags are counted, so
it still has to arrive at every flag the cube half sets and waits on; it just
does no work between them.
