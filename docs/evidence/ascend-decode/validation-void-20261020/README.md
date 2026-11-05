# The criterion was broken, so the validations do not hold

2026-10-20, a3-84, one die.

Varying the weights and the scales separately, at sixteen experts:

    random weights, random scales   out absmax 2.363561e-17
    real weights,   random scales   out absmax 1.020432e-04
    random weights, real scales     out absmax 8.377795e-33

The first row is the configuration every kernel validation today ran in.

## What that voids

The criterion used throughout was "the output is finite, nonzero, and the
right number of rows were written". An absmax of 2.4e-17 satisfies all three
and is arithmetically dead. So the following do not hold as validations:

- the variable group list, "37.868 µs and nothing unequal"
- the absorbed gather and scatter's index checks, including the negative
  control -- the control proved the index path changes the output, not that
  either output is right
- the 256-expert shape, "output fully populated"
- the stage-major restructuring's 79.206 µs against the vendor's 157.176:
  the timing stands, the correctness behind it does not

The numerics check against the vendor under a constant weight gave absmax
0.45117 on both sides, a bfloat16 ULP apart. That one is real but narrow: a
constant weight is a magnitude I chose, and the random-weight case was already
collapsing at 1e-17 without my noticing.

## What the collapse is

The one healthy row is real weights with random scales. So the arithmetic path
works at workable magnitudes, and what fails is dynamic range: the middle
quantisation normalises by a per-row max, and then three scales around 1e-4
multiply -- `gu_scale`, the per-token quantum, `down_scale`. bfloat16's
smallest normal is about 1.2e-38, and three such factors reach it.

The vendor does not have this problem because the fixpipe dequantises inside
L0C at higher precision before anything is written out. That mechanism was
recorded in this project as an optimisation opportunity; it is a correctness
prerequisite.

## The rule this gives

A pass criterion has to be able to fail for the reason you care about.
"Nonzero and finite" cannot distinguish right from 1e-17, and no amount of
repeating it across shapes adds information. Every kernel check from here
compares against a reference and reports a relative error, which is what the
host reference was being built for when this turned up.
