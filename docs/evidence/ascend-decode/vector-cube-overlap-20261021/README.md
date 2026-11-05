# Vector work hides under cube work, 59 percent of it

2026-10-21, a3-84, one die, twelve rows. A matmul (K 4096) as the cube work and
six of the layer's own vector operators, each timed from its own graph.

    cube alone                   64.11 us
    six vector ops alone         73.69
    sum                         137.81
    larger of the two            73.69
    both, one stream            105.45
    both, side stream chained   111.59
    both, issued concurrently    94.05
    vector time hidden           43.75 us   59% of the vector work

## Which is the mechanism the fold needed, without the fold

The target needs 54.5 us a layer. Issuing the vector work on a side stream
concurrently with the cube work recovers 43.75 us of 73.69 -- the same order as
what is needed, measured, and requiring no new kernel. The six operators stay
as they are; only the stream they are issued on changes.

## Three things the numbers say that the headline does not

One stream already overlaps partially: 105.45 against a sum of 137.81, so 32 us
is recovered inside a single stream -- an operator's own AIC and AIV pipelines
overlap with the next operator's without any help.

The side stream chained behind an event is **slower** than one stream, 111.59
against 105.45. Making the dependency explicit serialises what the hardware was
already overlapping, and adds the event's own cost. So the gain comes from
issuing without a dependency, not from using a second stream as such.

And the ceiling is the larger of the two, 73.69. Concurrent issue reaches 94.05,
so 20 us of the cube's 64 is still exposed. Perfect overlap would be 73.69, and
that is 64 us of cube time fully hidden -- a further 20 us a layer beyond what
this measurement already shows.

## What it does not settle

That the engine's layer has this structure available. Here the vector work is
independent of the cube work; in a real layer the norm feeds the projection
which feeds the attention, so the dependencies are real and only operators from
*different* positions in the chain can overlap. The honest next question is
which of the layer's six small operators are independent of the cube operator
running beside them -- which is a dependency reading, not a timing.
