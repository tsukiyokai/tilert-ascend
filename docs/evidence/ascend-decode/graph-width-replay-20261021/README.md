# A wider captured graph replays slower, and the same operators eagerly do not

2026-10-21, a3-84, one die, no model. Eight layers of a layer's projections and
norms, captured and replayed at each row count.

    rows   eager us   replay us   replay a row
       8      568.7       527.0
      12      570.4       532.8          1.449
      16      558.6       545.2          2.284
      32      554.4       582.9          2.332

Eager is flat and slightly decreasing -- 568.7 to 554.4 across a fourfold row
range. Replay rises monotonically, 527.0 to 582.9.

Scaling 2.3 us a row over eight layers to 78 gives about 22.7 us a layer a row,
or 1.77 ms a row a round. The round model fits 0.814; this probe runs eight
layers of an incomplete operator set and extrapolates, so being high by about
twice is the expected direction.

## This answers the question the queue was going to answer

The round's second term is the graph's, not the operators'. Capture fixes the
shapes into the task list, and a task list built for more rows costs more to
replay even when the operators themselves are flat -- larger argument blocks
per task, or tiling chosen for the wider shape.

Measured without the whole machine, on a synthetic graph, because the question
is a property of capture and replay rather than of the weights.

## What follows for the drafter

Confirmed rather than conjectured: a tree's budget is expensive because of the
captured width. So the captured width should match the tree's typical node
count, not its ceiling.

    adaptive tree          13.1 nodes on average, 64 ceiling
    capture at 16          pays width 16 every round
    capture at 64          pays width 64 every round for the same 13 nodes

And the drafter should fill the captured width rather than exceed it, which is
what the adaptive budget already does -- its value is now mechanical rather
than empirical.
