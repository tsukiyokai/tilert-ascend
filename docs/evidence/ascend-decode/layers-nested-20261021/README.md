# Nesting layers inside one launch pays the tax N times

2026-10-21, a3-84, one die, one process, decode shape.

    vendor                            69.388 us
    mine, 1 layer in one launch       71.839 us
    mine, 2 layers in one launch     124.402 us   1.73x
    mine, 4 layers in one launch     230.181 us   3.20x
    mine, 8 layers in one launch     442.088 us   6.15x

Eight layers cost 6.15 times one, not once. So one launch covering several
layers does not amortise the per-layer cost; it saves about 23 percent, which
is the launch and the entry, and nothing more.

## What this corrects

Yesterday's reading of the same tax -- that it is a per-execution constant and
that one execution over several layers would pay it once -- is wrong. The
constant is independent of rows and of weight size but it accumulates with
layers, so it lives inside each layer's work rather than at the operator's
entry.

The mechanism with that shape is the grid barrier. The stage-major kernel takes
four `SyncAll` a layer, and nesting layers multiplies them by the layer count:
4N barriers for N layers, which is exactly the observed scaling.

## What it points at instead

Stage-major across layers, not layer-major across stages. Do every layer's
split, then every layer's gate projection, then every layer's activation: the
barrier count falls from 4N to 4 for the whole group. That is a loop
interchange in the kernel body, and it is the version the measurement
actually supports.

## The interchange is worse, and for a reason that invalidates it

    mine, 1 layer    71.857 us
    mine, 2 layers  212.379 us
    mine, 4 layers  778.819 us
    mine, 8 layers 3044.945 us

About N squared. The layer loop went outside the expert loop within each
stage, so every layer rescans every expert -- but the scratch was reused, so
all layers write the same addresses. The measurement answers "N layers
crowding one scratch", not "N layers each with scratch, sharing four
barriers", and it is the second configuration today that cannot answer the
question it was built for.

A correct interchange needs scratch per layer: `partial` alone is 4 MB a
layer at this shape, so eight layers is 32 MB, which is affordable.

## But the nested numbers already answer it

Eight nested layers cost 6.15x one, and nesting multiplies the barriers by
eight. If barriers dominated, 4N of them would put the scaling well above
linear; it is below. So the bulk of a layer's cost is neither the barriers nor
the launch -- it is the per-layer work, 77 percent of it, accumulating
linearly.

That caps cross-layer fusion at the 23 percent nesting already collects, not
at an order of magnitude. The round's fixed term is not reachable this way.
