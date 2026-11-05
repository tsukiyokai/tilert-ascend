# The per-kernel floor collapses under graph replay

2026-10-21, a3-84, one die. One elementwise multiply over twelve elements,
eager against captured replay.

    count   eager a call   replay a call
        1      17.992 us        9.721 us
        8      11.847 us        1.675 us
       64      11.386 us        1.064 us

Eleven times. The engine decodes inside a captured graph, so the figure that
applies to it is 1.06 us a kernel, not the 28.824 us measured eager.

## What this corrects

The round was being modelled as launch count times a 28.8 us floor, which
matched: 25 kernels a layer times 28.8 is 720 us against the round's 710. The
match was a coincidence of two wrong numbers. In a graph, 25 kernels a layer
cost about 27 us -- under four percent of the layer -- so the remaining 683 us
is device time, and the launch count is not what sets the round.

## What survives, and why

The measured saving stands: the routed experts in one launch at 72.463 us
against the vendor's four-launch chain at 181.172, which is 8.48 ms a round.
But its mechanism is not the three launches saved -- those are worth 3 x 1.06 x
78, about 0.25 ms. It is the three round trips through HBM that the chain makes
and the single kernel does not: init_routing writes sorted rows, the first
grouped matmul writes the activation and its scale, the second writes the wide
product, and token_unpermute reads it back.

So the case for one kernel over a chain is bandwidth and residency, not issue
cost. That is the classical argument for a megakernel, reached after a day of
measuring the wrong quantity twice -- once as arithmetic, once as launches.

## The number to carry forward

Each vendor operator boundary inside a layer costs an HBM round trip of the
intermediate. At the decode shape the routed experts' intermediates are 96
rows: 96 x 6144 bytes INT8 in, 96 x 6144 x 2 bytes BF16 out. Three boundaries
removed measured 108.709 us a layer. That is the unit to price the remaining
boundaries with -- the attention projections, the MLA preprocess, the shared
expert -- rather than the launch count.
