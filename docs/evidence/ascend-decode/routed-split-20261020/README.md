# The routed experts want the expert axis split, not the column axis

Three cheap measurements on the fused shared-expert kernel, which already has
the routed experts' shapes, settle how to structure them — and correct an
arithmetic error of ours on the way.

## The cube does not care how many rows it gets

Gate-and-up projection only, four blocks, by row count:

| rows | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- |
| µs | 10.742 | 10.709 | 10.717 | 10.852 | 11.416 |

Sixteen times the rows for 6% more time. The cost is the weight stream and the
fixed cost, not the arithmetic — which is the whole of the batch=1 problem in
one table.

It also closes a route before it was built. A routed expert at width 8 holds
about 1.45 of the 64 permuted rows, so a per-expert matmul was the obvious
shape; at roughly 10 µs each and 44 activated experts a layer that is 440 µs,
against the vendor's `GroupedMatmul` plus `GroupedMatmulSwigluQuantV2` at
165 µs a layer. Per-group matmuls are not the win.

## Cores are what it wants

Same projection, 8 rows, 1.57 MB of INT8 weights whichever way it is split:

| blocks | 1 | 2 | 4 |
| --- | --- | --- | --- |
| µs | 20.610 | 13.120 | 10.795 |
| GB/s | 76 | 120 | 145 |

**Our earlier reading of this was wrong.** Calling 10.795 µs "31x off the
bytes" divided the per-core traffic by the die's aggregate 1.155 TB/s. A die's
streaming read shared over 24 AI cores is about 48 GB/s each, so four cores
reaching 145 GB/s together are not starved — they are near what four cores
can have. The cube is short of cores, not of efficiency, and the fix is more
of them rather than a better inner loop.

Our channel split caps at four blocks, because a `DataCopy` moves whole
32-byte data blocks and at eight blocks a block's INT8 slice is 16 bytes.
That cap is an artefact of splitting the column axis.

## So split the expert axis

One kernel, the activated experts spread over all available cores, each core
taking its experts at full `N` = 256. At 44 experts a layer that is about two
experts a core over 24 cores, 2.2 MB of weights each.

Two consequences, one of them a simplification we did not expect:

A permuted row belongs to exactly one expert and therefore to exactly one
core, so the per-token maximum for the intermediate's dynamic quantisation is
**core-local**. The routed chain needs no grid barrier at all — none of the
two `SyncAll<false>()` calls the shared expert needs, and none of the
cross-block reduction. It is the simpler kernel of the two.

And the ceiling is the byte floor: 44 experts at 1.18 MB of W4 is 52 MB a
layer, which at the die's measured streaming read is 45 µs. The vendor spends
165. Whether 24 cores actually reach that rate is unmeasured — four reach
145 GB/s and the scaling past four is exactly what the expert split would
test.

## The core count is a hard ceiling on the launch

The scaling measurement past four blocks did not complete: a sweep of 1, 2, 4,
8, 16 and 32 blocks hung and was killed. 32 exceeds the die's physical AI core
count, and the documented behaviour is that the framework's multi-round
scheduling then inserts a sync that does not pair and the kernel hangs. So the
expert split has a ceiling at the core count, not merely a preference for it,
and the sweep has to stop at 16.

Rerunning it needs the die to itself, and the whole machine freed up at the
same moment — the queue that has been blocked all session took priority.

## Next

Build it: one MIX kernel, expert-split, per-core dynamic quantisation, no
barrier. Measure against the vendor's 165 µs a layer, and measure the 24-core
bandwidth the plan assumes.
