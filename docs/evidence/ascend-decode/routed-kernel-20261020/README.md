# A layer's routed experts in one kernel, and what it costs

`a3-187`, one die, `trt-asc` with CANN 9.2.0. Hand-written AscendC, built by
`bisheng` at `--npu-arch=dav-2201`, checked against a host reference that
computes both INT32 products exactly and rounds where the device rounds.

## The shape, and why

The expert axis is the split. Three measurements forced it:
[the cube is flat in rows](../routed-split-20261020/README.md) -- 10.742 µs at
two against 11.416 at thirty-two -- so a matmul an expert would be 440 µs a
layer against the vendor's 165; what the cube is short of is cores, 76, 120
and 145 GB/s at one, two and four blocks; and a permuted row belongs to one
expert and therefore to one core, which makes the per-token maximum for the
intermediate's quantisation core-local. So this kernel has no grid barrier at
all -- neither of the two `SyncAll<false>()` calls the shared expert's chain
needs, nor its cross-block reduction.

Each cube block walks its own experts serially: gate and up projection at full
width, flag, dequantise and SwiGLU and requantise per token, flag back, down
projection, flag, scale to BF16. The two vector subblocks of a block share the
6144-wide epilogue; subblock zero alone runs the narrow 128-column activation
and answers the cube.

## What it costs

Steady state over fifty launches a round, three rounds, three separate
processes. Correctness is reported for every configuration, sentinel included,
so an unwritten element cannot pass as a right one.

| experts | blocks | wall, µs | against the host reference |
| --- | --- | --- | --- |
| 4 | 4 | 47.108 | 0 of 49152 |
| 8 | 4 | 85.090 | 0 of 98304 |
| 16 | 4 | 162.047 | 0 of 196608 |
| 8 | 8 | 47.755 | 0 of 98304 |
| 16 | 8 | 87.104 | 0 of 196608 |
| 24 | 8 | 125.433 | 0 of 294912 |
| **16** | **16** | **49.276, 49.279, 49.725, 49.815, 49.791, 49.666** | **0 of 196608, every run** |
| 44 | 16 | 128.234 | 0 of 540672 |
| 44 | 22 | 91.953 | 0 of 540672 |

The wall tracks experts a block, not experts: four on four blocks and sixteen
on sixteen cost the same, while sixteen on four costs three times as much.
That is the expert-axis split behaving as designed.

## The figure that matters

Sixteen experts on sixteen blocks is the per-rank shape. The profile is rank
zero's, and at TP16 with 256 routed experts a rank holds sixteen; a round at
width 32 puts 32 tokens through top-8, so 256 selections spread over 256
experts and a rank's own sixteen are each hit once or twice.

So 49.3 to 49.8 µs a layer against the vendor's 165.2 -- `GroupedMatmul-
SwigluQuantV2` at 6.793 ms over 75 layers plus `GroupedMatmul` at 5.596 --
which is 3.3 times, and within 12% of the 45 µs byte floor this chain was
given. Over a round: 3.71 ms against 12.39, saving 8.68 ms, which at
acceptance 7 is 1.24 ms of TPOT.

The comparison is conservative in one direction and incomplete in another.
This kernel also absorbs the BF16 epilogue, which the two vendor ops do not
include; it does not do the final unpermute and weighted sum, which is a
separate op in the vendor graph too.

## What is not established

Rows an expert is fixed at two, the measured average of 1.45 rounded up. The
cube is flat in rows, so the cost should hold under a real group list, but
variable rows are unmeasured and the host side of a group list is a separate
change. And none of this is integrated: the graph still runs the vendor's two
grouped matmuls, so no end-to-end TPOT has moved.

## Padding is not free, and the floor is weight streaming

Rows an expert is a build-time capacity, swept at sixteen experts on sixteen
blocks. The premise under test was that the cube is flat in rows, so padding
every expert to one capacity would cost nothing and retire the host side of a
group list.

| capacity | wall, µs | against the host reference |
| --- | --- | --- |
| 2 | 49.289 | 0 of 196608 |
| 4 | 51.969 | 0 of 393216 |
| 8 | 55.263 | 0 of 786432 |
| 16 | 63.308 | 0 of 1572864 |
| 32 | 79.820 | 0 of 3145728 |

The premise is wrong, and visibly so: 1.62 times from two rows to
thirty-two. The cube is flat in rows, but this kernel is not only cube -- the
epilogue walks rows by 6144 and is linear in them. The fit is
47.25 µs + 1.02 µs a row, over the whole range.

That fixed 47.25 µs is weight streaming. Each block pulls its expert's
2.36 MB of INT8 weights, sixteen blocks at once, so 799 GB/s against the
die's 1.155 TB/s: 69% of peak. For batch-one decode that is where the floor
belongs, and it is the thing to attack next, not the per-row term.

So capacity must follow the real row counts rather than the worst case. A
round at width 32 puts 32 tokens through top-8, which is 256 selections over
256 experts, so a rank's sixteen average one row each -- but a popular expert
can take several, and a capacity sized for 32 would pay 62% over the
measurement above. The group list the routing op already emits gives the
actual counts, so variable rows cost 47.25 + 1.02 × the busiest block's rows,
which beats every fixed capacity.


## Hinting the weight out of L2 buys nothing here

The vendor's grouped matmul calls
`weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)` when
`numBlocksM == 1`, which is the decode condition and the only condition this
kernel has. The reasoning is sound and sharper than it looks: a tile read
exactly once can only evict something that is reused, and an eviction writes
the old line back to global memory before a new one enters, so an unhinted
streaming read spends bandwidth displacing what you wanted resident.

Measured on the weight tensors of both projections, sixteen experts on
sixteen blocks, three separate processes: 49.238, 49.517 and 49.611 µs against
the unhinted 49.276, 49.725 and 49.791. Inside the noise, and correctness
unchanged at 0 of 196608.

So the hint is a no-op for this kernel and the line is gone. The negative is
worth more than the line was: this kernel streams 37.7 MB a layer at 66% of
the die's bandwidth, and whatever accounts for the missing third, L2
eviction by its own weight stream is not it.

## It is not bandwidth bound. It is scalar bound.

`msprof --aic-metrics=PipeUtilization` on the same binary, sixteen experts on
sixteen blocks:

| pipe | time, µs | ratio |
| --- | --- | --- |
| scalar | 27.860 | **0.625** |
| MTE2 | 32.826 | 0.733 |
| fixpipe | 13.924 | 0.311 |
| MTE1 | 12.563 | 0.284 |
| MAC | 4.845 | **0.109** |
| vector (AIV) | 0.622 | 0.013 |

The cube spends 62.5% of its time in the scalar unit and 11% doing the
multiplication. The vendor's own MTE2-bound cases read 0.93 and 0.99 here;
this reads 0.733, so MTE2 is heavy but not the wall.

That overturns the account this page has been giving. "66% of the die's
bandwidth" described a kernel that was not bandwidth limited: it is limited by
the `Matmul` library's scalar control path, with the weight stream partly
hidden behind it. Three things fall into place at once -- the L2 hint buying
nothing, INT4 halving the weight bytes for only 21% rather than the ~45% a
bandwidth-bound kernel would have given, and the wall being nearly flat in
block count from eight to sixteen.

The scalar work is per invocation, and this kernel invokes a lot: `Init`,
`SetOrgShape`, `SetTensorA`, `SetTensorB`, `IterateAll` and `End` for each of
two projections for each of sixteen experts, so thirty-two constructions and
teardowns against 4.845 µs of actual MAC. `ASCENDC_CUBE_ONLY` is defined
before the include, so the KFC messaging the vendor measures at 15.02 of
17.85 µs is already gone; what remains is the library's own control flow.

So the next lever on this kernel is not bandwidth and not cores. It is either
hoisting the `Matmul` object out of the expert loop, or driving L1 and L0 by
hand -- `DataCopy`, `LoadData`, `Mmad`, `Fixpipe` -- which this project has
already proven end to end on the INT4 probe.

## Fixing the tile sizes: 16% from the scalar diagnosis

If the cost is scalar and the scalar work is per iteration inside
`IterateAll`, then fewer and larger tiles mean fewer address computations.
`SetFixSplit(baseM, baseN, baseK)` is the lever. Swept at sixteen experts on
sixteen blocks, correctness unchanged at 0 of 196608 throughout:

| baseN | baseK | wall, µs |
| --- | --- | --- |
| the library's own choice | | 49.692 |
| 256 | 128 | 49.874 |
| 128 | 256 | 42.095 |
| **128** | **512** | **41.677** |
| 64 | 512 | 56.410 |

128 by 512 of INT8 fills L0B exactly at 64 KB, so it is the corner rather
than a sweet spot, and the shape of the curve says what the profile said:
widening N without deepening K does nothing (256 by 128 is the library's own
figure), and deepening K while narrowing N is worse than either.

Two things that did not work, for the record. Hoisting the `Matmul` objects
out of the expert loop -- two constructions a launch instead of thirty-two --
gave 49.169, 49.587 and 49.574 µs against the baseline's 49.276 to 49.815, so
the scalar cost is not in `Init` and `End`. And the L2 hint above gave nothing.
The cost is inside the iteration, which is why the tile size is what moved it.

With the tiling fixed:

| experts | blocks | wall, µs | before | vendor |
| --- | --- | --- | --- | --- |
| 4 | 4 | 38.761 | 47.108 | |
| 8 | 8 | 40.000 | 47.755 | |
| **16** | **16** | **41.990** | 49.276 | 165.2 |
| 44 | 22 | 74.555 | 91.953 | |

So 3.93 times the vendor at the per-rank shape, where it was 3.3. Over a
round: 3.15 ms against 12.39, saving 9.24 ms, which at acceptance 7 is 1.32 ms
of TPOT.


## The 3.93x is not apples to apples, and the capture says so

The vendor figure and this kernel's figure were taken at different shapes, and
the comparison has been stated as though they were the same. From the capture
itself:

| op | input shapes | µs |
| --- | --- | --- |
| `GroupedMatmulSwigluQuantV2` | `8,6144;8;256;256,4,384,16,64;256,256;256,256` | 98.950 |
| `GroupedMatmul` | `8,128;256,128,6144;256,6144;256,1,6144;;;;256;8` | 58.226 |

The weight argument declares **256 experts**, and the activation is `8,6144`
-- eight rows in total across all of them. So the pair costs 157.2 µs for
eight rows spread over however many of 256 experts the group list activates.
This kernel's 41.990 µs was measured at sixteen experts of two rows each,
which is thirty-two rows.

Four times the rows against an unknown activation count is not a ratio. Per
row it reads 1.31 µs against 19.65, which flatters this kernel and is equally
not a clean comparison, because what dominates is the weight traffic of the
experts actually activated and that is in the group list, which has not been
read.

What is defensible is narrower and stays: at sixteen experts of two rows, one
MIX kernel with no grid barrier does the whole chain in 41.990 µs, correct at
0 of 196608, and the expert-axis split makes the wall track experts a block
rather than experts. The round and TPOT figures derived from the 3.93x --
3.15 ms and 1.32 ms -- are withdrawn until the activation count is read from a
capture and both sides are measured at the same shape. One `torch.load` of
`group_counts` settles it.
