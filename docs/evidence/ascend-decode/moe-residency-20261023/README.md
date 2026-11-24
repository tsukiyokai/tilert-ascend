# Keeping the expert union in L2 pays nothing, because it is never read twice

2026-10-23, a3-180, one die. The floor the previous entry found is MoE bytes,
and weight residency was one of three routes named for cutting them. The union
fits: 68 active experts read about 78 MB across the two grouped calls and this
die's L2 is 192 MB. `bench_gmm_shape_scaling` rotates banks precisely to defeat
that reuse, so one bank is the resident case and the rotation is the streaming
one.

| union | projection | banks | median |
| --- | --- | --- | --- |
| 68 | gate_up | 1 | 114.36 us |
| 68 | gate_up | 17 | 111.27 |
| 68 | down | 1 | 108.31 |
| 68 | down | 32 | 111.49 |
| 99 | gate_up | 1 | 138.07 |
| 99 | gate_up | 12 | 148.53 |
| 99 | down | 1 | 115.92 |
| 99 | down | 22 | 118.72 |

Four pairs, differences within 3 to 7 percent and inconsistent in sign -- the
resident case is *slower* for gate_up at 68. So L2 reuse buys nothing here.

The mechanism is that there is no reuse to buy. A grouped matmul walks each
group's weights once, so a shard is read once a call; what L2 saves is a second
read, and there is no second read. Across rounds it is no better: the next
round's union overlaps the last by about a quarter, and the weights are
streamed rather than held.

## What is left of the three routes

A narrower expert dtype has no A3 checkpoint -- the only published GLM-5.2
W4A4 is `Eco-Tech/GLM-5.2-w4a4c8-mxfp4`, whose `verified_tags` are Atlas_350.
Residency is refuted here. So the only route still standing to the MoE floor is
**a smaller union a row**: making the draft rows' routes overlap, which lives
in the drafter rather than in the operator. The teacher routes put the union at
44.35, 68.07 and 99.36 experts for widths 8, 16 and 32, so a round that drafted
rows sharing experts would pay the width-8 price at a larger width.
