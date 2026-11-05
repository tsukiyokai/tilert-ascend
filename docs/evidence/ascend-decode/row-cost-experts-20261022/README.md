# An active expert costs one read of its shard, so the union costs 0.190 ms a row

2026-10-22, a3-180, W4A8 `gate_up` grouped matmul, TP16 shard widths
(6144 -> 2 x 128), bank rotation sized to defeat L2. Two serial chains on dies
from different modules, because A3 pairs two dies a module and this path is
HBM-bound.

## The expert axis, one row a group

| active experts | median | weight read | banks |
| --- | --- | --- | --- |
| 8 | 62.041 us | 6.29 MB | 129 |
| 16 | 68.454 | 12.58 | 65 |
| 32 | 78.719 | 25.17 | 33 |
| 44 | 91.187 | 34.60 | 25 |
| 68 | 111.896 | 53.48 | 17 |
| 99 | 137.968 | 77.86 | 12 |
| 128 | 148.185 | 100.66 | 9 |
| 256 | 237.870 | 201.33 | 5 |

The slope from 8 to 256 is **0.709 us an expert**, and an expert's `gate_up`
shard is 786 KB, so the marginal expert runs at **1.08 TB/s** against the die's
measured 1.155. There is no fixed charge per expert beyond its bytes.

## The row axis is nearly free

At 68 active experts, 1 to 32 rows a group -- 68 to 2176 activation rows --
moves 111.896 to 132.918 us. Thirty-two times the activation for 19 percent
more time: the weights dominate, as the round's account already said.

| rows a group | median |
| --- | --- |
| 1 | 111.896 |
| 2 | 117.148 |
| 4 | 113.399 |
| 8 | 114.127 |
| 12 | 118.362 |
| 16 | 117.308 |
| 32 | 132.918 |

## The down projection is dearer a byte, and it was measured too

`down` is (128 -> 6144), a 384 KB shard an expert. 8 to 256 active experts:
68.864, 82.921, 77.748, 85.899, 99.175, 117.821, 136.422, 214.105 us, slope
**0.586 us an expert** -- 0.671 TB/s against `gate_up`'s 1.08. Its K is 128
rather than 6144, so the walk is skinny and the read is less efficient. Records
in `down/`.

## What that makes the verify row cost

The teacher routes give mean active experts 44.35, 68.07 and 99.36 at M8, M16
and M32, so **2.29 experts a row**. A layer makes both calls, so the pair costs
0.709 + 0.586 = **1.295 us an expert a layer**:

    2.29 experts a row x 1.295 us x 78 layers = 0.231 ms a row

## The round model's row term, now 74 percent attributed

| stage | ms a row | share of 0.814 |
| --- | --- | --- |
| attention, per-row gather feeding dense FIA | 0.356 | 44% |
| expert union, two grouped calls | 0.231 | 28% |
| TP16 collective | 0.019 | 2% |
| attributed | 0.606 | 74% |
| unattributed | 0.208 | 26% |

The two large terms differ in kind. Attention's is removable: a shared read of
the whole cache is flat in width and the engine already implements it. The
union's is bytes at the roofline, so it is only reducible by drafting rows
whose routes overlap -- and it decelerates on its own, since the union
saturates towards 256 as the width grows, which makes a wide tree cheaper a
row than a narrow one.

## Scope

Synthetic weights; values do not affect timing on this path. One shape a
process, one die, 20 batches of 20 calls. Both projections measured. The 2.29
experts a row comes from one trajectory's teacher routes, not from the running
engine, so it is the weakest term in the product.
