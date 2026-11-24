# Sparse attention's 73 µs: four explanations measured and excluded

`SparseFlashAttention` is the largest single deviation in the capture — 4.70 ms
a verify above its structural floor, 73.43 µs a call, at 24 cube blocks with 48
vector subblocks. Four candidates, each measured rather than argued.

| candidate | measurement | verdict |
| --- | --- | --- |
| the rows it carries | 85.31 µs at one row, 89.17 at eight | excluded: 8× the rows costs 4.5% |
| the structure it runs in | the cheapest MIX call is 13.58 µs, already subtracted | excluded by construction |
| the bytes it gathers | 2048 slots of 1152 B read in **2.30 µs** | excluded: 2.6% of the call |
| its arithmetic | 142.61 MFLOP → 0.40 µs on cube, 12.96 on vector | excluded on cube; on vector it is 18% |

## The gather is not slow, and a permutation is the fastest pattern

2048 slots of 1152 bytes, the paged cache's slot size, over 16 blocks:

| pattern | a pass | rate |
| --- | --- | --- |
| contiguous | 2.54 µs | 863.7 GB/s |
| one page apart | 4.84 | 454.4 |
| a permutation with no locality | **2.30** | **956.7** |

A top-k selection's scattered access is the fastest of the three. So 1152
bytes is a large enough granularity that scatter costs nothing, and L2
absorbs it. The hypothesis this benchmark was written to test — that the 6.07×
is bandwidth at this access granularity — is refuted.

## What is left

142.61 MFLOP is 0.40 µs on the die's cube and 12.96 on its vector unit. The
operator is `MIX_AIC`, so it has both. If the scores and the value sum run on
cube, the arithmetic accounts for under 1% of 73.43 µs and nothing measured so
far explains the call. If they run on the vector unit, 12.96 µs accounts for
18%.

Either way the majority is unexplained, and the next step is reading what the
kernel puts on which unit rather than measuring another hypothesis from
outside. Four have now been excluded by measurement; a fifth guessed from
outside would be the fifth.
