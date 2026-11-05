# The verify's 39.677 ms, in terms that do not overlap

Each kernel's kind has a floor — the cheapest call of that kind in the whole
capture, since no call can be cheaper than the structure it carries. What a
kernel spends beyond that floor is its own execution. Separating the two
splits the verify without double counting.
[the floors](../mix-structure-20261019/README.md)

| | ms a verify | share |
| --- | --- | --- |
| the MIX structure: 595 kernels × 12.84 µs | 7.64 | 19.3% |
| 1788 single-engine kernels at their floors | 1.69 | 4.3% |
| what the operators spend above those floors | **21.77** | **54.9%** |
| the row-dependent part and the residual | 8.58 | 21.6% |

The third term attributes to 82% of what was previously unattributed.

## Above the floor, by operator

| ms a verify | above floor | p10 | calls | operator |
| --- | --- | --- | --- | --- |
| 4.759 | 74.36 µs | 88.56 | 64 | `SparseFlashAttention` |
| 4.450 | 72.06 | 86.26 | 62 | `GroupedMatmulSwigluQuantV2` |
| 3.538 | 57.30 | 71.50 | 62 | `GroupedMatmul` |
| 1.467 | 22.92 | 37.12 | 64 | `MlaPrologV3` |
| 1.052 | 17.04 | 17.76 | 62 | `ForeachAddList` |
| 0.789 | 11.56 | 13.90 | 68 | `MatMulV3` |

## The 10× was structure and count, not arithmetic

Against each operator's own live bytes, once the structural floor is removed:

| operator | above floor | its bytes | ratio |
| --- | --- | --- | --- |
| `SparseFlashAttention` | 74.36 µs | 12.26 | **6.07×** |
| `GroupedMatmulSwigluQuantV2` | 72.06 | 32.68 | 2.20× |
| `GroupedMatmul` | 57.30 | 16.34 | 3.51× |

So the grouped matmuls are 2.2 and 3.5 times the bandwidth they need, which is
close for a kernel that also dequantises int4 and runs SwiGLU. The round's
apparent 10× was the structure and the kernel count, not operator efficiency.

`SparseFlashAttention` is the exception at 6.07×, and it is also the operator
whose cost does not move with rows — 85.31 µs at one row against 89.17 at
eight. Three explanations are therefore excluded for it: rows, bytes, and the
structural floor. Nothing on this board has touched it.

## What this does to the route

"Reduce the kernel count from 1867 to 467" reaches 7.64 + 1.69 = 9.33 ms,
23.6% of the verify, because fewer kernels does not change what each operator
spends above its floor. The MIX term is 4.5× the single-engine term, so the
order within it is *stop being MIX* before *be fewer*.

The other half of the route is the 21.77 ms, and within it
`SparseFlashAttention` at 4.76 ms is the largest single deviation from any
floor the capture can define.
