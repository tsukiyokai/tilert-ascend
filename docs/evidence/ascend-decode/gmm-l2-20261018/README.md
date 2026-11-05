# The expert GMM re-measured past L2

This supersedes [the first sweep](../gmm-shape-20261018/README.md), whose
numbers were taken with a single resident bank. One bank of 256 groups is
`256 × 6144 × 32 × 4 = 201,326,592` bytes, which is exactly the die's L2 of
192 MiB, so twenty calls per batch reread the cache rather than HBM. Two
checks condemn that sweep: its working set ran from 12.5% of L2 at
thirty-two groups to 100% at 256, and its `gate_up` marginal rate of
1261.2 GB/s over 64→128 groups exceeds the 1190.77 GB/s maximum the
[HBM probe](../a3-hbm-20261017.json) measured on one die. A rate above the
measured ceiling is not a rate.

[The sweep](../../../../tests/glm52/bench_gmm_shape_scaling.py) now rotates
five independent banks, a 1.0 GB footprint, so a call reads memory five
calls old. A3 `a3-84` die 0 shared with an idle tenant, one rank, twenty
calls per batch after twenty warmup calls, twenty batches, medians.

| groups | `gate_up` (µs) | marginal | GB/s | `down` (µs) | marginal | GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 65.867 | | | 62.392 | | |
| 4 | 65.233 | | | 67.191 | | |
| 16 | 70.175 | | | 76.046 | | |
| 32 | 73.673 | | | 77.125 | | |
| 64 | 104.325 | 0.958 µs/group | 821 | 98.990 | 0.683 | 576 |
| 128 | 151.524 | 0.737 | 1066 | 134.707 | 0.558 | 705 |
| 256 | 239.970 | 0.691 | 1138 | 216.089 | 0.636 | 618 |

`gate_up` converges to 1138 GB/s, which is 95.6% of the measured single-die
maximum and no longer above it. Subtracting that stream leaves a fixed cost
of 60–63 µs per call, against the 35.8–39.5 µs the contaminated sweep
implied: the old figure understated it by about 60%. `down` converges to
618–705 GB/s with about 53 µs fixed.

The old numbers were optimistic by 14–25% exactly where it mattered:
`gate_up` at 64 groups moves 83.10 → 104.325 µs, at 128 123.00 → 151.524,
at 256 210.38 → 239.970.

One contamination remains and bounds what this sweep licenses. The rotation
defeats L2 only when `groups × bytes-per-group × banks` exceeds 192 MiB. At
thirty-two groups that product is 125.8 MB, still inside L2, so the points
at thirty-two groups and below stay optimistic; the clean region starts at
sixty-four groups, where the product is 251.7 MB. The model's measured
expert union is 27.121 per layer
([route audit](../execution-gap-20261018/route-audit.json)), which sits in
the contaminated region.

## What survives of the earlier conclusion

That the kernel streams near the roof survives in corrected form for
`gate_up`: 1138 GB/s is 95.6% of the measured ceiling once traffic
dominates. That reducing MoE bytes buys little also survives, but now rests
on the fixed cost rather than on bytes being hidden: at the model's union,
streaming is about 19 µs of a call whose fixed cost is 60 µs, so traffic is
roughly a fifth of it. The profiled fused `GroupedMatmulSwigluQuantV2` is
104.095 µs per call, so about 25 µs of that call is the SwiGLU and quant
epilogue beyond this unfused measurement.
