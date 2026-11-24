# W4A8 grouped matmul: fixed cost against streaming slope

**Superseded.** This sweep kept one resident bank, and one bank of
256 groups is exactly the die's 192 MiB L2, so its slope measures
cache reread. Its 64→128 marginal rate of 1261.2 GB/s also exceeds
the 1190.77 GB/s single-die maximum. Read
[the re-measurement](../gmm-l2-20261018/README.md) instead: with five
banks rotated the fixed cost is 60–63 µs, not the 35.8–39.5 µs below,
and `gate_up` converges to 1138 GB/s.

A3 `a3-84` die 0, shared with another tenant's idle `vllm serve`, one rank,
no model. [The sweep](../../../../tests/glm52/bench_gmm_shape_scaling.py)
keeps the full 256-group bank resident and uses `group_list` to select the
active groups, so the weight traffic scales while the kernel shape does not.
Values are arbitrary bits; timing does not depend on them. Twenty calls per
batch after twenty warmup calls, twenty batches, one shape per process. The
`down` case at thirty-two groups reports 74.17 µs against 74 µs for the
model's profiled `GroupedMatmul`, so the synthetic shape reproduces the
production cost.

## Active groups at one row each

| groups | `gate_up` (µs) | marginal | `down` (µs) | marginal |
| --- | --- | --- | --- | --- |
| 1 | 73.00 | | 60.89 | |
| 2 | 69.19 | | 70.46 | |
| 4 | 59.49 | | 60.79 | |
| 8 | 58.03 | | 59.63 | |
| 16 | 54.77 | | 70.24 | |
| 32 | 68.40 | 0.852 µs/group | 74.17 | 0.246 µs/group |
| 64 | 83.10 | 0.459 | 100.63 | 0.827 |
| 128 | 123.00 | 0.624 | 132.84 | 0.503 |
| 256 | 210.38 | 0.683 | 216.70 | 0.655 |

Up to sixteen groups the time does not move with the bytes: thirty-two times
the weight traffic costs nothing measurable, and the spread of 55–73 µs is
the noise of a shared die. The slope only emerges past thirty-two groups and
converges to 1152–1261 GB/s for `gate_up`, which is the 1.155 TB/s event
median of [the HBM probe](../a3-hbm-20261017.json). Subtracting that stream
leaves a fixed cost of 35.8–39.5 µs. `down` converges lower, 600–780 GB/s,
with about 49 µs fixed; the difference between the two aspect ratios
(K=6144/N=256 against K=128/N=6144) is not explained here.

Rows per group do not matter either: at thirty-two groups, one, two and four
rows cost 68.40, 63.05 and 65.84 µs.

## What this changes

The kernel streams weights at the HBM roof. At the model's shape of about
thirty-two active experts, `gate_up` needs 21.8 µs of streaming and `down`
10.9 µs, so 60–80% of each profiled call is fixed cost rather than traffic.
A bytes-divided-by-total-time reading gives 212 GB/s and invites the
conclusion that the kernel reaches a fifth of bandwidth; that figure is an
artifact of dividing by a mostly fixed time and should not be used.

Two consequences. Reducing MoE weight bytes buys little while the fixed cost
stands, because the bytes are already hidden under it. And the 13.3 ms that
one round spends in expert GMM is roughly 6.75 ms of per-call fixed cost over
150 calls, which alone equals the whole 6.752 ms round budget.

The fixed cost is not dispatch: the in-graph floor for a trivial kernel is
1.079 µs ([chain probe](../graph-chain-20261018/README.md)), sixty times
less. It is internal structure. At `loopCount == 1`, which is every M≤32
shape, `Process()` in `grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h`
runs `pre → SyncAll → mid → (prefetch for a non-existent iteration) →
SyncAll → SyncAll → post`, with only an AIV `pipe->Reset()` between the last
two device-wide barriers. The project's [single-chunk
patch](../../../../tests/glm52/patches/gmm1-single-chunk.patch) already
removed a fourth barrier for 3.679 µs. Three remain, one of them redundant
in this case, and `preProcess` still stages the activation's two int4 halves
through GM for `midProcess` to read back — a round trip whose only purpose is
to feed the cube engine an int4 operand, which a batch of one to four rows
does not need.
