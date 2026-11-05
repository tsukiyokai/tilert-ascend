# Attention shape scaling and the per-kernel floor

A3 `a3-84` die 0, shared with another tenant's idle `vllm serve`, one rank,
no model. [The sweep](../../../../tests/glm52/bench_sfa_shape_scaling.py)
builds a synthetic 8K paged MLA cache and times one shape per process, so an
unsupported shape faults alone. MLA keeps one KV head, so raising the query
head count leaves `kv_bytes` unchanged and isolates occupancy, while raising
top-k scales the read. Twenty calls per batch with a single synchronize, after
twenty warmup calls, twenty batches, medians below. At heads=4 and top-k=2048
this harness reports 88.46 µs against 89.05 µs for the real layer-0 hot map,
so the synthetic shape reproduces the production cost within 2%.

## Query heads at top-k 2048, `kv_bytes` fixed at 2,359,296

| heads | median (µs) |
| --- | --- |
| 4 | 88.303 |
| 8 | 89.151 |
| 16 | 89.745 |
| 32 | 89.149 |
| 64 | 95.312 |

Sixteen times the query work costs 8% more time. The fused kernel is not
compute bound and the model's four heads per rank leave the die's query
capacity unused; TP16 does not starve this kernel.

## Top-k at heads 4

| top-k | SFA (µs) | gather only (µs) | gather+FIA (µs) |
| --- | --- | --- | --- |
| 256 | 36.334 | 23.448 | 137.073 |
| 512 | 43.482 | 23.776 | 136.728 |
| 1024 | 58.751 | 23.375 | 134.187 |
| 2048 | 87.986 | 25.171 | 136.819 |
| 4096 | 144.031 | 24.038 | 135.938 |

SFA fits `29.4 µs + 27.9 ns per selected row` over this range: the marginal
cost per 1,152-byte row is 27.9, 29.8, 28.5 and 27.4 ns across the four
steps, so the intercept is shape independent and the slope is 41.3 GB/s,
3.6% of the 1.155 TB/s event-median read measured in
[the HBM probe](../a3-hbm-20261017.json).

The same rows read by `index_select` alone are flat: sixteen times the bytes
cost 2.5% more time, and the 4096-row case reaches 196 GB/s without
saturating. The scattered read is therefore not the limit; SFA's internal
gather loop is. At 27.9 ns per row and roughly 600 ns of HBM latency the
kernel keeps about twenty row fetches in flight.

`gather+FIA` in this harness is flat near 135 µs and slower than SFA, which
contradicts the 69.15 µs measured on the real layer-0 map. The synthetic
cache is base-format ND while the production latent cache passes through
`npu_format_cast`; this path is unresolved and no conclusion is drawn from
it. The head sweep for it is equally flat (8/16/32 heads: 133.147, 134.247,
142.627 µs).

## Per-kernel floor

[The floor probe](../../../../tests/glm52/bench_kernel_floor.py) times
trivial device work back to back, fifty calls per batch, twenty batches.

| operation | median (µs) |
| --- | --- |
| `add`, 1 element | 10.160 |
| `add`, 1 row of 6144 BF16 | 10.094 |
| `add`, 4 rows | 10.246 |
| `npu_rms_norm`, 1 row | 24.858 |
| `index_select`, 1 row | 12.254 |
| `empty_like`, 1 row | 4.902 |

Elementwise cost is independent of size from one element to 49 KiB, so
~10 µs is a floor rather than work. The allocation-only case isolates
~4.9 µs of that as runtime dispatch. These are eager figures and a graph
removes the host share; the profiled in-graph SFA of 90.3 µs/call agrees
with the 88 µs here, so SFA's intercept is device side: about 19 µs per
call, 1.5 ms per round over 78 layers.
