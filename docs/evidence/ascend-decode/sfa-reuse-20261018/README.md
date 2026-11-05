# Existing CANN sparse-attention alternatives

A3 `a3-gs:trt-asc`, CANN 9.2.0, Torch/torch-npu 2.12.0, BF16
latent512 + RoPE64, four heads/rank, 2048 selected keys. The
[audit](audit.json) binds source and raw outputs. Runs use shared die 0.

## Numerical replay

The real input is layer 0/rank 0 at position 8192, M=1. An explicitly
synthetic M=4 case changes queries, KV rows and each query's causal indices.
All three pipelines use the same values and reversed physical-page table.
Each pipeline's two executions have identical output bits.

| Pipeline versus original SFA | Real M1 NRMSE | Synthetic M4 NRMSE |
| --- | --- | --- |
| Device paged gather → existing nonpaged FIA | 0.000521 | 0.000876 |
| Four 512-slot SFA partitions → FP32 AttentionUpdate | 0.002370 | 0.002153 |

Maximum absolute BF16 difference is 0.0078125 in all four comparisons.
Peak Torch reserved memory is 96 MiB for M1 and 182 MiB for M4. These
cross-kernel diagnostics do not release model precision or establish speed.
The [single replay entry](../../../../tests/glm52/compare_cann_sparse_attention.py)
performs eight explicit attention/merge calls, saves six raw outputs and
checks repeatability. It has no performance loop or model loading.

The device-gather/FIA prototype also passes changing-input ACL replay for
M1 and synthetic M4: one capture, replay A/B/reset A. All six persistent
inputs change for B; context, gathered latent/rope and physical indices
match eager bits. Reset A overwrites poisoned output/intermediate buffers.
The independent raw-output check covers 17,756,160 replay elements.

```bash
python tests/glm52/compare_cann_sparse_attention.py \
  --input /workspace/offline/cann-mla8k-20261017/early-bind/result.pt \
  --output /workspace/offline/attention-replay --device 0
```

## Execution choices

The pinned arch22 SFA assigns M4 to four active AIC workers, each iterating
over four 512-slot KV segments. This is derived from source; the profile's
24 launch blocks are not an active-core count. Its 78 calls sum to about
7 ms in the existing profiled target round, versus the 6.752 ms total
round budget at observed acceptance 1023/303.

Head-as-batch passes real M1 bit comparison but repeats selected KV
traffic fourfold; both one and four heads still pad to sixteen matrix
rows. The two alternatives above expose KV parallelism instead. FIA uses
one virtual batch per query, preserving each query's selected-key order.
The split variant retains SFA gather and merges four partials with
`L_i=max_i+log(sum_i)`, `O=Σ O_i exp(L_i−logsumexp(L))`. Partials are
BF16; conversion to FP32 for merging cannot undo that extra rounding.

Existing SFA already stages selected KV through GM. External gather is
not automatically an additional whole-KV round trip: both payload
traversal models total 9,175,040 bytes/query before padding and other
workspace. Its added launch boundary and actual traffic need measurement.
The source's incomplete `FLASH_DECODE` branch is not enabled.

## Complete graph screening

The [benchmark](../../../../tests/glm52/bench_cann_sparse_attention.py)
captures ten complete calls per graph, warms ten graphs and rotates
pipeline order over thirty samples. Same-stream event time is divided by
ten; mapping, gather, input transforms and merging are included.

| Pipeline | Real M1 p50 (µs) | Synthetic M4 p50 (µs) |
| --- | --- | --- |
| Original SFA | 86.67 | 88.44 |
| Paged gather → FIA | 91.11 | 136.26 |
| Split SFA → AttentionUpdate | 59.83 | 95.59 |

All 120 saved graph outputs match their eager outputs bit for bit before
and after timing. Sixteen external jobs share the host; p95 spans
437–506 µs. These hot-input measurements screen candidates, not idle-device
latency or decode TPOT. These generic M4 candidates leave SFA as baseline.

A short eager trace contains all six pipeline labels and 44 kernels;
all six contexts match the nonprofiled replay. In its second iteration,
FIA takes 36.24 µs, while its mapping and gather kernels take 96.46 µs.
The full pipeline takes 132.70 µs versus SFA's 88.38 µs. Split SFA takes
53.28 µs plus 48.58 µs of transforms and merging. The first original-SFA
call is a 2734.84 µs outlier. This shared-device Level0 trace supplies no
active-core or HBM counters.

## Contiguous cache and real M4

TileRT's cache uses identity block tables. Direct INT32 token indexing
removes page division, lookup, remainder and dtype conversion, then feeds
the same existing FIA. Two gathers remain inside the captured pipeline.

The [real M4 capture](real-m4-input-audit.json) runs layer 0 on all sixteen
ranks with the frozen 8K prefix and teacher tokens `[6500,382,262,576]`.
Checkpoint embeddings, original prefix fields and causal 2048-slot indices
are independently checked. Only one layer is resident, below 1 GiB/die.

The [runtime replay audit](contiguous-graph-audit.json) checks the actual
[per-instance implementation](../../../../tilert/models/glm_5_2_ascend/cann_attention.py)
and [benchmark](../../../../tests/glm52/bench_cann_contiguous_attention.py).
Changing-input A/B/reset graphs and post-timing outputs agree with each
kernel's eager execution across 819,200 BF16 elements. SFA reproduces the
original real M1/M4 capture contexts bit for bit.

| Complete hot graph | Real M1 p50 (µs) | Real M4 p50 (µs) |
| --- | --- | --- |
| SFA | 86.36 | 89.05 |
| Direct INT32 gathers → FIA | 46.52 | 69.15 |

Thirty interleaved samples per pipeline include both gathers and graph
scheduling. The real-M4 median difference is 19.89 µs, or 22.3%; applying
that difference to all 78 layers would save 0.460 ms/token at A=1023/303.
That is a scenario, not a model speedup. Shared-device p95 remains
404–464 µs. FIA differs from SFA in 115/8192 real-M4 elements, maximum
absolute difference 0.0078125 and NRMSE 0.000628; these are diagnostics.

Both MTP runners expose `--attention-kernel gather_fia`, selecting the
candidate for the 78 target layers and layer-78 MTP. Each query must have
2048 valid causal indices in the contiguous BF16 cache. Default SFA and
the frozen-teacher release gate remain in effect. Full-model precision
and matched 1024-output timing are the next acceptance steps.

[Existing GatherPaKvCache](contiguous-candidates-audit.json) also copies
the same rows exactly when given explicit INT32 zero `seq_offset`. In
this installed CANN build, omitting it creates an uninitialized optional
tensor; the device reads offset 1066385309 and faults. Passing zero fixes
that defect. The complete synthetic-M4 graph still takes 858.42 µs versus
64.34 µs for direct gathers; this implementation is not selected.

## Raw artifacts

Under `/home/shanshan/trt-asc/offline/` on a3-gs:

- `cann-attention-reuse-20261018-01/`: repository runner, real M1 and
  synthetic M4 inputs, graph samples, short trace, raw outputs and
  independent CPU audits.
- `cann-sfa-m4-input-20261018-01/`: all-rank layer-0 capture, original
  three-field KV and rank-0 real-M4 attention fixture.
- `cann-gather-fia-20261018-01/`: frozen repository runtime, real M1/M4
  dynamic graph replays, interleaved samples and independent CPU audit.
- `cann-sfa-head-batch-20261018/`: pinned SFA source/profile contract and
  head-as-batch numerical diagnostic.
- `cann-fia-gather-20261018-01/`: FIA schema/source contract, independently
  checked CPU/device gathers and page-permutation cases.
- `cann-sfa-splitkv-20261018-01/`: partials/max/sum and FP64 merge
  reference; device-only arithmetic matches the earlier CPU bridge.
