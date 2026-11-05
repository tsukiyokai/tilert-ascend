# A non-contiguous rope cache cost 9% of TPOT

`a3-84` idle, TP16, source `04a6628`, the frozen 8K prefix, the composed flag
set with `HCCL_OP_EXPANSION_MODE=AIV`, `--max-tokens 1024 --capacity 9344
--warmup 3 --requests 3`, profiler off. One line of
`vllm_ascend/ops/rotary_embedding.py` differs between the two cases.

| case | TPOT median (ms) | three runs | rounds | A | round (ms) |
| --- | --- | --- | --- | --- | --- |
| before | 15.978 | 15.927 / 15.978 / 16.051 | 298 | 3.433 | 54.85 |
| after | 14.540 | 14.529 / 14.540 / 14.572 | 298 | 3.433 | 49.92 |

1.438 ms faster, 9.0%, and 4.93 ms per round. Both cases run 298 rounds at
A = 3.433, so the trajectory is unchanged: this is speed only, with no change
to the numbers the model produces. [report](rank0.json). New distance to the
2 ms target is 7.27×.

## How it was found

The compiled configuration could not be profiled until the runner's serial
M1 validation phase was skipped under `--compile-target`; the compiled model
is traced at the width of its first forward, so a width-1 pass fixed it at
one row and the M4 phase then fed four rows into a one-row buffer.

With that fixed, the profile of the compiled configuration
([before](op_statistic-before.csv)) showed `Slice` as the second largest
operator at 15.5% of device time: 190 calls, 112.05 µs mean, 381.02 µs max,
7.10 ms per round. `SuperKernel` had absorbed almost everything else —
`GroupedMatmulSwigluQuantV2` fell from 235 calls to 10, `MlaPrologV3` from
244 to 10 — but not this.

`kernel_details.csv` named the shape: input `[1048576, 128]` BF16, output
`[1048576, 64]`, so 256 MiB read and 128 MiB written per call. At 364 µs
that is 1053 GB/s, which is the measured streaming rate — the kernel was not
slow, it was moving 384 MiB. Over 19.3 calls per round that is 7.4 GB, more
than the 6.83 GB of model weights a round reads.

`operator_details.csv` attributed it to `aten::index`, and `Model ID`
`4294967295` placed it outside every graph, which is why `SuperKernel` could
not absorb it. Re-capturing with `--profile-stacks`, a flag added for this,
named the caller:

```
vllm_ascend/ops/rotary_embedding.py(104): get_cos_and_sin_mla
vllm_ascend/attention/sfa_v1.py(675): _build
tilert/models/glm_5_2_ascend/cann_model.py(339): _metadata
```

So it ran on every attention-metadata build, which this project does once per
decode step.

## The cause and the fix

`rotary_embedding.py` derives the two caches like this:

```python
cos_cache, sin_cache = cos_sin_cache.view(-1, 2, hidden_dim).repeat(1, 1, 2).chunk(2, dim=1)
_cos_cache = cos_cache.squeeze(1)
_sin_cache = sin_cache.squeeze(1)
```

`repeat` materialises one contiguous `[1048576, 2, 64]` buffer. `chunk` then
leaves each half a view with a row stride of 128 rather than 64. Line 104 is
`cos = _cos_cache[positions]`, an advanced index on that non-contiguous view,
and the lowering materialises the whole view before gathering the one to four
rows a decode step needs.

[The patch](../../../../tests/glm52/patches/rope-cos-sin-contiguous.patch)
adds `.contiguous()` to both. Memory does not grow — the same 256 MiB already
existed as the combined buffer — and the copy happens once during
initialisation, before any graph capture, so the constraint that the globals
keep stable addresses is preserved.

This looks worth reporting upstream: any vLLM-Ascend deployment that builds
attention metadata per step pays it.
