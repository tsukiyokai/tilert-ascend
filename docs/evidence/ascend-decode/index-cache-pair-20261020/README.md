# Paired C8 index-cache writes

A3 a3-194/trt-asc, shared die15, CANN9.2.0 B114; INT8 index keys128B/slot,
FP16 scales2B/slot, capacity9344. Reuse the existing `ScatterPaKvCache` Norm
kernel to write both caches in one call. The FP16 scale input/cache is viewed
as INT8 with trailing dimension2; its original bytes and storage are preserved.

Norm requires matching key/value dtypes. The first mixed INT8/FP16 attempt
failed at host validation before loading the candidate ELF. An integer dtype
view satisfies this constraint without numerical conversion. FP8 C8 uses
FP32 scales and retains the original path. [Source and ABI receipt](receipt.json).

## Store-only screen

Server CPU checks cover nine M1/8/32×A/B/reset byte/address cases. The
[device run](device-result-02.json) covers18 eager/Graph cases, checking both
entire caches including unchanged rows. All are byte-exact. Candidate ELF
SHA4548b3a8758c11d7c5a2278d056936303a11aa41843026e5f239130b10b127ff
was opened and read by the runtime; [receipt](device-access-02-2.json).

| Width | Two scatters p50 µs | Paired p50 µs | Paired wins/30 |
| --- | ---: | ---: | ---: |
| 1 | 19.943 | 4.727 | 30 |
| 8 | 19.363 | 4.929 | 30 |
| 32 | 21.320 | 8.685 | 29 |

One graph replay of10 updates per sample;30 alternating AB/BA pairs, event
time divided by10, one warmup replay per arm. Input/cache reset and sync are
outside timing; views are created before capture. Profiler off, clocks
unlocked, foreign resident workload, no outliers removed. Peak NPU reservation
96MiB under a256MiB allocator quota. [Summary](screen-summary.json).

## Actual producer and method

The [second probe](method-result.json) loads the frozen original and patched
`AscendSFAIndexerBackend.write_cache` methods in isolated modules. Each update
runs real dynamic quantization, casts scale to FP16, then invokes the method.
Views are constructed inside the method during eager execution/capture.
K and scale alias one uint8 allocation at nonzero byte offsets; the comparison
checks the entire backing allocation, including untouched canaries.

All18 eager/Graph A/B/reset cases pass. Ten alternating pairs use five complete
updates/Graph, one warmup per arm, and the same event timing scope. Every
candidate pair wins; peak NPU reservation84MiB. [Summary](method-summary.json).

| Width | Original p50 µs | Patched p50 µs |
| --- | ---: | ---: |
| 1 | 25.794 | 8.586 |
| 8 | 28.966 | 10.342 |
| 32 | 28.460 | 16.716 |

Commit `48a23ac` adds13 lines only to the INT8 C8 branch. Its base is frozen unified
`indexer.py` SHA6513151ea1baa0a806abdbecbf776016ab16dd0cf6c75d9c5cd4aa02892b7dd1.
[Application check](release-patch-check.json) applies without fuzz and proves
AST identity with the tested candidate; one duplicate comment was removed.
BF16/C4/FP8 code is unchanged. Apply `patch -p1 --fuzz=0` from the isolated
vLLM-Ascend source root using `patches/vllm-ascend/paired-c8-index-cache.patch`.
The shared installation remains unchanged. Whole-model teacher, compiled
target graph and request TPOT are pending.

The old M8 target profile attributes0.364ms/round to32 C8 scatter calls. This
is a bounded launch-removal opportunity; it is not the main2ms gap. Source and
launch scripts are in [source](source/). Raw server root:
`/home/shanshan/trt-asc-194/experiments/index-cache-pair-20261020`.
