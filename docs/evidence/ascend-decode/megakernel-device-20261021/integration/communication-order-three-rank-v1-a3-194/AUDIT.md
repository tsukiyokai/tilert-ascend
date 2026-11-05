# Three-rank reduction-order counterexample

Ascending-rank FP32 accumulation is not byte-equivalent to the installed HCCL FP32 SUM in this fixture. All 36 rank/shape/input records preserve the native serial result and its immediate consumer, while HCCL matches a different grouping. [Summary](summary.json), [all case checks](case-audit.json), [all selected algorithms and INFO launches](hccl-selection-and-launch.log).

The run uses CANN 9.2, Torch/TorchNPU 2.12, three logical ranks mapped to physical dies 4/6/7, 24 MIX groups, and M1/M12 with H6144. Each shape has two eager calls and four changing-input graph replays per rank. `HCCL_OP_EXPANSION_MODE=AIV` and `HCCL_DETERMINISTIC=true`; neither `HCCL_ALGO` nor `HCCL_ALGO_CONFIG` is set. The diagnostic compares the first reduction of the existing probe. Its dependent CUBE projection and second reduction execute, but their output is not checked here. This is not a TP16, whole-layer, or timing result.

With epsilon = 2^-25, the following three columns repeat across each row and are multiplied by 2^(iteration-2). Every input is exactly representable as BF16.

| Rank | Column 0 | Column 1 | Column 2 |
| --- | ---: | ---: | ---: |
| 0 | 1 | 1 | 1 |
| 1 | epsilon | -1 | 2 |
| 2 | -1 | epsilon | 3 |

The native order `BF16(F32(F32(x0+x1)+x2))` produces `[0, epsilon, 6]` before the common power-of-two scaling. The alternative `BF16(F32(F32(x0+x2)+x1))` produces `[epsilon, 0, 6]`, exactly matching HCCL for these inputs. All 36 cases differ in 4096 elements per row; the maximum absolute difference ranges from 2^-27 to 2^-22. The exact integer column remains equal. These observations identify incompatible accumulation results; they do not uniquely reconstruct HCCL's complete reduction tree.

## Input, kernel and graph boundaries

- [communication_order_devicecheck.py](source/communication_order_devicecheck.py):20–35 constructs the BF16 inputs and both reference groupings. Lines 85–103 pass the same persistent `partial` allocation to the original probe and retain independent `first`, `output` and consumer allocations. Lines 117–134 update and synchronize that input before execution, snapshot the native result, then call `dist.all_reduce(partial.float(), group=hccl)` through a separate FP32 tensor and round it to BF16. The HCCL input is not the native result or a host reduction.
- [communication_device.h](source/communication_device.h):96–136 reads the received payload in ascending sender order, uses a separate FP32 Add for each later sender, and casts once to BF16. The input pointer is only read by the sender path. The original [probe kernel](source/communication_probe.cpp):94–112 writes the first result to a distinct output, then uses it as the CUBE input. The C ABI in [communication_probe_host.cpp](source/communication_probe_host.cpp):187–203 preserves this argument mapping.
- The probe DSO is `1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f`, verified by the launcher and driver before loading. Its retained build identity pins header `cf9e9a843c82ff77a4a4c2d6e4ff65024107cbd3c45044280c6502d65be8bde7`. The three communication sources are byte-identical to the C2 whole-layer build, whose [kernel](c2-source/source/kernel.cpp):43–56 calls this same `AllReduceBf16` helper. [Source identities](source-audit.json).
- Before every measured check, the source is updated and first/output/consumer storage is poisoned with -29/31/-37. Both eager cases already show the mismatch. In graph cases the native launch and immediate `consumer.copy_(first)` are captured together; the integer control column changes on every replay. The HCCL call executes afterward, outside the graph. Full synchronization separates input writes, native snapshots and HCCL results. The graph and all tensor owners remain alive until `graph.reset()` and synchronization precede native-window destruction.
- Each rank's actual runtime mapping is `[4,6,7]`, current logical device equals rank, and both peer-access checks succeed. Every rank records the same six mapped HCCL libraries and three SHMEM libraries. SHMEM `libshmem.so` matches the retained probe build's `cf9b853b…` identity. All nine full hashes are retained in [summary.json](summary.json).

## Algorithm and launch evidence

There are six complete `SelectAlgfor91093` records, two per rank, all selecting `AllReduceMeshAivFor91093Executor`. The 36 INFO `ExecuteKernelLaunchInner` records specify rankSize 3, float32, sum, deterministic 1, and equal send/receive pointers. Per rank, tags 1–6 have count 6144 and tags 7–12 have count 73728, matching the driver loop exactly. No HCCL warmup or graph-captured HCCL call is present in that loop. The log audit pairs each launch with the preceding selection in its own process and records the original path and line number. [Parsed launches](launch-audit.json).

The source and results support the intended counterexample without an auxiliary sum or graph-only failure. The JSON reports retain full-tensor byte-comparison metrics and the first three values, rather than full device tensor dumps. `audit.py` checks all those fields against an independent Python IEEE FP32/BF16 calculation, including byte-mismatch counts and aggregate errors. Its reconstructed input hashes describe the exact CPU fixture; they are not additional device observations. The driver's top-level `pass` alone checks native serial/consumer correctness, so this audit also requires every HCCL alternative-grouping match and every native/HCCL mismatch explicitly.

## Retained raw evidence

All 20 original files, totaling 81,185,055 bytes, are preserved under `raw/`; the 17 log files use deterministic gzip and the three rank reports remain JSON. Each compressed log is decompressed and checked against its complete original bytes. [raw-manifest.json](raw-manifest.json) records original and archived byte sizes and SHA256 values. The archive occupies 4,981,450 bytes before audit metadata. The original files under `/tmp/tilert-megakernel-validation-20261021/communication-order-three-rank-v1-raw` remain unchanged. The launcher, exact container argv and idle-admission records are retained under `launch/`.

Run `python3 audit.py` from this directory to recheck the retained original ROOT data and regenerate the archive. The audit imports no Torch/NPU module, does not launch the frozen driver, and performs no device operation.
