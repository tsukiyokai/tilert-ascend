# Two-rank native FP32 HCCL comparison

This fixture compares C2 `c2b72e46…` with same-address native norm `40a9e3c0…`, attention `8deff750…` / `64108f6f…`, FFN `381febf9…` and four real FP32 HCCL2 reductions. It uses two controlled TP16-shaped shards of the actual layer6/layer7 constructors. Fourteen other constructor ranks remain CPU-only. It does not execute TP16 collectives or measure model TPOT.

[The A3-194 M12/24 result](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-timing-m12-v1-a3-194/summary.json) passes both retained arenas. Each event replays a 16-pair graph 30 times; each arena has 30 alternating AB/BA samples. Times are microseconds per indexed/shared two-layer pair, using the larger rank interval for each arm before paired subtraction.

| Arena | FP32 HCCL control p50 | C2 SHMEM candidate p50 | Paired saving p50 | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1148.304 | 1125.374 | 23.049 | 30/30 |
| 1 | 1153.384 | 1130.584 | 23.264 | 30/30 |

The separate one-pair graph series saves 32.449 and 31.283 µs, with 59/60 wins. All 18,532 production byte checks, 640 literal FP32 checks and 4,320 post-event checks pass. Capture leaves 1,808 observed boundaries unchanged; 16 ranks exit normally and only ranks 0/1 initialize an NPU. [Two independent audits](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-timing-m12-v1-a3-194/README.md) recompute every event maximum and paired difference from the retained rank intervals.

The control keeps both dtype conversions: BF16 partial → preallocated FP32 buffer → HCCL SUM → original BF16 destination. Literal `.float()/all_reduce/.bfloat16()` and ordered two-rank FP32 sums are checked outside capture and timing. Both dispatcher paths are recorded; preallocated copies are not claimed to use the same underlying CANN kernels as literal conversion. The process explicitly sets `HCCL_OP_EXPANSION_MODE=AIV` and `HCCL_DETERMINISTIC=true` before Torch import, matching the project's defaults in a fresh environment. The earlier BF16 HCCL correctness gates did not pin those values.

Each of two retained compute arenas has one repeated weight bank and both arms use the same production addresses. Caches, persistent top-k, original model weights and packed FFN weights are shared across arenas; their compute scratch, hidden/residual owners and four reduced BF16 outputs are disjoint. Control-only storage comprises four FP32 `[M,6144]` collective buffers. No host epoch reset occurs. Reported expected epoch increments are derived from graph-call counts; the device counter is not read. The SHMEM window remains on the actual official graph capture stream until every graph is destroyed.

The numerical gate checks 113 boundaries per rank over eight changing inputs. Original eager parents remain the oracle. Capture must leave all production bytes unchanged; one explicit replay must consume the current input. Repeated 16-pair graphs and consecutive replays must retain identical final-call values without resetting cache or epoch. Live inputs, complete caches, ordered top-k and all 264 active-table words remain checked. Timed graphs contain only production work, including the layer6-to-layer7 top-k copy; observation copies and CPU gathers are outside events.

The measured environment has no `HCCL_ALGO`, `HCCL_ALGO_CONFIG`, `HCCL_BUFFSIZE` or `ASCEND_LAUNCH_BLOCKING` override. Raw per-rank durations, nine post-event output checks and full post-run checks are retained. Repeated-pair checks observe the final call's retained values. The comparison combines launch fusion and SHMEM versus HCCL; it does not isolate fusion alone. `graph_tasks` records the length of `UpdatableGraph.tasks`, not a device task or launch count.

Build a new immutable staging directory with `stage.py`, supplying the accepted `/tmp/tilert-tp-graph-staging-v1`, the frozen norm DSO and the [source protocol](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/layer-tp-timing-protocol-v1/README.md). `cpu_check.py` verifies the actual Torch 2.12 ABI, all parent imports and source/ownership contracts without mapping an NPU. Device execution requires the two-die idle gate. `audit.py` checks the completed reports and reconstructs the paired statistics.
