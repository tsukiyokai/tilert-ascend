# Two-rank native FP32 HCCL timing

M12 with 24 mixed groups passes on A3-194 dies 6/7. C2 `c2b72e46…` saves 23.049 and 23.264 µs per indexed/shared two-layer pair across two retained compute arenas. All 60 primary paired samples favor C2. The [summary](summary.json), [independent reconstruction](independent-audit.json) and [root reconstruction](root-independent-summary.json) agree.

| Arena | FP32 HCCL control p50 (µs) | C2 SHMEM candidate p50 (µs) | Paired saving p50 (µs) | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1148.304 | 1125.374 | 23.049 | 30/30 |
| 1 | 1153.384 | 1130.584 | 23.264 | 30/30 |

Each primary graph contains 16 repeated two-layer pairs; an event contains 30 replays. Thirty AB/BA samples per arena use the larger local rank interval for each arm, divided by 480, before paired subtraction. The separate one-pair graph series gives control/candidate p50 values of 1241.589/1209.764 and 1236.902/1206.920 µs. Its paired savings are 32.449/31.283 µs, with 59/60 wins. These series remain separate because changing the calls per replay changes the measurement.

The control uses native norm `40a9e3c0…`, indexed/shared attention `8deff750…`/`64108f6f…`, FFN `381febf9…` and four real HCCL2 FP32 sums. Both conversions remain in the graph: BF16 partial to one of four retained FP32 buffers, then reduced FP32 to the original BF16 owner. Literal `.float()/all_reduce/.bfloat16()` and rank-ascending FP32 sums pass outside events. The measured environment is `HCCL_OP_EXPANSION_MODE=AIV`, `HCCL_DETERMINISTIC=true`; `HCCL_ALGO`, `HCCL_ALGO_CONFIG`, `HCCL_BUFFSIZE` and `ASCEND_LAUNCH_BLOCKING` are unset. Torch deterministic algorithms remain disabled; that flag is distinct from the HCCL setting.

All 18,532 production byte checks, 640 literal FP32 checks and 4,320 post-event checks pass. Sixteen captures leave 1,808 observed boundaries unchanged. Every replay uses the actual capture/window stream and executes no Python runnable. Rank-local attention and FFN partial hashes differ, while reduced outputs and retained residuals agree. Four NZ banks per rank keep their bytes and allocation. Compute arenas are disjoint; caches, persistent top-k and model/FFN weights are shared. All 16 ranks exit normally, with NPU initialization limited to ranks 0/1. These assertions are checked in [audit.py](audit.py), the unchanged [root auditor](audit-tp-timing-root-v1.py), and the [supplemental contract record](supplemental-contracts.json).

The scope is two controlled TP16-shaped shards from the actual layer6/layer7 constructors, one repeated weight bank per arena, and combined launch-fusion/collective effects. It does not measure full-model TPOT or TP16 communication. Repeated-call checks retain final-call values, not every intermediate call. Expected epoch increments come from call counts; the device counter is not read. `graph_tasks=0` records the length of `UpdatableGraph.tasks`, not the number of device tasks or launches.

[raw-artifacts.json](raw-artifacts.json) pins all 68 original text reports/logs and both audit sources. The [frozen harness](../vendor-hook-tp-timing-harness-v1/README.md) retains staging manifest `5f187350…`, all changed sources, CPU gates and the exact launcher. No device binary was added to this archive.
