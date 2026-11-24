# A3 TP16 dependent AllReduce chains

Environment: `a3-gs:trt-asc`, CANN 9.2.0, Torch/Torch-NPU 2.12.0,
HCCL AIV, `HCCL_BUFFSIZE=200`, `HCCL_DETERMINISTIC=true` and task queue 1.
Both launches observe sixteen idle dies and no NPU processes. Clocks use
stock settings; frequency is neither locked nor sampled.
The [evidence](cann-tp-sum-timing-20261017.json) binds both scripts,
launches, thirty-two rank reports, raw samples and recomputed statistics.

Each case uses 10 warmups and 30 measured samples on a dedicated stream,
with device synchronization and a Gloo barrier before timing. NPU events
span either Python submissions or one graph replay; wall time includes
the final device synchronization. Every sample performs 156 dependent,
in-place SUMs on the same zero buffer. A separate rank-dependent input
must sum to 136; graph replay also passes that check on all ranks.

The table reports the p50 of the per-sample maximum across sixteen ranks,
in milliseconds for the complete 156-call chain.

| Payload | Python submission | Graph replay |
| --- | ---: | ---: |
| BF16, 12 KiB | 25.114 | 5.518 |
| BF16, 48 KiB | 22.910 | 5.033 |
| FP32, 24 KiB | 26.756 | 5.028 |
| FP32, 96 KiB | 25.159 | 5.074 |

The 24 KiB FP32 graph chain has wall p50 5.185 ms. Its event total divided
by 156 is a batch average, including replay submission, rank skew and
dependency scheduling. Isolated FP32 graph replays take 170–212 us in the
same event scope. Neither statistic measures bare HCCL latency.

The current M1 communication chain alone exceeds 2 ms. At M4 and four
delivered tokens per round, the 96 KiB chain costs 1.268 ms/token before
draft, casts, computation and other communication. At 3.2 delivered tokens
it costs 1.586 ms/token. The design therefore needs communication overlap
and fewer synchronization boundaries alongside weight reuse. Decode TPOT
and model precision are measured separately. Real BF16 partials retain
the [FP32 accumulation contract](cann-tp-sum-20261017.json).

## Matched AI_CPU and AIV

The newer [mode comparison](tp-mode-20261017/README.md) uses the same
scripts and protocol, changing only the configured engine within each
submission method. All 64 rank reports pass. The table is event rank-MAX
p50 in milliseconds for 156 dependent graph sums, without a profiler.

| Payload | AI_CPU | AIV |
| --- | --- | --- |
| BF16, 12 KiB | 10.951 | 5.455 |
| BF16, 48 KiB | 10.896 | 4.970 |
| FP32, 24 KiB | 10.925 | 4.976 |
| FP32, 96 KiB | 10.931 | 5.044 |

Separate post-timing traces identify actual AICPU and AIV kernels, not
just environment settings. AIV improves these communication-only chains.
The existing decode-session medians, 20.44 ms for AI_CPU and 21.47 ms for
AIV, cannot be used to deny that gain or identify its loss in the model.
Use the full decode trace to locate the additional time before changing
the communication design. The chain includes replay and scheduling;
it remains distinct from bare HCCL latency and decode TPOT.

Run the [Python entry](../../../tests/glm52/bench_tp_sum.py) and
[graph entry](../../../tests/glm52/bench_tp_sum_graph.py) sequentially with
fresh output directories:

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
for script in bench_tp_sum bench_tp_sum_graph; do
  HCCL_DETERMINISTIC=true HCCL_BUFFSIZE=200 HCCL_OP_EXPANSION_MODE=AIV \
    HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto \
    ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0 \
    OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python -m torch.distributed.run --standalone --nproc-per-node=16 \
    "tests/glm52/$script.py" --output "/workspace/experiments/glm52/$script"
done
```
