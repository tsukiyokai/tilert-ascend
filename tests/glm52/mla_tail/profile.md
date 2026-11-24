# MLA tail stage diagnostic

The diagnostic executes the five original primitives in `mla_tail_native.h`:
head staging, value matmul, dynamic quantization, output matmul and output
dequantization. It uses a separate library and does not change the ordinary
entry point. The harness compares their BF16 outputs on four changing inputs
before collecting counters or timings.
Its default ordinary library is the preserved `build-composable` artifact.
The promoted `Forward` uses the later eight-row epilogue, while this profile
keeps `DequantizeOutput` for attribution of the earlier measurement.

Each AIC and AIV records its own stage start, completed work and grid-barrier
release through `AscendC::GetSystemCycle()`. The SDK implementation reads
`SYS_CNT`. Reports retain raw ticks and calculate only differences within one
core; they assume neither a counter frequency nor aligned clocks across cores.
The per-core work fraction uses that same core's complete recorded window.

The profile adds `PIPE_ALL` before every work-completed timestamp. It also adds
a fifth grid barrier after output dequantization, so idle AIC trace stores
cannot overlap that stage. The first four grid barriers retain the ordinary
stage order. Each core then publishes one exclusive 512-byte scalar-cache
record. These changes perturb execution. Separate graph-event pairs report
ordinary and instrumented whole-kernel times; stage ticks do not replace an
uninstrumented latency measurement.

```bash
CANN_ROOT=/opt/ascend/cann-9.2.0 bash tests/glm52/mla_tail/build_profile.sh
python tests/glm52/mla_tail/profile_stages.py --widths 1 12 32 --blocks 8 \
  --samples 30 --output /tmp/mla-tail-profile.json
```

The harness loads the ordinary Torch library for its validated allocation,
tiling and physical-core preparation. The independent C ABI launch consumes
those operands and a separate scratch/output allocation. Capture obtains the
current stream through torch_npu's `npu_stream` property, whose native getter
flushes the host task queue through `stream()`. It inserts no device
synchronization between a producer and the direct launch. Scratch, tiling,
weights, outputs and trace remain alive for the graphs' lifetime.

At fixed eight groups, M=1 and M=12 both use capacity 16. A growing vector
output-dequantization duration supports the small-tile epilogue hypothesis;
a growing value/output matmul duration points to cube or movement work. The
separate `bench_weight_banks.py` reuse/distinct-address experiment tests the
weight-working-set contribution. None of these measurements establishes full
attention or decoder TPOT.
