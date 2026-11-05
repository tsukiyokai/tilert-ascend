# TP16 MLA projection tail

One AscendC mixed kernel computes the continuous chain after sparse attention:
`latent [M,4,512] → W_UV → BF16 [M,1024] → dynamic INT8 → o_proj → BF16 [M,6144]`.
The caller retains the existing TP reduction, residual, normalization and MoE.
This is the projection tail, not a complete attention or decoder-layer kernel.

The source targets A3 `dav-2201` and CANN 9.2. It uses the repository's existing
direct-launch Matmul and `SyncAll<false>()` pattern. No `aclnn` call or additional
device launch occurs inside the tail. Four grid barriers separate the vector
and cube phases. `__schedmode__(1)` requests batch scheduling; preparation limits
the launch to `min(physical AIC, physical AIV / 2)`. The default is at most eight
mixed core groups. A caller using a smaller stream resource allocation must
request a block count within that allocation. Independent streams need separate
scratch/output tensors.

The output epilogue batches up to eight rows of one 256-column tile per DMA.
It retains the original per-row arithmetic and BF16 boundary, loads the channel
scale once per batch, and waits on `MTE3_MTE2` before reusing the output arena.
Its 22,784-byte UB allocation coexists with the earlier 19,488-byte vector
workspace; the GM layout and four grid barriers are unchanged.

The [original-BF16-scale comparison](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/mla-tail-bf16-scales.json)
passes M=1,2,12,32 with eight changing graph replays and 24 core groups.
The vendor receives BF16 channel scales with varied mantissas and exponents;
the native entry receives their exact FP32 conversion. All final outputs
match raw bytes with the existing channel-then-token multiplication order.

Build offline on a machine with CANN, torch and torch_npu:

```bash
CANN_ROOT=/opt/ascend/cann-9.2.0 PYTHON_BIN=/path/to/python \
  bash tests/glm52/mla_tail/build_so.sh
```

The script finds torch headers, libraries and C++ ABI through that Python.
`CANN_ROOT`/`PYTHON_BIN` take precedence over the compatible
`ASCEND_HOME_PATH`/`PYTHON` settings.
`BISHENG`, `CXX` and `SANITIZE=1` are optional build settings. A missing compiler
is an error; CPU checks do not compile or validate the device kernel.

Load `mla_tail_torch.so` before graph capture. Prepare one workspace per active
width/stream and keep it alive with the captured graph:

```python
scratch, tiling, blocks, sync_base = torch.ops.tilert.mla_tail_prepare(
    reference, capacity, requested_blocks,
)
torch.ops.tilert.mla_tail_out(
    latent, value_tiles, output_tiles, output_scale, bias,
    scratch, tiling, output, capacity, blocks, sync_base,
)
```

`capacity` is 16 or 32 and must cover the actual positive row count. `tiling` is
UINT8 `[capacity,1024]`; its leading extent encodes capacity for host validation.
Only its first 1024 bytes contain the two cube tilings. `scratch` and `output`
are mutated. The runtime call performs no allocation, device-to-host copy or
synchronization. The registered Meta implementation preserves the output alias.

Weights are prepared once in ordinary ND storage. Convert an NZ tensor with
`npu_format_cast(..., 2)` before reshaping; logical `is_contiguous()` alone does
not establish physical ND storage. The binding checks the actual NPU format
of each argument; the model adapter performs the required casts before packing.

| Argument | Dtype and shape | Layout |
| --- | --- | --- |
| latent | BF16 `[M,4,512]` | SFA output, row/head/latent |
| value_tiles | BF16 `[16,512,64]` | `W_UV[4,512,256].reshape(4,512,4,64).permute(0,2,1,3).contiguous().reshape(16,512,64)` |
| output_tiles | INT8 `[24,1024,256]` | logical `o_proj.weight[1024,6144].reshape(1024,24,256).permute(1,0,2).contiguous()` |
| output_scale | FP32 `[6144]` | loaded per-channel scale converted to FP32 |
| bias | absent, INT32 `[6144]`, or FP32 `[6144]` | INT32 before dequantization; FP32 after both scales |
| output | BF16 `[M,6144]` | rank-local output, before TP reduction |

The checkpoint's attention projections use `W8A8_DYNAMIC` and its `o_proj` has
no bias. Dynamic quantization is per row of this rank's 1024 input columns:
`scale=max(abs(BF16(value_up)))/127`, with FP32 arithmetic, nearest-even rounding
and codes in `[-127,127]`. A zero row has scale zero and zero codes. The output
order is FP32 integer accumulation times channel scale, then token scale, then
optional floating bias, then BF16 rounding. Static W8A8 quantization parameters
and routed-expert W4 `scale_bias` are different contracts.
The kernel obtains `127/maxabs` with vector `Div`, matching the CANN execution
unit. The CPU formula checks below use IEEE FP32 division and cannot establish
that device instruction's threshold behavior.
The INT32-to-FP16 cast also consumes the AIV `DEQSCALE` register. The kernel
sets it to one before each cast, as CANN dynamic quantization does; otherwise
earlier kernels can change the resulting INT8 codes while scales remain
correct. On A3 `Ascend910_9392`, CANN 9.2.0, the corrected library
`6fe3a50ec4ccd72d43a96d814c359463a46584103e62d96faa657ffcc51df821`
matched the vendor chain numerically at zero tolerance at
M=1,2,3,4,6,8,12,16,24,31,32, with
eight changing-input graph replays per width (88 total), eight mixed groups,
and synthetic seed 829. Reports are `logs/mla-m1-m2-v2.json` and
`logs/mla-widths-v2.json` in the isolated experiment workspace. These checks
exclude real checkpoint weights, other block counts, SFA and TP reduction.

Host checks:

```bash
python -m unittest discover -s tests/glm52/mla_tail -v
```

They check exact TP16 tiling, complete padded allocations, idle-core ownership,
mixed-core residency, BF16 quantization thresholds, bias units, weight packing
and a full-shape CPU comparison. They also compile the actual shared layout
header with a host C++ compiler. Remaining validation includes sanitizer runs,
real checkpoint inputs, other core counts, and an end-to-end paired TPOT
measurement. CPU checks do not establish device correctness or latency.

When a target-device run is authorized, the separate harness compares the
native tail with `npu_transpose_batchmatmul → npu_dynamic_quant →
npu_quant_matmul` on identical inputs. It tests changing inputs under replay
and reports graph event timings with alternating measurement order:

```bash
python tests/glm52/mla_tail/check.py --device npu:0 \
  --widths 1 2 4 6 8 12 16 24 32 --output /tmp/mla-tail-check.json
```

Equality is exact by default. Explicit `--atol`/`--rtol` values appear in the
report. A failed comparison stops timing and reports the value-up BF16 values,
INT8 codes and activation scales separately. `--fixture` accepts a CPU `.pt`
dictionary with `latent`, `value_weight`, `output_weight`, `output_scale` and
optional `bias` in the logical layouts above. The report's timing excludes
SFA, TP reduction and the rest of the model. Reports record the library and
source SHA-256 hashes, fixture identity, torch/CANN/runtime/driver versions,
device properties and timestamp.
`--queued-producer --no-timing` adds a Torch clone inside capture immediately
before the native consumer. It checks producer submission order without a
device synchronization between the two. The direct-launch binding uses
`NPUStream::stream()` to flush Torch's host task queue before launch.

The independent `build-dequant/mla_tail_dequant.so` candidate
(`4a9304fe9342733d3078943592f2671083400d086a1b6eecb54baca2794beabb`)
passed changing-input captured checks against both the earlier native library
and the stock vendor chain, including INT32 and FP32 bias cases. At eight
mixed groups, six paired graph-event measurements with a clone in all three
arms gave these medians in microseconds:

| Rows | Earlier native | Batched epilogue | Vendor |
| --- | ---: | ---: | ---: |
| 1 | 31.573 | 31.389 | 36.184 |
| 12 | 40.689 | 34.249 | 37.791 |
| 32 | 58.153 | 39.848 | 39.308 |

The raw report is `a3-84/mla-dequant-timing.json` in the isolated validation
directory. It records four changing-input replays per width, exact numerical
and raw-byte equality, and 30 iterations per timing sample. These are tail
measurements on synthetic seed-829 operands. The earlier library and the
candidate remain separate from the promoted `Forward` builds. See
[the stage diagnostic](profile.md) for the counter protocol that identified
the epilogue cost.
