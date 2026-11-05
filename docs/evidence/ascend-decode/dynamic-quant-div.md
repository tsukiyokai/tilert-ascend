# A3 DynamicQuant division

On `a3-gs:trt-asc`, Ascend910_9392 and CANN 9.2.0
master20260917, native FP32 `Div(127,amax)` explains the first 8K
indexer difference. For the frozen BF16 row, `amax=1.078125`:

| Boundary | Native A3 | CPU IEEE FP32 |
| --- | --- | --- |
| Factor | 117.7970962524414 (`0x42eb981d`) | 117.79710388183594 (`0x42eb981e`) |
| Product at `x=amax/2` | 63.499996185302734 | 63.5 |
| RINT | 63 | 64 |

The [64-lane AscendC probe](../../../tests/glm52/quant_div_probe/quant_div_probe.cpp)
separates Div, Mul and FP32-to-INT32 RINT with vector barriers. Literal
`±63.5` controls round to `±64`; all 64 actual quantization samples match
the captured DynamicQuant codes. This isolates the factor difference.
The installed `dynamic_quant_single_row.h:147–180` returns scale via
`Muls(amax,1/127)`, then uses Div, FP32 Mul and RINT for the codes.

## Calibration and CPU reference

The [calibration](../../../tests/glm52/quant_div_probe/calibrate.py) enumerates
all 128 BF16 mantissas at every exponent from −120 through 127: 31,744
positive maxima, raw BF16 bits `0x0380..0x7f7f`. Every quotient is finite
and normal. The signed native-factor-bit difference from IEEE FP32 is
independent of exponent: −1 for these 15 mantissas, zero for the other 113:

```text
3, 7, 10, 12, 19, 22, 23, 28, 29, 35, 43, 48, 55, 73, 113
```

All 31,744 factors, FP32 products and RINT results pass an independent
integer/rational oracle. The observed division is neither round-toward-zero
nor round-to-odd; the raw audit includes concrete counterexamples.
The [evidence bundle](cann-a3-div-20261017.json) retains the native reports,
independent audit, raw tensor hashes and CPU-reference replay.

`quantize_dynamic_a8` applies this measured factor correction only to
ordinary CPU BF16 DynamicQuant inputs, with zero rows handled separately.
Its domain requires nonzero amax ≥2^-120. Scale remains FP32
`amax*FP32(1/127)`; codes use FP32 Mul and nearest-even RINT. C8 and the
FP32 fused Prolog/SwiGLU quantizers retain their separate formulas.

The [fast CPU check](../../../tests/glm52/quant_div_probe/check_reference.py)
passes all 31,744 positive/negative half-amax rows, maxima and zero rows.
On the frozen 8K layer-0 input, all 6,144 input codes, 2,048 Prolog query
codes and 4,096 final BF16 query values are bitwise equal to CANN. Reusing
the already verified FLOAT key/head-weight outputs reproduces all 2,048
selected indices; the cutoff gap is 0.35972595. Prolog's internal FP32
query scale still differs by one ULP (`4.6566e-10`), without changing the
BF16 query. The [updated 8K HF reference](hf-8k-div-calibrated-20261017.json)
passes all 1,273 two-step/reset checks at positions 8192 and 8193 and
predicts `[382,262]`. Its raw directory is
`/workspace/offline/cann-div-reference-20261017/hf-8k`.
Full TP16 reset and numerical acceptance follow these checks.

An additional CPU replay covers the captured layer-4 MoE dispatch on all
16 ranks at both positions, before and after reset: 512 dispatched rows,
3,145,728 INT8 elements and all scales are exact. Raw report:
`/workspace/offline/cann-div-reference-20261017/dispatch-quant-reference-01.json`.
No performance measurement is attached to these results.

The independent rational audit's source is
`/workspace/offline/cann-div-probe-20261017/independent-cpu-01.py`, SHA256
`cc18f3a31c9c8484d45dbb4f510c11556b88fc4f052b21962f21fe19b5877248`.
Its entry binds the source text to the report:

```python
from pathlib import Path
path = Path("/workspace/offline/cann-div-probe-20261017/independent-cpu-01.py")
ANALYSIS_SOURCE = path.read_text()
exec(compile(ANALYSIS_SOURCE, str(path), "exec"))
```

## Reproduce on the server

Raw runs are under `/workspace/offline/cann-div-probe-20261017` and
`/workspace/offline/cann-div-reference-20261017`. From this repository:

```bash
source /workspace/environment/activate.sh
cmake -S tests/glm52/quant_div_probe -B /tmp/quant-div-build \
  -DASCEND_CANN_PACKAGE_PATH=/opt/ascend/cann-9.2.0 \
  -DSOC_VERSION=Ascend910_9392 -DRUN_MODE=npu \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_PYTHON_EXECUTABLE=/opt/trt-asc/venv/bin/python3
cmake --build /tmp/quant-div-build -j4
python tests/glm52/quant_div_probe/run.py \
  --capture /workspace/offline/cann-indexer-20261017/device-02 \
  --library /tmp/quant-div-build/lib/libquant_div_probe.so \
  --output /tmp/quant-div-sample.json --device npu:15
python tests/glm52/quant_div_probe/calibrate.py \
  --library /tmp/quant-div-build/lib/libquant_div_probe.so \
  --output /tmp/quant-div-calibration.json --device npu:15
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  python tests/glm52/quant_div_probe/check_reference.py \
    --calibration /workspace/offline/cann-div-probe-20261017/calibration-01/result.pt \
    --capture /workspace/offline/cann-indexer-20261017/device-02 \
    --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
    --output /tmp/quant-div-reference.json
```

The native probes reserve 2 MiB in the recorded run and cap the allocator
at 1.5 GiB. The reference check uses one CPU thread and no NPU calls.
