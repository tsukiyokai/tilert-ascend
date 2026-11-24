# CANN sparse MLA with frozen Prolog inputs

On `a3-gs:trt-asc`, die 15 runs the existing custom SFA and CANN transpose
batch matmul for layer 0, logical TP16 rank 0: four heads, latent512,
RoPE64 and value256. Inputs are the actual [Prolog observations](mla-prolog.md)
at teacher positions 0/1. The page contains both rows; each call exposes
only its valid prefix and selected indices. No model or HCCL group loads.

The scale is `1/sqrt(192+64)=1/16`, from the original QK dimension.
This checkpoint uses default RoPE without a YaRN scale correction. Query
absorption into 512 channels retains the same attention scale.

## Results

The [device run](cann-mla-device-20261017.json) passes 57/57 checks.
The [CPU preflight](cann-mla-cpu-20261017.json) passes 9/9 without device
initialization. Numerical thresholds remain per-head NRMSE ≤0.01 and
cosine ≥0.999; inputs, indices and caches are checked exactly.

| Comparison | Checks | Maximum NRMSE | Minimum cosine |
| --- | --- | --- | --- |
| Reconstructed absorbed query vs actual query | 8/8 | 0, bitwise equal | 0.99999988 |
| SFA and compressed chain vs FP32 CPU oracle | 16/16 | 0.002808 | 0.99999613 |
| Value projection from the same device context | 8/8 | 0, bitwise equal | 1 |
| Complete local output vs HF expanded KV | 8/8 | 0.004512 | 0.99998981 |
| Preserved inputs and cache | 16/16 | exact | exact |
| CPU single-key identity | 1/1 | exact | exact |

The expanded reference calls the installed HF `eager_attention_forward`.
Position 0 matches it bitwise; position 1 supplies the nontrivial two-key
softmax case. The unabsorbed CPU query is rebuilt
from the observed INT8 query and scale; all 2,048 absorbed query values at
each position match the device. The value-projection check fixes the actual
SFA context,
separating its arithmetic from the attention comparison.

Peak Torch allocation is 76,951,040 bytes, with a 1.5 GiB allocator cap;
non-Torch runtime allocations are outside this counter. This run covers
two short prefixes and one rank's four heads. Full 8K, cross-rank output
projection, routing and decode trajectory remain the acceptance work.

## Reproduce

Run [the check](../../../tests/glm52/check_cann_mla.py) in the unified
server environment, on a die with sufficient free memory:

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  python tests/glm52/check_cann_mla.py \
  --position-zero /workspace/offline/cann-prolog-20261017/device.pt \
  --position-one /workspace/offline/cann-prolog-pos1-20261017/device-pos1.pt \
  --device npu:15 --output /workspace/offline/mla-replay/device.json
```

`--cpu-only` replaces `--device` for input reconstruction. The report pins
the raw artifacts, checkpoint identity, operator schema, loader, custom
library, source files and package versions. No profiling or latency
measurement is performed by this entry.
