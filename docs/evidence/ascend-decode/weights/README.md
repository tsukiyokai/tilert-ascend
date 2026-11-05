# Real W4A8 checkpoint validation

Run on 2026-10-16 in `a3-gs:trt-asc`: Torch/TorchNPU 2.12, CANN 9.2.0,
one Ascend910_9392 die. Inputs are seeded synthetic tensors; weights come
from layer 3, expert 0 of the checkpoint in the [weight contract](../weight-contract.md).

| Run | Result | Scope |
| --- | --- | --- |
| [CPU packing](weights-cpu-final.json) | 102/102 exact | INT32 layout, all 16 TP weight slices and scale-bias identities |
| [NPU projections](weights-npu-snapshot.json) | 172/172; maximum projection NRMSE 0.001108 | rank 0/15, M=1/4 gate/up/down; W8 qA; zero inputs; input preservation |
| [NPU expert](expert-chain.json) | 70/70; maximum output NRMSE 0.003695 | merged gate/up, BF16 SiLU/product, dynamic A8 and down, rank 0/15, M=1/4 |

The expert output is one TP rank's contribution, before routing weights and
allreduce. The reference uses separate CPU integer projections and explicit
BF16 rounding after projection, SiLU and multiplication. It does not model
the different rounding of vLLM's fused SwiGLU kernel. Reports contain source
hashes and individual comparisons. Full-layer parity and TPOT remain in the
[plan](../../../plan.md).

## Input and rounding contracts

CANN 9.2's native A8W4 MSD preprocess consumes its INT8 input. The direct
[mutation probe](w4-mutation.json) changed 6,105 of 6,144 activation bytes.
Its output matched the original-input integer reference with NRMSE 0.001108.
`NpuQuantizedLinear.apply_quantized` copies the activation before this native
call, so gate/up may safely share the caller's quantized input. Reference
values are calculated from snapshots taken before execution.

The local source is under
`/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul/op_kernel/`:
`grouped_matmul.cpp:263` initializes preprocessing with the same input/output;
`grouped_matmul_antiquant_a8w4_msd_pre.h:82` binds them, and lines 186/207
write packed high/low parts into the input. The postprocess restores
`16*hi+lo`, applies the channel scale and correction, then the token scale.

The [layout probe with fresh inputs](w4-layout-fresh.json) gives identical
outputs for ND INT32 and INT8→NZ→INT32 weights. Converting to NZ after the
INT32 view gives different results on this stack. The
[earlier report](weights-npu-nd.json) read its integer reference after native
input mutation; its projection errors do not measure the projection contract.

Dynamic A8 returns `scale=amax/127`, then multiplies by the FP32 reciprocal
before nearest-even rounding. In the [fixed-row probe](a8-rounding.json),
`x*(1/scale)` matched all INT8 values; `x/scale` differed at 11 half-integer
positions. The CPU reference preserves this operation order.

## Reproduce

From `/workspace/src/TileRT-Ascend`, after
`source /opt/trt-asc/activate.sh`:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACL_OP_INIT_MODE=1 \
  ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python -m tilert.models.glm_5_2_ascend.check_weights \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 --device npu:0 \
  --source-commit f554fa57d0a69e7a44ac4f3134d0134b4964e848 \
  --output /workspace/experiments/glm52/weights-npu.json
PYTHONPATH="$PWD:$PYTHONPATH" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  ACL_OP_INIT_MODE=1 ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python tests/glm52/check_expert.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 --device npu:0 \
  --source-commit f554fa57d0a69e7a44ac4f3134d0134b4964e848 \
  --output /workspace/experiments/glm52/expert-chain.json
```
