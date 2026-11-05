# Layer-streamed CANN verification

The verifier constructs one decoder layer from raw CPU checkpoint tensors,
executes its real TP16 forward with the captured 8K prefix, and discards
the processed device weights. It retains separate hidden and residual
states between layers. Processed NZ weights never pass through CPU storage.

## Five-layer result

On `a3-gs:trt-asc`, layers 0 through 4 pass 3,568/3,568 execution and
reconstruction checks across all 16 ranks. The input is the original
8192-token prefix and pending token 6500. The path includes three dense
layers, two W4 MoE layers, actual M1 head projections and the complete
SwiGLU barrier vendor. Every decoder forward performs two real FP32 SUM
reductions of `[1,6144]`.

Each layer's parameters and expert submodule are released after execution.
Reconstruction repeats raw hidden/residual, KV, logits and ordered indices
exactly. Layers 3 and 4 preserve the actual shared selection from layer 2.
The [result and rank receipts](streamed-five-20261017.json) bind all
192 checked artifacts and 18 distinct checked source files. Frozen source
and outputs are under
`/workspace/offline/cann-streamed-five-20261017/source-01` and `device-01`.

The original HF comparison recorded 70/80 under a CPU BF16 matmul
reference subsequently falsified by exact dot products. The table below
is historical, not an acceptance result. The corrected reference disables
oneDNN, retains the calibrated Div boundary, and is rerun before the
78-layer curves. Index equality becomes a diagnostic; logits report their
BF16 spacing before interpreting a fixed absolute difference. Hidden
NRMSE and cosine thresholds await the full curves. End-to-end frozen
teacher trajectories determine acceptance, as specified in
[the plan](../../plan.md#4-可独立验收).
The [corrected five-layer diagnostics](hf-corrected-20261017/README.md)
use the same frozen CANN execution and the repaired HF reference.

| Layer | Hidden NRMSE | Logits max-abs | Common indices / 2,048 |
| --- | ---: | ---: | ---: |
| 0 | 0.003889 | 0.062500 | 2,048 |
| 1 | 0.006168 | 0.077148 | 2,046 |
| 2 | 0.007783 | 0.093750 | 2,044 |
| 3 | 0.009019 | 0.093750 | 2,044 |
| 4 | 0.009431 | 0.140625 | 2,044 |

The [layer-1 input counterfactual](layer1-index-input-20261017.json) holds
the checkpoint, 8K index cache, position and HF implementation fixed.
Its control reproduces the frozen HF choices and new key exactly.
Replacing only the merged input before layer-1 normalization with the
actual CANN input changes agreement from 2,046 to all 2,048 selected
positions. Selection order still differs. The new BF16 key passes the
existing numerical bound but differs in 10/128 values. This locates the
selection-set discrepancy in the incoming trajectory; the earlier
operator that introduced that input difference remains to be isolated.

The [layer-4 joint intervention](layer4-input-20261017.json) replaces both
the incoming hidden state and shared indices. Its unchanged HF control
reproduces six frozen fields exactly. Logits max-abs falls from 0.140625
to 0.03125 and merged-hidden NRMSE from 0.009430 to 0.003050. M1 and M77
HF head results are bitwise equal in both cases. Applying the HF head to
the actual CANN merged state gives max-abs 0.015625. The post-attention
residual still differs at 0.002103 NRMSE; the ensuing FFN has different
inputs, so its output difference does not isolate a local FFN error.

## Residency

The original million-position RoPE tables occupy 384 MiB. The verifier
clones their already-computed first 8,320 rows, checks exact equality, and
replaces both module buffers and global references. Retained table storage
is 3.046875 MiB; frequency, scaling and maximum-position parameters retain
their original values. Subsequent layers reuse the same RoPE object.

| Measurement | Result |
| --- | --- |
| Torch allocator limit per rank | 2 GiB |
| Peak allocated, including initial full RoPE construction | 907.62 MiB |
| Allocated after each five-layer phase | 245.5 MiB |
| Decoder, expert and parameter weak references after GC | Cleared on every rank |

The retained allocation includes the bounded RoPE cache and base weights.
The measurements cover Torch allocations; physical HBM also holds HCCL
and runtime resources. The test covers one consumed token and five layers;
complete-model HF precision and TPOT remain open.

The earlier [layer-0 prototype](streamed-layer0-20261017.json) passes
560/560 checks with a full RoPE table and 626.43 MiB retained allocation.

## Reproduce

Use a fresh output directory. `--stop-layer 0` runs the first layer only.

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
cann_overlay=/workspace/offline/cann-workspace-owner-20261017/kernel-overlay/vendors/custom_transformer
cann_vendor=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
env HCCL_BUFFSIZE=4 HCCL_OP_EXPANSION_MODE=AI_CPU \
  HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto \
  TASK_QUEUE_ENABLE=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  ASCEND_CUSTOM_OPP_PATH="$cann_overlay:$cann_vendor" \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$PWD:$PWD/tests/glm52:$PYTHONPATH" \
  timeout --kill-after=15s 600s python -m torch.distributed.run \
  --standalone --nproc-per-node=16 tests/glm52/run_cann_streamed.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --baseline /workspace/offline/cann-routed-stage-20261017/continuation8k \
  --stop-layer 4 --output /workspace/offline/cann-streamed-check
```

## Shared-host communication

With the same source, input and other launch settings, the AIV run reaches
the first post-forward device synchronization on every rank and times out
at 300 seconds. Changing only `HCCL_OP_EXPANSION_MODE` to `AI_CPU` completes
the numerical checks. The saved Python stacks locate the waiting boundary;
they do not identify which queued device operation stalled.

Shared numerical checks use `AI_CPU`. The HCCL A3 documentation at
`b524f5a5946bd98216abe0cc55a8a69137355593` names it as the default mode,
advises against concurrent AIV processes on one NPU, and disallows parallel
AIV communication domains. Sources are `docs/zh/user_guide/constraints.md`
and `docs/zh/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md` in the HCCL
repository. These documentation sources are independent of the installed
CANN binary revision. Performance experiments use an idle host and report
their communication mode explicitly.
