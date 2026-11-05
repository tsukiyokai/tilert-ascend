#!/bin/bash
set -eo pipefail
source /workspace/environment/activate.sh
SOURCE=/workspace/offline/cann-layer-major-20261018-02/source
export PYTHONPATH="$SOURCE:$SOURCE/tests/glm52:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH=/workspace/offline/cann-workspace-owner-20261017/kernel-overlay/vendors/custom_transformer:/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 HCCL_DETERMINISTIC=true HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AI_CPU HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
exec torchrun --standalone --nnodes=1 --nproc-per-node=16  "$SOURCE/tests/glm52/run_cann_layer_major.py"  --checkpoint /workspace/models/GLM-5.2-w4a8c8  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00  --teacher-json "$SOURCE/docs/evidence/ascend-decode/vllm-baseline-20261017/teacher-32.json"  --prompt-json /workspace/experiments/glm52/prompt-8192-code-review.json   --layers 78 --steps 32 --attention-kernel sfa  --output /workspace/offline/cann-layer-major-20261018-02/device-78x32-sfa-01
