#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_CUSTOM_OPP_PATH=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
cd /workspace/experiments/fia-m16-20261020
mkdir -p device
watch=/workspace/experiments/gmm-compact-20261020/watch_kernel.py
sfa=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/sparse_flash_attention/SparseFlashAttention_2e25ad8fefac1f04c03bd78f2d2ab366.o
fia=/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_transformer/fused_infer_attention_score
python "$watch" "$sfa" device/sfa-elf-access.json \
    python "$watch" "$fia/FusedInferAttentionScore_29c62fc24e72bbc0878b0bd85b4a328d.o" device/fia-fp16-optional-elf-access.json \
    python "$watch" "$fia/FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da.o" device/fia-bf16-optional-elf-access.json \
    timeout --signal=TERM --kill-after=5s 90s python -I screen-launched.py > device/run.log 2>&1
