#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
base=/workspace/experiments/teacher-routes-20261020
tree=$base/source-7747b5c
out=$base/capture-lookup-m32
export PYTHONPATH="$tree:${PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export VLLM_CACHE_ROOT=$base/cache-capture-lookup-m32
export HCCL_NPU_SOCKET_PORT_RANGE=53020-53039
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
bash "$tree/tests/glm52/precheck.sh" 57
mkdir "$out"
mkdir -p "$VLLM_CACHE_ROOT"
cd "$out"
date -Is
set +e
timeout 2400 torchrun --nproc-per-node 16 --master-port 53221 \
  "$tree/tests/glm52/capture_cann_routes.py" \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --source-commit 7747b5ca132aafe964fe5ac5c6b9af4321a347f8 \
  --lookup --lookup-unigram --width 32 --max-tokens 1024 --warmup 1 --requests 1 \
  --capacity 9344 --attention-kernel shared_fia \
  --fp32-head --target-graph --gpu-memory-utilization 0.78 --output "$out"
status=$?
printf '%s\n' "$status" > "$base/capture-lookup-m32.exit"
date -Is
exit "$status"
