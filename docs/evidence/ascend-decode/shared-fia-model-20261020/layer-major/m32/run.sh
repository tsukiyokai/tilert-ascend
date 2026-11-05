#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
base=/workspace/experiments/layer-major-m32-20261020
arm=${1:?sfa or shared_fia}
layers=${2:-8}
out=$base/$arm-$layers
test ! -e "$out"
export PYTHONPATH="$base/source:$PYTHONPATH"
export ASCEND_CUSTOM_OPP_PATH=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer:$ASCEND_CUSTOM_OPP_PATH
export HCCL_DETERMINISTIC=true
export HCCL_OP_EXPANSION_MODE=AI_CPU
export HCCL_BUFFSIZE=200
export HCCL_NPU_SOCKET_PORT_RANGE=55400-55419
export VLLM_CACHE_ROOT=$base/cache-$arm-$layers
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export OMP_NUM_THREADS=1
bash "$base/source/tests/glm52/precheck.sh" 3
mkdir "$out"
cd "$base/source"
date -Is
set +e
timeout --signal=TERM --kill-after=30s 1800 \
  torchrun --standalone --nproc-per-node=16 \
  tests/glm52/run_cann_layer_major.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --teacher-json /workspace/experiments/shared-fia-20261020/teacher-32.json \
  --prompt-json /workspace/experiments/shared-fia-20261020/prompt.json \
  --layers "$layers" --steps 32 --query-width 32 \
  --attention-kernel "$arm" --fp32-head --output "$out"
status=$?
printf '%s\n' "$status" > "$base/$arm-$layers.exit"
date -Is
exit "$status"
