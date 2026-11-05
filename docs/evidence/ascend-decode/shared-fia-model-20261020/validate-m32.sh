#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
base=/workspace/experiments/shared-fia-20261020
tree=$base/tree
out=$base/validate-m32
export PYTHONPATH="$tree:${PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export VLLM_CACHE_ROOT=$base/cache-validate-m32
export HCCL_NPU_SOCKET_PORT_RANGE=53000-53019
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
bash "$tree/tests/glm52/precheck.sh" 57
mkdir "$out"
mkdir -p "$VLLM_CACHE_ROOT"
cd "$tree"
date -Is
set +e
timeout 2400 torchrun --nproc-per-node 16 --master-port 53201 tests/glm52/run_cann_mtp_rounds.py --checkpoint /workspace/models/GLM-5.2-w4a8c8 --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 --source-commit fba76ee41eeb236265d806856e938f5134c1446b --width 32 --max-tokens 33 --capacity 9344 --ignore-eos --attention-kernel shared_fia --fp32-head --target-graph --gpu-memory-utilization 0.78 --output "$out"
status=$?
printf '%s\n' "$status" > "$base/validate-m32.exit"
date -Is
exit "$status"
