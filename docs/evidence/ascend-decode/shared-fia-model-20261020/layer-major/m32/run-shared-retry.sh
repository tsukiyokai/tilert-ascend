#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
base=/workspace/experiments/layer-major-m32-20261020
arm=${1:?sfa or shared_fia}
layers=${2:-8}
out=$base/$arm-$layers-retry01
test ! -e "$out"
export PYTHONPATH="$base/source:$PYTHONPATH"
export ASCEND_CUSTOM_OPP_PATH=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer:$ASCEND_CUSTOM_OPP_PATH
export HCCL_DETERMINISTIC=true
export HCCL_OP_EXPANSION_MODE=AI_CPU
export HCCL_BUFFSIZE=200
export HCCL_NPU_SOCKET_PORT_RANGE=55500-55519
export VLLM_CACHE_ROOT=$base/cache-$arm-$layers
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export OMP_NUM_THREADS=1
bash "$base/source/tests/glm52/precheck.sh" 3
python3 - <<'GATE'
import re
import subprocess
import time

for sample in range(2):
    text = subprocess.check_output(["npu-smi", "info"], text=True)
    rows = re.findall(
        r"\|\s+\d+\s+\d+\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)"
        r"\s+0\s*/\s*0\s+(\d+)\s*/\s*(\d+)", text,
    )
    if len(rows) != 16 or any(int(util) != 0 for util, used, total in rows):
        raise SystemExit("AIC activity gate refused this experiment")
    if sample == 0:
        time.sleep(2)
GATE
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
printf '%s\n' "$status" > "$base/$arm-$layers-retry01.exit"
date -Is
exit "$status"
