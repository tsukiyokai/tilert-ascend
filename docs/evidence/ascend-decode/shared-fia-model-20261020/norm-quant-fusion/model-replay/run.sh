#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
source_root=/workspace/experiments/m32-full-teacher-20261020/two-rounds
base=/workspace/experiments/m32-full-teacher-20261020/norm-fusion/model-replay
layers=${1:?layer count required}
case=${2:?output name required}
arm=${3:?fusion arm required}
root=$base
base=$root/$case
mkdir "$base"
out="$base/output"
test ! -e "$out"
export PYTHONPATH="$source_root/source:$PYTHONPATH"
vendor=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$vendor:$ASCEND_CUSTOM_OPP_PATH"
export HCCL_DETERMINISTIC=true HCCL_OP_EXPANSION_MODE=AI_CPU
export HCCL_BUFFSIZE=200 HCCL_NPU_SOCKET_PORT_RANGE=56400-56419
export VLLM_CACHE_ROOT="$base/cache"
export TORCHINDUCTOR_AUTOGRAD_CACHE=0 OMP_NUM_THREADS=1
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
bash "$source_root/source/tests/glm52/precheck.sh" 3
python3 - "$base" "$source_root" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(sys.argv[1])
samples = []
for sample in range(2):
    raw = subprocess.check_output(['npu-smi', 'info'], text=True)
    (root / f'preflight-{sample}.txt').write_text(raw)
    rows = re.findall(
        r'\|\s+\d+\s+\d+\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)'
        r'\s+0\s*/\s*0\s+(\d+)\s*/\s*(\d+)', raw,
    )
    samples.append(rows)
    if sample == 0:
        time.sleep(2)
passed = all(len(rows) == 16 and all(
    int(total) - int(used) >= 6000
    for util, used, total in rows
) for rows in samples)
(root / 'preflight.json').write_text(json.dumps({
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'passed': passed, 'samples': samples,
    'scope': 'Shared functional validation; no performance measurement',
    'torch_quota_bytes': 2 * 1024 ** 3,
}, indent=2))
if not passed:
    raise SystemExit('Memory check refused model launch')
source_root = Path(sys.argv[2])
files = [p for p in (source_root / 'source').rglob('*') if p.is_file()]
(root / 'source-manifest.json').write_text(json.dumps({
    str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in files
}, indent=2))
PY
mkdir "$out"
cd "$source_root/source"
gmm="$vendor/op_impl/ai_core/tbe/kernel/ascend910_93/grouped_matmul_swiglu_quant_v2/GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.o"
sfa="$vendor/op_impl/ai_core/tbe/kernel/ascend910_93/sparse_flash_attention/SparseFlashAttention_2e25ad8fefac1f04c03bd78f2d2ab366.o"
watch=/workspace/experiments/gmm-compact-20261020/watch_kernel.py
date -Is
set +e
timeout --signal=TERM --kill-after=30s 3600 \
    python3 "$watch" "$gmm" "$base/gmm-access.json" \
    python3 "$watch" "$sfa" "$base/sfa-access.json" \
    torchrun --standalone --nproc-per-node=16 \
    "$root/replay_intervention.py" --norm-fusion "$arm" \
    --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
    --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
    --teacher-json /workspace/experiments/shared-fia-20261020/teacher-32.json \
    --prompt-json /workspace/experiments/shared-fia-20261020/prompt.json \
    --layers "$layers" --steps 32 --query-width 8 --attention-kernel sfa \
    --round-trace "$source_root/chain-w8-rank0.json" \
    --fp32-head --output "$out"
status=$?
printf '%s\n' "$status" > "$base/exit.txt"
date -Is
exit "$status"
