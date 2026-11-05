#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
root=/workspace/experiments/gmm2-compact-20261020/guard-release-screen
arm=${1:?Use A1, B1, B2, A2 in that order}
device=${2:-npu:0}
kernel=$(python3 - "$root" "$arm" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, arm = Path(sys.argv[1]), sys.argv[2]
assert arm in ("A1", "B1", "B2", "A2")
entry = json.loads((root / "identity.json").read_text())[arm[0]]
binary = Path(entry["binary"])
assert entry["elf_sha256"] and hashlib.sha256(binary.read_bytes()).hexdigest() == entry["elf_sha256"]
assert json.loads(binary.with_suffix(".json").read_text())["sha256"] == entry["elf_sha256"]
for name, sha in entry["sources"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
print(binary)
PY
)
vendor=${kernel%/op_impl/*}
export ASCEND_CUSTOM_OPP_PATH="$vendor:${ASCEND_CUSTOM_OPP_PATH:-}"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
cd "$root"
exec python3 /workspace/experiments/gmm-compact-20261020/watch_kernel.py \
    "$kernel" "$root/$arm-access.json" \
    python3 "$root/bench.py" --arm "$arm" --device "$device" \
    --output "$root/$arm.json"
