#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Usage: bench_run.sh preflight|device OUTPUT_DIRECTORY}
output=${2:?Usage: bench_run.sh preflight|device OUTPUT_DIRECTORY}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_root=${FFN_TASK_ROOT:-/workspace}
composed="$task_root/tests/glm52/whole_layer/ffn_composed/build-v2"
parent="$task_root/tests/glm52/whole_layer/ffn_token_output/build-v2"
candidate="$task_root/tests/glm52/whole_layer/ffn_token_join/build-v1"
bridge="$task_root/logs/ffn-bridge-runtime212-v1"
body="$task_root/tests/glm52/whole_layer/routed_body/build-v2"
python_bin=${PYTHON_BIN:-python3}
extra=()
case "$mode" in
    preflight) extra=(--preflight-only) ;;
    device) ;;
    *) exit 2 ;;
esac

"$python_bin" - "$parent/ffn_token_output.so" "$bridge/native/ffn_bridge.so" \
    "$body/routed_int4_torch.so" "$candidate/ffn_token_join.so" <<'PY'
import hashlib
from pathlib import Path
import sys

expected = (
    "e92a9cbe4d11307a8ea8beae20b5cbe845be10f9aba5280cbc3f438b049ed041",
    "57392089102d11b0d3c9389228345406e7473c50b9877876332ba147dba55e6c",
    "6dcb8401cf722f9b3c3650d2432321ce6100d79aa98ff329d3a17a63a0f1d6c4",
    "4417fd158e48a2da4d59e61fec48ebd0dd6e113022f13cbeb4f01d596eead757",
)
for name, identity in zip(sys.argv[1:], expected):
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != identity:
        raise RuntimeError(f"Unexpected frozen library: {name}: {actual}")
PY

exec "$python_bin" "$here/bench.py" \
    --library "$parent/ffn_token_output.so" \
    --candidate-library "$candidate/ffn_token_join.so" \
    --bridge-library "$bridge/native/ffn_bridge.so" \
    --body-library "$body/routed_int4_torch.so" \
    --checker "$composed/harness-v1/check.py" \
    --bridge-helper "$bridge/check.py" \
    --body-helper "$composed/body/source/routed_int4/check.py" \
    --rows 12 --blocks 24 --banks 16 --calls 16 --samples 30 \
    --iterations 30 --replays 8 --timed-case 3 \
    --shared-scale-dtype bfloat16 --output "$output" "${extra[@]}"
