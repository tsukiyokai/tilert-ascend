#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Usage: bench_run.sh preflight|device OUTPUT_DIRECTORY}
output=${2:?Usage: bench_run.sh preflight|device OUTPUT_DIRECTORY}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_root=${FFN_TASK_ROOT:-/workspace}
composed="$task_root/tests/glm52/whole_layer/ffn_composed/build-v2"
candidate="$task_root/tests/glm52/whole_layer/ffn_down_overlap/build-v3"
bridge="$task_root/logs/ffn-bridge-runtime212-v1"
body="$task_root/tests/glm52/whole_layer/routed_body/build-v2"
python_bin=${PYTHON_BIN:-python3}
extra=()
case "$mode" in
    preflight) extra=(--preflight-only) ;;
    device) ;;
    *) exit 2 ;;
esac

"$python_bin" - "$composed/ffn_composed.so" "$bridge/native/ffn_bridge.so" \
    "$body/routed_int4_torch.so" "$candidate/ffn_down_batch.so" <<'PY'
import hashlib
from pathlib import Path
import sys

expected = (
    "3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655",
    "57392089102d11b0d3c9389228345406e7473c50b9877876332ba147dba55e6c",
    "6dcb8401cf722f9b3c3650d2432321ce6100d79aa98ff329d3a17a63a0f1d6c4",
    "c83c9cb66f9109b810023d95794a4907734e73f66b9526ffd0c3b5987e63d984",
)
for name, identity in zip(sys.argv[1:], expected):
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != identity:
        raise RuntimeError(f"Unexpected frozen library: {name}: {actual}")
PY

exec "$python_bin" "$here/bench.py" \
    --library "$composed/ffn_composed.so" \
    --candidate-library "$candidate/ffn_down_batch.so" \
    --bridge-library "$bridge/native/ffn_bridge.so" \
    --body-library "$body/routed_int4_torch.so" \
    --checker "$composed/harness-v1/check.py" \
    --bridge-helper "$bridge/check.py" \
    --body-helper "$composed/body/source/routed_int4/check.py" \
    --rows 12 --blocks 24 --banks 16 --calls 16 --samples 30 \
    --iterations 30 --replays 8 --timed-case 3 \
    --shared-scale-dtype bfloat16 --output "$output" "${extra[@]}"
