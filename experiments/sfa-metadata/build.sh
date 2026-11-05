#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$here/../.." && pwd)
out=${1:?Pass a new build directory}
overlay=${SFA_METADATA_OVERLAY:?Set the prepared SFA overlay}
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}
compiler=${BISHENG:-}
if [[ -z "$compiler" ]]; then
    for candidate in "$cann_root/bin/bisheng" \
        "$cann_root/compiler/ccec_compiler/bin/bisheng"; do
        if [[ -x "$candidate" ]]; then
            compiler=$candidate
            break
        fi
    done
fi
if [[ ! -x "$compiler" ]]; then
    echo 'Set BISHENG to the activated CANN device compiler.' >&2
    exit 2
fi
if [[ -e "$out" ]]; then
    echo 'Use a new build directory.' >&2
    exit 2
fi
SFA_METADATA_BISHENG="$compiler" SFA_METADATA_OVERLAY="$overlay" \
    BISHENG="$here/compiler.sh" \
    bash "$repository/tests/glm52/whole_layer/attention_build.sh" "$out"
"${PYTHON_BIN:-${PYTHON:-python3}}" - "$overlay" "$out" "$compiler" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

overlay, output = (Path(path).resolve() for path in sys.argv[1:3])
manifest = json.loads((overlay / "manifest.json").read_text())
directory = overlay / "attention/sparse_flash_attention/op_kernel/arch22"
for name, expected in manifest["overlay_sha256"].items():
    if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Overlay identity changed: {name}")
path = output / "provenance.json"
report = json.loads(path.read_text())
report["sfa_metadata_overlay"] = manifest
report["sfa_metadata_compiler"] = sys.argv[3]
report["sfa_metadata_compile_prefix"] = ["-I", str(overlay)]
path.write_text(json.dumps(report, indent=2) + "\n")
PY
