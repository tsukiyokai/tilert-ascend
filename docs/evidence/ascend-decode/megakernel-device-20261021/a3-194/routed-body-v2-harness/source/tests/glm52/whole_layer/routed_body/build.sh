#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:?Usage: build.sh NEW_OUTPUT_DIRECTORY FROZEN_PARENT}
parent=${2:?Usage: build.sh NEW_OUTPUT_DIRECTORY FROZEN_PARENT}
python_bin=${PYTHON_BIN:-python3}
"$python_bin" "$here/prepare.py" "$parent" "$out"
out=$(cd -- "$out" && pwd)
"$python_bin" - "$out/torch-build.json" <<'PY'
import json
from pathlib import Path
import sys
import torch

Path(sys.argv[1]).write_text(json.dumps({
    "site_packages": str(Path(torch.__file__).resolve().parent.parent),
    "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
}) + "\n")
PY
mapfile -t torch_build < <("$python_bin" - "$out/torch-build.json" <<'PY'
import json
from pathlib import Path
import sys

data = json.loads(Path(sys.argv[1]).read_text())
print(data["site_packages"])
print(data["cxx11_abi"])
PY
)
SITE_PACKAGES=${torch_build[0]} TORCH_CXX11_ABI=${torch_build[1]} \
ROUTED_ROW_BALANCED=1 ROUTED_SHARED_PIPELINE=1 ROUTED_INPUT_FRONT=1 \
    bash "$out/source/routed_int4/build_so.sh" "$out"
"$python_bin" - "$out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch_npu

output = Path(sys.argv[1])
report = json.loads((output / "source-transforms.json").read_text())
for name, expected in report["sources"].items():
    if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Build source changed: {name}")
library = output / "routed_int4_torch.so"
torch.ops.load_library(str(library))
if torch.npu.is_initialized():
    raise RuntimeError("CPU build initialized NPU")
parent = json.loads((output / "parent-manifest.json").read_text())
report.update({
    "namespace": "tilert_pipeline_body",
    "torch": torch.__version__, "torch_npu": torch_npu.__version__,
    "npu_initialized": False,
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "scratch_tensors": parent["scratch_tensors"],
    "tilings": parent["tilings"],
    "device_validation": "not run",
})
(output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"library": str(library),
                  "sha256": report["library_sha256"],
                  "npu_initialized": False}))
PY
