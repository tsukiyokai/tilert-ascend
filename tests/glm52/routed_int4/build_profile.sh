#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_out=${1:-$routed_here/build-profile}
mkdir -p "$routed_out"
routed_profile_flags=()
if [[ ${ROUTED_ROW_BALANCED:-0} == 1 ]]; then
  routed_profile_flags+=(-DROUTED_ROW_BALANCED)
fi
routed_compile "$routed_out/kernel.log" "$routed_bisheng" \
  "${routed_device_flags[@]}" "${routed_device_includes[@]}" \
  "${routed_profile_flags[@]}" \
  -DROUTED_ENABLE_PROFILE -c "$routed_here/routed_int4_kernel.cpp" \
  -o "$routed_out/profile.o"
routed_compile "$routed_out/link.log" "$routed_cxx" -shared \
  "$routed_out/profile.o" -o "$routed_out/routed_int4_profile.so" \
  "${routed_libs[@]}"
"${PYTHON_BIN:-python3}" - "$routed_here" "$routed_out" \
  "$routed_cann" "$routed_target" "$routed_bisheng" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

source, output, cann, target, compiler = map(Path, sys.argv[1:])
files = [source / name for name in (
    "routed_int4_kernel.cpp", "routed_int4_kernel.h", "routed_int4_contract.h",
    "routed_int4_dispatch.h", "routed_int4_trace.h", "routed_int4_profile.h",
    "profile_routed.py", "build_profile.sh", "build_common.sh",
)]
headers = [target / name for name in (
    "asc/impl/basic_api/dav_c220/kernel_operator_sys_var_impl.h",
    "asc/impl/basic_api/dav_c220/kernel_operator_sync_impl.h",
    "asc/include/basic_api/kernel_operator_cache_intf.h",
    "asc/include/adv_api/matmul/matmul_intf.h",
)]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

version = subprocess.run([str(compiler), "--version"], check=True,
                         capture_output=True, text=True)
report = {
    "cann_root": str(cann.resolve()), "compiler": version.stdout + version.stderr,
    "row_balanced": os.environ.get("ROUTED_ROW_BALANCED", "0") == "1",
    "sources": {str(path): sha(path) for path in files},
    "sdk_headers": {str(path): sha(path) for path in headers if path.is_file()},
    "artifacts": {str(path): sha(path) for path in output.glob("*.so")},
}
(output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
PY
printf 'Built %s\n' "$routed_out/routed_int4_profile.so"
