#!/usr/bin/env bash
set -euo pipefail

swiglu_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
swiglu_source=${1:?Pass the vLLM-Ascend csrc/moe/dequant_swiglu_quant directory}
swiglu_out=${2:-$swiglu_here/build-swiglu-reference}
swiglu_cann=${CANN_ROOT:-${ASCEND_HOME_PATH:-}}
if [[ -z "$swiglu_cann" || ! -d "$swiglu_cann" ]]; then
    echo 'Set CANN_ROOT to the matching CANN installation.' >&2
    exit 2
fi
mkdir -p "$swiglu_out/source/kernel_tiling"
cp "$swiglu_source/op_kernel/dequant_swiglu_quant.h" "$swiglu_out/source/"
cp "$swiglu_source/op_host/dequant_swiglu_quant_tiling.h" "$swiglu_out/source/"
cp "$swiglu_here/swiglu_reference.cpp" "$swiglu_out/source/"
"${PYTHON_BIN:-python3}" - "$swiglu_out/source" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
text = (root / "dequant_swiglu_quant_tiling.h").read_text()
body = text.split("BEGIN_TILING_DATA_DEF(DequantSwigluQuantBaseTilingData)", 1)[1]
body = body.split("END_TILING_DATA_DEF", 1)[0]
fields = re.findall(r"TILING_DATA_FIELD_DEF\((\w+),\s*(\w+)\);", body)
if len(fields) != 22 or {kind for kind, _ in fields} != {"int64_t", "float"}:
    raise ValueError("Unexpected source tiling structure")
header = ("#pragma once\n#include <cstdint>\n#include <kernel_tiling.h>\n"
          "struct DequantSwigluQuantBaseTilingData {\n")
header += "".join(f"    {kind} {name};\n" for kind, name in fields)
header += "};\n"
(root / "kernel_tiling/kernel_tiling.h").write_text(header)
PY
swiglu_arch=$swiglu_cann/$(uname -m)-linux
swiglu_bisheng=${BISHENG:-$swiglu_cann/compiler/ccec_compiler/bin/bisheng}
if [[ ! -x "$swiglu_bisheng" ]]; then
    swiglu_bisheng=$swiglu_cann/bin/bisheng
fi
swiglu_includes=(-I "$swiglu_out/source" -I "$swiglu_cann/include")
for relative in asc asc/include asc/include/adv_api asc/include/basic_api \
    asc/include/c_api asc/include/simt_api asc/include/utils \
    asc/include/basic_api/reg_compute asc/impl/adv_api asc/impl/basic_api \
    asc/impl/c_api asc/impl/simt_api asc/impl/utils \
    asc/impl/basic_api/reg_compute tikcpp/tikcfw tikcpp/tikcfw/lib \
    tikcpp/tikcfw/impl tikcpp/tikcfw/interface; do
    if [[ -d "$swiglu_arch/$relative" ]]; then
        swiglu_includes+=(-I "$swiglu_arch/$relative")
    fi
done
"$swiglu_bisheng" -xcce --npu-arch=dav-2201 -std=c++17 -O2 -fPIC \
    -ffp-contract=off "${swiglu_includes[@]}" \
    -c "$swiglu_out/source/swiglu_reference.cpp" -o "$swiglu_out/reference.o"
"${CXX:-g++}" -shared "$swiglu_out/reference.o" \
    -o "$swiglu_out/shared_swiglu_reference.so" \
    -L "$swiglu_arch/lib64" -L "$swiglu_cann/lib64" \
    -lascendcl -lruntime -lunified_dlog -lc_sec -lprofapi
"${PYTHON_BIN:-python3}" - "$swiglu_out" "$swiglu_cann" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

out, cann = map(Path, sys.argv[1:])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Unmodified source kernel with explicit fixed-shape tiling",
    "cann_root": str(cann),
    "source_sha256": {str(path.relative_to(out)): digest(path)
                      for path in sorted((out / "source").rglob("*"))
                      if path.is_file()},
    "library_sha256": digest(out / "shared_swiglu_reference.so"),
}
(out / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
PY
