#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_out=${1:-$routed_here/build-gu-pad16}
mkdir -p "$routed_out/source"
routed_out=$(cd -- "$routed_out" && pwd)
routed_sources=(routed_int4_binding.cpp routed_int4_contract.h
  routed_int4_dispatch.h routed_int4_kernel.cpp routed_int4_kernel.h
  routed_int4_tiling.cpp routed_int4_trace.h routed_int4_profile.h
  build_common.sh build_so.sh build_gu_pad16.sh contract_test.cpp
  dump_tiling.cpp check.py bench_weight_banks.py)
for routed_file in "${routed_sources[@]}"; do
  cp -p "$routed_here/$routed_file" "$routed_out/source/$routed_file"
done
sha256sum "${routed_sources[@]/#/$routed_out/source/}" \
  > "$routed_out/source-identity.sha256"
ROUTED_ROW_BALANCED=1 ROUTED_GU_PAD16=1 \
  bash "$routed_out/source/build_so.sh" "$routed_out"
routed_compile "$routed_out/contract.log" "$routed_cxx" -std=c++17 -O2 \
  -DROUTED_GU_PAD16 "$routed_out/source/contract_test.cpp" \
  -o "$routed_out/contract_test"
"$routed_out/contract_test" >> "$routed_out/contract.log"
routed_compile "$routed_out/tiling_dump.log" "$routed_cxx" -std=c++17 -O2 \
  -DROUTED_GU_PAD16 "${routed_host_includes[@]}" \
  "$routed_out/source/dump_tiling.cpp" "$routed_out/tiling.o" \
  -o "$routed_out/tiling_dump" "${routed_libs[@]}"
"$routed_out/tiling_dump" > "$routed_out/tiling.jsonl"
"$routed_bisheng" --version > "$routed_out/compiler.txt"
sha256sum -c "$routed_out/source-identity.sha256" \
  > "$routed_out/source-verification.log"
sha256sum "${routed_sources[@]/#/$routed_out/source/}" \
  "$routed_out"/{kernel.o,tiling.o,binding.o,routed_int4_torch.so} \
  > "$routed_out/identity.sha256"
"${PYTHON_BIN:-python3}" - "$routed_target/asc" "$routed_out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

sdk, output = map(Path, sys.argv[1:])
headers = (
    "include/adv_api/matmul/matmul_intf.h",
    "include/adv_api/matmul/tiling.h",
    "impl/adv_api/detail/matmul/scheduler/base/scheduler_norm_base.h",
    "impl/adv_api/detail/matmul/scheduler/iterator/m_loop/m_loop_norm_base.h",
    "impl/adv_api/detail/matmul/stage/compute/mmad_compute.h",
    "impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h",
)
report = {
    "variant": "row-balanced INT32 Norm with gate/up M16 padding only",
    "compiled_from_snapshot": str(output / "source"),
    "own_sources": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((output / "source").iterdir())
    },
    "sdk_headers": {
        str(sdk / name): hashlib.sha256((sdk / name).read_bytes()).hexdigest()
        for name in headers
    },
    "gate_rows_per_route": 16,
    "valid_gate_rows_per_expert": "2 * count",
    "matrix_gate_rows_per_expert": "ceil(2 * count / 16) * 16",
    "tilings": [
        json.loads(line) for line in (output / "tiling.jsonl").read_text().splitlines()
    ],
}
(output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
PY
