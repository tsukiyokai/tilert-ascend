#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_out=${1:-$routed_here/build-input-front}
mkdir -p "$routed_out/source"
routed_out=$(cd -- "$routed_out" && pwd)
routed_quant_source=$routed_cann/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/moe_init_routing_v3/op_kernel/moe_v3_full_load_dynamic_quant.h
routed_quant_sha=97d7a1a73709650aa6a88f2369a709ace98779f0591ec3f7f8c7f7b811494020
printf '%s  %s\n' "$routed_quant_sha" "$routed_quant_source" \
  | sha256sum -c > "$routed_out/quant-source-verification.log"
routed_sources=(routed_int4_binding.cpp routed_int4_contract.h
  routed_int4_dispatch.h routed_int4_front.h routed_int4_kernel.cpp
  routed_int4_kernel.h routed_int4_tiling.cpp routed_int4_trace.h
  routed_int4_profile.h build_common.sh build_so.sh build_input_front.sh
  contract_test.cpp dump_tiling.cpp check.py check_input_front.py test_input_front.py)
for routed_file in "${routed_sources[@]}"; do
  cp -p "$routed_here/$routed_file" "$routed_out/source/$routed_file"
done
sha256sum "${routed_sources[@]/#/$routed_out/source/}" \
  > "$routed_out/source-identity.sha256"
ROUTED_ROW_BALANCED=1 ROUTED_INPUT_FRONT=1 \
  bash "$routed_out/source/build_so.sh" "$routed_out"
routed_compile "$routed_out/contract.log" "$routed_cxx" -std=c++17 -O2 \
  -DROUTED_INPUT_FRONT "$routed_out/source/contract_test.cpp" \
  -o "$routed_out/contract_test"
"$routed_out/contract_test" >> "$routed_out/contract.log"
routed_compile "$routed_out/tiling_dump.log" "$routed_cxx" -std=c++17 -O2 \
  -DROUTED_INPUT_FRONT "${routed_host_includes[@]}" \
  "$routed_out/source/dump_tiling.cpp" "$routed_out/tiling.o" \
  -o "$routed_out/tiling_dump" "${routed_libs[@]}"
"$routed_out/tiling_dump" > "$routed_out/tiling.jsonl"
"$routed_bisheng" --version > "$routed_out/compiler.txt"
sha256sum -c "$routed_out/source-identity.sha256" \
  > "$routed_out/source-verification.log"
sha256sum "${routed_sources[@]/#/$routed_out/source/}" \
  "$routed_out"/{kernel.o,tiling.o,binding.o,routed_int4_torch.so} \
  > "$routed_out/identity.sha256"
"${PYTHON_BIN:-python3}" - "$routed_target/asc" "$routed_out" \
  "$routed_quant_source" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

sdk, output, quant_source = map(Path, sys.argv[1:])
headers = (
    "include/adv_api/matmul/matmul_intf.h",
    "include/adv_api/matmul/tiling.h",
    "impl/basic_api/dav_c220/kernel_operator_vec_reduce_impl.h",
    "impl/basic_api/dav_c220/kernel_operator_vec_binary_impl.h",
    "impl/basic_api/dav_c220/kernel_operator_vec_unary_impl.h",
)
report = {
    "variant": "BF16 input front with balanced INT32 Norm projections",
    "namespace": "tilert_front",
    "c_abi": ["routed_int4_front_launch", "routed_int4_front_build_tiling"],
    "compiled_from_snapshot": str(output / "source"),
    "flags": ["ROUTED_INPUT_FRONT", "ROUTED_ROW_BALANCED", "ROUTED_BALANCED_NAMESPACE"],
    "own_sources": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((output / "source").iterdir())
    },
    "sdk_headers": {
        str(sdk / name): hashlib.sha256((sdk / name).read_bytes()).hexdigest()
        for name in headers
    },
    "quant_contract_source": {
        "path": str(quant_source),
        "sha256": hashlib.sha256(quant_source.read_bytes()).hexdigest(),
    },
    "front_ub_bytes_per_aiv": 76320,
    "front_ownership": "AIV0 stable counting order; token%(2*blocks) quantizes each input row",
    "gate_rows_per_route": 2,
    "tilings": [
        json.loads(line) for line in (output / "tiling.jsonl").read_text().splitlines()
    ],
}
(output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
PY
