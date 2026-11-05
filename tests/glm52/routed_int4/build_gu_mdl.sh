#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_out=${1:-$routed_here/build-gu-mdl}
ROUTED_ROW_BALANCED=1 ROUTED_GU_MDL=1 \
  bash "$routed_here/build_so.sh" "$routed_out"
routed_compile "$routed_out/tiling_dump.log" "$routed_cxx" -std=c++17 -O2 \
  -DROUTED_GU_MDL "${routed_host_includes[@]}" \
  "$routed_here/dump_tiling.cpp" "$routed_out/tiling.o" \
  -o "$routed_out/tiling_dump" "${routed_libs[@]}"
"$routed_out/tiling_dump" > "$routed_out/tiling.jsonl"
"$routed_bisheng" --version > "$routed_out/compiler.txt"
sha256sum "$routed_here"/routed_int4_{binding.cpp,contract.h,dispatch.h,kernel.cpp,kernel.h,tiling.cpp,trace.h,profile.h} \
  "$routed_here"/{build_common.sh,build_so.sh,build_gu_mdl.sh,dump_tiling.cpp,check.py,bench_weight_banks.py} \
  "$routed_out"/{kernel.o,tiling.o,binding.o,routed_int4_torch.so} \
  > "$routed_out/identity.sha256"
"${PYTHON_BIN:-python3}" - "$routed_target/asc" "$routed_out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

sdk, output = map(Path, sys.argv[1:])
tilings = [json.loads(line) for line in (output / "tiling.jsonl").read_text().splitlines()]
gate_resources = []
for tiling in tilings:
    if tiling["projection"] != 0:
        continue
    m, n, k = (tiling[name] for name in ("baseM", "baseN", "baseK"))
    l1 = (tiling["depthA1"] * m * k + tiling["depthB1"] * k * n) // 2
    a2 = tiling["dbL0A"] * m * k // 2
    b2 = tiling["dbL0B"] * k * n // 2
    c2 = tiling["dbL0C"] * m * n * 4
    assert l1 == tiling["shareL1Size"] <= 512 * 1024
    assert a2 <= 64 * 1024 and b2 <= 64 * 1024
    assert c2 == tiling["shareL0CSize"] <= 128 * 1024
    assert tiling["shareUbSize"] == 0
    assert (m, n, k, tiling["stepKa"], tiling["stepKb"]) == (64, 256, 256, 24, 4)
    assert tiling["stepM"] == tiling["stepN"] == 1
    gate_resources.append({
        "layout": tiling["layout"], "l1_bytes": l1,
        "l0a_payload_bytes": a2, "l0b_payload_bytes": b2,
        "l0a_pool_bytes": 65536, "l0b_pool_bytes": 65536,
        "l0c_bytes": c2, "ub_bytes": 0,
    })
headers = (
    "include/adv_api/matmul/matmul_intf.h",
    "include/adv_api/matmul/tiling.h",
    "impl/adv_api/detail/matmul/scheduler/base/scheduler_mdl_base.h",
    "impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_mdl_base.h",
    "impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h",
    "impl/adv_api/detail/matmul/utils/matmul_config_utils.h",
)
report = {
    "variant": "row-balanced INT32 with gate/up CFG_MDL only",
    "sdk_headers": {
        str(sdk / name): hashlib.sha256((sdk / name).read_bytes()).hexdigest()
        for name in headers
    },
    "tilings": tilings,
    "gate_up_resources": gate_resources,
}
(output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
PY
