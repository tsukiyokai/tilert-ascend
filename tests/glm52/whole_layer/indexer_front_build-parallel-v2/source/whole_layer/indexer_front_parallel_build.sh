#!/usr/bin/env bash
set -euo pipefail

front_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
front_out=${1:-$front_here/indexer_front_build-parallel}
front_python=${PYTHON_BIN:-${PYTHON:-python3}}
if [[ -e "$front_out/whole_layer_indexer_front.so" ]]; then
    echo 'The output library already exists; choose a new build directory.' >&2
    exit 2
fi
mkdir -p "$front_out/source"/{whole_layer,mla_tail}
front_out=$(cd -- "$front_out" && pwd)
for front_file in "$front_here"/indexer_front*.*; do
    cp -p "$front_file" "$front_out/source/whole_layer/"
done
for front_file in attention_host.h attention_contract.h; do
    cp -p "$front_here/$front_file" "$front_out/source/whole_layer/"
done
for front_file in mla_tail.h mla_tail_native.h mla_tail_dequant.h; do
    cp -p "$front_here/../mla_tail/$front_file" "$front_out/source/mla_tail/"
done
sha256sum "$front_out/source"/{whole_layer,mla_tail}/* \
    > "$front_out/source-identity.sha256"
INDEXER_FRONT_PARALLEL=1 bash \
    "$front_out/source/whole_layer/indexer_front_build.sh" "$front_out" \
    2>&1 | tee "$front_out/build.log"
${CXX:-g++} -std=c++17 -O2 -Wall -Wextra -Werror \
    "$front_out/source/whole_layer/indexer_front_schedule_test.cpp" \
    -o "$front_out/schedule_test"
"$front_out/schedule_test" > "$front_out/schedule-test.log"
sha256sum -c "$front_out/source-identity.sha256" \
    > "$front_out/source-verification.log"
sha256sum "$front_out"/*.o "$front_out/whole_layer_indexer_front.so" \
    > "$front_out/artifact-identity.sha256"
"$front_python" - "$front_out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
manifest = output / "build-manifest.json"
data = json.loads(manifest.read_text())
data.update({
    "variant": "parallel independent indexer Q and WK projection tasks",
    "namespace": "tilert_front_parallel",
    "schedule_macro": "INDEXER_FRONT_PARALLEL_PROJECTIONS",
    "compiled_source_snapshot": str(output / "source"),
    "compiled_source_sha256": {
        str(path.relative_to(output / "source")):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((output / "source").glob("*/*"))
    },
    "parallel_groups_minimum": 21,
    "query_tiles": 16,
    "key_tiles": 5,
    "arithmetics": "unchanged per-tile GEMM and epilogues",
})
manifest.write_text(json.dumps(data, indent=2) + "\n")
PY
