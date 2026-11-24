#!/usr/bin/env bash
set -euo pipefail

packed_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
packed_out=${1:?Pass a new output directory}
packed_vendor=${OPS_TRANSFORMER_ROOT:?Set OPS_TRANSFORMER_ROOT to the recorded vendor root}
packed_sdk=${CANN_ROOT:-${ASCEND_HOME_PATH:-}}
packed_python=${PYTHON_BIN:-python3}
if [[ -z "$packed_sdk" || ! -x "$packed_sdk/bin/bisheng" ]]; then
    echo 'Source the recorded CANN set_env.sh before compiling.' >&2
    exit 2
fi
if [[ -e "$packed_out" ]]; then
    echo 'Choose a new output directory.' >&2
    exit 2
fi
mkdir -p "$packed_out/source"
packed_out=$(cd -- "$packed_out" && pwd)
cp -p "$packed_here"/*.* "$packed_out/source/"
cp -p "$packed_here/../attention_sfa_tiling.h" "$packed_out/source/"
PYTHONDONTWRITEBYTECODE=1 "$packed_python" "$packed_out/source/prepare_merge.py" \
    --source "$packed_vendor" --output "$packed_out/overlay" \
    > "$packed_out/overlay-manifest.log"
packed_arch="$packed_sdk/$(uname -m)-linux"
if [[ ! -d "$packed_arch" ]]; then
    packed_arch=$packed_sdk
fi
packed_includes=(-I "$packed_out/source" -I "$packed_out/overlay"
                 -I "$packed_sdk/include")
for packed_relative in asc asc/include asc/include/adv_api asc/include/basic_api \
    asc/include/c_api asc/include/simt_api asc/include/utils \
    asc/include/basic_api/reg_compute asc/impl/adv_api asc/impl/basic_api \
    asc/impl/c_api asc/impl/simt_api asc/impl/utils asc/impl/basic_api/reg_compute \
    tikcpp/tikcfw tikcpp/tikcfw/lib tikcpp/tikcfw/impl tikcpp/tikcfw/interface; do
    if [[ -d "$packed_arch/$packed_relative" ]]; then
        packed_includes+=(-I "$packed_arch/$packed_relative")
    fi
done
sha256sum "$packed_out/source"/*.* "$packed_out/overlay"/attention/*/op_kernel/arch22/* \
    > "$packed_out/source-identity.sha256"
"$packed_sdk/bin/bisheng" -xcce --npu-arch=dav-2201 -std=c++17 -O2 -fPIC \
    -DSFA_PACKED_SHADOW_KV "${packed_includes[@]}" \
    -c "$packed_out/source/compile_probe.cpp" -o "$packed_out/compile_probe.o" \
    2>&1 | tee "$packed_out/compile.log"
sha256sum -c "$packed_out/source-identity.sha256" > "$packed_out/source-verification.log"
sha256sum "$packed_out/compile_probe.o" > "$packed_out/artifact-identity.sha256"
"$packed_sdk/bin/bisheng" --version > "$packed_out/compiler-version.txt"
"$packed_python" "$packed_out/source/test_contract.py" \
    > "$packed_out/cpu-contract.log" 2>&1
"$packed_python" "$packed_out/source/test_merge_contract.py" \
    > "$packed_out/cpu-merge-contract.log" 2>&1
