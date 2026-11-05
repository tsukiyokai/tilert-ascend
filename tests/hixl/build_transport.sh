#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 CANN_SDK_DIRECTORY OUTPUT_DIRECTORY" >&2
    exit 2
fi

sdk_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
compiler=${HIXL_CXX:-g++}
source_file="$repo_dir/tilert/pd_vllm/hixl_transport.cpp"
runtime_lib=/opt/ascend/cann-9.2.0/lib64
mkdir -p "$output_dir"

"$compiler" -std=c++17 -O2 -fPIC -fvisibility=hidden -shared \
    -Wall -Wextra -Werror -pthread \
    -I"$sdk_dir/include" "$source_file" \
    -L"$sdk_dir/lib64" -Wl,-rpath-link,"$sdk_dir/lib64" \
    -Wl,-rpath,"$runtime_lib" -Wl,-z,defs \
    -lcann_hixl -lmetadef -lacl_rt \
    -o "$output_dir/libtilert_hixl.so"

sha256sum "$source_file" "$sdk_dir/include/hixl/hixl.h" \
    "$sdk_dir/include/hixl/hixl_types.h" "$sdk_dir/lib64/libcann_hixl.so" \
    "$output_dir/libtilert_hixl.so" > "$output_dir/build.sha256"
"$compiler" --version > "$output_dir/compiler.txt"
readelf -d "$output_dir/libtilert_hixl.so" > "$output_dir/dynamic.txt"
readelf -Ws "$output_dir/libtilert_hixl.so" > "$output_dir/symbols.txt"
echo "$output_dir/libtilert_hixl.so"
