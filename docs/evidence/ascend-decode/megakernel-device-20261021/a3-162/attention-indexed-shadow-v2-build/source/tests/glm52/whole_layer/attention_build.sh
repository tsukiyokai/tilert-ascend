#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-$here/attention_build}
python_bin=${PYTHON_BIN:-${PYTHON:-python3}}
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}
vendor_root=${OPS_TRANSFORMER_ROOT:?Set OPS_TRANSFORMER_ROOT to the recorded ops-transformer source}
compiler_host=${CXX:-g++}
stem=attention
library=whole_layer_attention
if [[ ${ATTENTION_EXPORT:-0} == 1 ]]; then
    stem=attention_export
    library=whole_layer_export_attention
fi
if [[ ${ATTENTION_FULL:-0} == 1 ]]; then
    stem=attention_full
    library=whole_layer_full_attention
fi
if [[ ${ATTENTION_NORM:-0} == 1 ]]; then
    stem=attention_norm
    library=whole_layer_norm_attention
fi
if [[ ${ATTENTION_INDEXED:-0} == 1 ]]; then
    stem=attention_indexed
    library=whole_layer_indexed_attention
fi
variant_defines=()
if [[ ${ATTENTION_INDEXED_PROFILE:-0} == 1 ]]; then
    if [[ ${ATTENTION_INDEXED:-0} != 1 ]]; then
        echo 'ATTENTION_INDEXED_PROFILE requires ATTENTION_INDEXED=1.' >&2
        exit 2
    fi
    library=whole_layer_indexed_attention_profile
    variant_defines=(-DATTENTION_INDEXED_PROFILE
        -DATTENTION_OP_NAMESPACE=tilert_indexed_profile
        -Dwhole_layer_indexed=whole_layer_indexed_profile
        -Dwhole_layer_indexed_attention=whole_layer_indexed_attention_profile
        -Dwhole_layer_indexed_attention_launch=whole_layer_indexed_attention_profile_launch)
fi
if [[ ${ATTENTION_INDEXED_PARALLEL:-0} == 1 ]]; then
    if [[ ${ATTENTION_INDEXED:-0} != 1 ]]; then
        echo 'ATTENTION_INDEXED_PARALLEL requires ATTENTION_INDEXED=1.' >&2
        exit 2
    fi
    if [[ ${ATTENTION_INDEXED_PROFILE:-0} == 1 ]]; then
        variant_defines+=(-DINDEXER_FRONT_PARALLEL_PROJECTIONS)
    else
        library=whole_layer_indexed_attention_parallel
        variant_defines=(-DINDEXER_FRONT_PARALLEL_PROJECTIONS
            -DATTENTION_OP_NAMESPACE=tilert_indexed_parallel
            -Dwhole_layer_indexed=whole_layer_indexed_parallel
            -Dwhole_layer_indexed_attention=whole_layer_indexed_attention_parallel
            -Dwhole_layer_indexed_attention_launch=whole_layer_indexed_attention_parallel_launch)
    fi
fi
if [[ ${ATTENTION_INDEXED_SHADOW:-0} == 1 ]]; then
    if [[ ${ATTENTION_INDEXED:-0} != 1 || ${ATTENTION_INDEXED_PROFILE:-0} == 1 ||
          ${ATTENTION_INDEXED_PARALLEL:-0} == 1 ]]; then
        echo 'ATTENTION_INDEXED_SHADOW requires indexed attention without other variants.' >&2
        exit 2
    fi
    shadow_overlay=${SFA_PACKED_OVERLAY:?Set SFA_PACKED_OVERLAY to the recorded private SFA headers}
    library=whole_layer_indexed_attention_shadow
    variant_defines=(-DSFA_PACKED_SHADOW_KV
        -DATTENTION_OP_NAMESPACE=tilert_indexed_shadow
        -Dwhole_layer_indexed=whole_layer_indexed_shadow
        -Dwhole_layer_indexed_attention=whole_layer_indexed_attention_shadow
        -Dwhole_layer_indexed_attention_launch=whole_layer_indexed_attention_shadow_launch)
fi
if [[ -z "$cann_root" || ! -d "$cann_root" ]]; then
    echo 'Set CANN_ROOT or source CANN set_env.sh.' >&2
    exit 2
fi
device_compiler=${BISHENG:-}
if [[ -z "$device_compiler" ]]; then
    for candidate in "$cann_root/bin/bisheng" \
                     "$cann_root/compiler/ccec_compiler/bin/bisheng"; do
        if [[ -x "$candidate" ]]; then
            device_compiler=$candidate
            break
        fi
    done
fi
if [[ ! -x "$device_compiler" ]]; then
    echo 'CANN bisheng was not found; set BISHENG explicitly.' >&2
    exit 2
fi
arch_root="$cann_root/$(uname -m)-linux"
if [[ ! -d "$arch_root" ]]; then
    arch_root=$cann_root
fi
includes=(-I "$here" -I "$cann_root/include")
if [[ ${ATTENTION_INDEXED_SHADOW:-0} == 1 ]]; then
    includes+=(-I "$shadow_overlay")
fi
includes+=(-I "$vendor_root")
if [[ ${ATTENTION_NORM:-0} == 1 ]]; then
    norm_vendor=${VLLM_ASCEND_ROOT:?Set VLLM_ASCEND_ROOT to the norm baseline checkout}
    includes+=(-I "$norm_vendor/csrc/moe/add_rms_norm_bias/op_kernel")
fi
for relative in asc asc/include asc/include/adv_api asc/include/basic_api \
                asc/include/c_api asc/include/simt_api asc/include/utils \
                asc/include/basic_api/reg_compute asc/impl/adv_api \
                asc/impl/basic_api asc/impl/c_api asc/impl/simt_api \
                asc/impl/utils asc/impl/basic_api/reg_compute \
                tikcpp/tikcfw tikcpp/tikcfw/lib tikcpp/tikcfw/impl \
                tikcpp/tikcfw/interface; do
    if [[ -d "$arch_root/$relative" ]]; then
        includes+=(-I "$arch_root/$relative")
    fi
done
flags=(-xcce --npu-arch=dav-2201 -std=c++17 -O2 -fPIC)
if [[ ${SANITIZE:-0} == 1 ]]; then
    flags+=(--cce-enable-sanitizer)
fi
mkdir -p "$out"
"$device_compiler" "${flags[@]}" "${variant_defines[@]}" "${includes[@]}" \
    -c "$here/${stem}_kernel.cpp" -o "$out/${stem}_kernel.o"
if [[ ${ATTENTION_KERNEL_ONLY:-0} != 1 ]]; then
    package_info=$($python_bin - <<'PY'
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.find_spec("torch_npu")
if spec is None or spec.origin is None:
    raise SystemExit("The selected Python must provide torch_npu.")
print(Path(torch.__file__).resolve().parent)
print(Path(spec.origin).resolve().parent)
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)
    mapfile -t packages <<< "$package_info"
    torch_root=${packages[0]}
    npu_root=${packages[1]}
    "$compiler_host" -std=c++17 -O2 -fPIC \
        "${variant_defines[@]}" \
        "-D_GLIBCXX_USE_CXX11_ABI=${packages[2]}" \
        "${includes[@]}" -I "$torch_root/include" \
        -I "$torch_root/include/torch/csrc/api/include" \
        -I "$npu_root/include" -c "$here/${stem}_binding.cpp" \
        -o "$out/${stem}_binding.o"
    extra_objects=()
    if [[ ${ATTENTION_FULL:-0} == 1 || ${ATTENTION_INDEXED:-0} == 1 ]]; then
        "$compiler_host" -std=c++17 -O2 -fPIC "${includes[@]}" \
            -c "$here/../mla_tail/mla_tail_tiling.cpp" -o "$out/mla_tail_tiling.o"
        extra_objects+=("$out/mla_tail_tiling.o")
    fi
    if [[ ${ATTENTION_INDEXED:-0} == 1 ]]; then
        "$compiler_host" -std=c++17 -O2 -fPIC "${includes[@]}" \
            -c "$here/indexer_front_tiling.cpp" -o "$out/indexer_front_tiling.o"
        extra_objects+=("$out/indexer_front_tiling.o")
    fi
    "$compiler_host" -shared "$out/${stem}_kernel.o" "$out/${stem}_binding.o" \
        "${extra_objects[@]}" -o "$out/$library.so" \
        -L "$arch_root/lib64" -L "$cann_root/lib64" \
        -L "$torch_root/lib" -L "$npu_root/lib" \
        -lascendcl -lruntime -lunified_dlog -lc_sec \
        -lregister -lprofapi -ltiling_api -lplatform \
        -ltorch -ltorch_cpu -lc10 -ltorch_npu
fi
provenance_args=(--vendor "$vendor_root" --cann "$cann_root" --output "$out/provenance.json")
if [[ ${ATTENTION_NORM:-0} == 1 ]]; then
    provenance_args+=(--norm-vendor "$norm_vendor")
fi
if [[ ${ATTENTION_INDEXED:-0} == 1 ]]; then
    provenance_args+=(--with-indexer)
fi
if [[ ${ATTENTION_INDEXED_PROFILE:-0} == 1 ]]; then
    provenance_args+=(--indexed-profile)
fi
if [[ ${ATTENTION_INDEXED_PARALLEL:-0} == 1 ]]; then
    provenance_args+=(--indexed-parallel)
fi
if [[ ${ATTENTION_INDEXED_SHADOW:-0} == 1 ]]; then
    provenance_args+=(--sfa-overlay "$shadow_overlay")
fi
"$python_bin" "$here/attention_provenance.py" "${provenance_args[@]}"
echo "built attention artifacts in $out"
