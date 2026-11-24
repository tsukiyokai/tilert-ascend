#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
root=/workspace/experiments/gmm2-compact-20261020
build="$root/$1"
probe=$2
label=$3
variant=GroupedMatmul_a0be48d01b162435398ed52e1b6182d4
kernel="$build/overlay/vendors/tilert_gmm2/op_impl/ai_core/tbe/kernel/ascend910_93/grouped_matmul/$variant.o"
export ASCEND_CUSTOM_OPP_PATH="$build/overlay/vendors/tilert_gmm2:${ASCEND_CUSTOM_OPP_PATH:-}"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
cd "$root"
exec python3 /workspace/experiments/gmm-compact-20261020/watch_kernel.py \
    "$kernel" "$root/$label-access.json" \
    python3 "$root/$probe.py" --output "$root/$label.json"
