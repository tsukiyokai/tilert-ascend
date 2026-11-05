#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
root=/workspace/experiments/gmm-compact-20261020/kloop-audit
overlay=$root/paired-build/overlay/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$overlay:${ASCEND_CUSTOM_OPP_PATH:-}"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset ASCEND_PROCESS_LOG_PATH
cd "$root"
python ../watch_kernel.py "$overlay/op_impl/ai_core/tbe/kernel/ascend910_93/grouped_matmul_swiglu_quant_v2/GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.o" "$root/bench-access.json" python "$root/bench_full_a.py" --device npu:15 --output "$root/bench.json" > "$root/bench.log" 2>&1
