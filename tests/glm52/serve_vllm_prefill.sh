#!/usr/bin/env bash
set -eo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source /workspace/environment/activate.sh
set -u

: "${TRT_HIXL_IP:?Set TRT_HIXL_IP to the prefill business IP}"
decode_host=${TRT_DECODE_HOST:-$TRT_HIXL_IP}
ctrl_port=${TRT_CTRL_PORT:-5557}
transfer_config=$(python3 -S - "$TRT_HIXL_IP" "$decode_host" "$ctrl_port" \
    "${TRT_MTP_PARENT_HIDDEN:-0}" \
    "${TRT_HIXL_LIBRARY:-/workspace/experiments/hixl-build/build-sync-contract/libtilert_hixl.so}" <<'PY'
import json
import sys

if sys.argv[4] not in ("0", "1"):
    raise ValueError("TRT_MTP_PARENT_HIDDEN must be 0 or 1")

print(json.dumps({
    "kv_connector": "AscendTileRTConnector",
    "kv_connector_module_path": "tilert.pd_vllm.ascend_connector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "tilert_host": sys.argv[2],
        "tilert_ctrl_port": int(sys.argv[3]),
        "tilert_max_seq_len": 8192,
        "hixl_host": sys.argv[1],
        "hixl_port_base": 26000,
        "hixl_library": sys.argv[5],
        "tilert_timeout_s": 1800,
        "transfer_mtp_parent_hidden": sys.argv[4] == "1",
    },
}))
PY
)

cd "$repo_dir"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

run_dir=${1:-/workspace/experiments/glm52/vllm-prefill-20261017}
mkdir -p "$run_dir"
exec 9> "$run_dir/service.lock"
flock -n 9 || { echo "The prefill service is already running." >&2; exit 1; }
printf '%s\n' "$$" > "$run_dir/service.pid"

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_CUSTOM_OPP_PATH=/workspace/offline/cann-workspace-owner-20261017/kernel-overlay/vendors/custom_transformer:/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer

command=(
    vllm serve /workspace/models/GLM-5.2-w4a8c8
    --host 127.0.0.1 --port 18000 --served-model-name glm-5.2
    --tensor-parallel-size 16 --pipeline-parallel-size 1 --data-parallel-size 1
    --dtype bfloat16 --quantization ascend --trust-remote-code
    --attention-config '{"indexer_kv_dtype":"int8"}'
    --kv-cache-dtype auto --block-size 128
    --max-num-seqs 1 --max-num-batched-tokens 8192 --max-model-len 9344
    --gpu-memory-utilization 0.4 --kv-cache-memory-bytes 1073741824
    --no-enable-prefix-caching --enable-chunked-prefill
    --enforce-eager --no-async-scheduling --return-tokens-as-token-ids
    --additional-config '{"enable_flashcomm1":false,"enable_dsa_cp":false,"enable_shared_expert_dp":false,"enable_fused_mc2":0}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true,"parallel_drafting":false}'
    --kv-transfer-config "$transfer_config"
)
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
exec "${command[@]}" > "$run_dir/service.log" 2>&1
