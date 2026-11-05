#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u

run_dir=${1:-/workspace/experiments/glm52/vllm-baseline-20261017}
mkdir -p "$run_dir"
exec 9> "$run_dir/service.lock"
flock -n 9 || { echo "The baseline service is already running." >&2; exit 1; }
printf '%s\n' "$$" > "$run_dir/service.pid"

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_CUSTOM_OPP_PATH=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer

command=(
    vllm serve /workspace/models/GLM-5.2-w4a8c8
    --host 127.0.0.1 --port 18000 --served-model-name glm-5.2
    --stream-interval 1
    --tensor-parallel-size 16 --data-parallel-size 1
    --dtype bfloat16 --quantization ascend --trust-remote-code
    --attention-config '{"indexer_kv_dtype":"int8"}'
    --kv-cache-dtype auto --block-size 128
    --max-num-seqs 1 --max-num-batched-tokens 8192 --max-model-len 9344
    --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 2147483648
    --no-enable-prefix-caching --enable-chunked-prefill
    --async-scheduling
    --additional-config '{"enable_flashcomm1":false,"enable_dsa_cp":false,"enable_shared_expert_dp":false,"enable_fused_mc2":0}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true,"parallel_drafting":false}'
)
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
exec "${command[@]}" > "$run_dir/service.log" 2>&1
