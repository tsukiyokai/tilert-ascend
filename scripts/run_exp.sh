#!/usr/bin/env bash
set -euo pipefail
TAG=${1:?usage: run_exp.sh TAG [fast|formal|diagnostic] [extra_args...]}
MODE=${2:-fast}
shift
if [ $# -gt 0 ] && { [ "$1" = "fast" ] || [ "$1" = "formal" ] || [ "$1" = "diagnostic" ]; }; then
  shift
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set +u
source /opt/ascend/cann-9.2.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
runtime=/assets/vendor-hook-runtime212-int8
export PYTHONPATH="/work/source-opt-v1:/assets/vllm-unified:/work/runtime-sfa-token-mlatail:$runtime/vendor-hook-pinned/vendor-metadata:$runtime/vendor-hook-deps/transformers519:$runtime/vendor-hook-deps/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-12/root/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
original=/work/runtime-sfa-token-mlatail/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="/opt/kernel-overlay/vendors/custom_transformer:$original${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=ascend,ascend_model HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true
export HCCL_BUFFSIZE=64
export VLLM_ASCEND_ENABLE_CONCRETE_GRAPH=1
export VLLM_ASCEND_ENABLE_SUPER_KERNEL=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_CACHE_PATH=/evidence/$TAG/cache/ascend
export XDG_CACHE_HOME=/evidence/$TAG/cache/xdg
export TORCHINDUCTOR_CACHE_DIR=/evidence/$TAG/cache/inductor
export TRITON_CACHE_DIR=/evidence/$TAG/cache/triton
export VLLM_CACHE_ROOT=/evidence/$TAG/cache/vllm
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_USE_MEGA_AOT_ARTIFACT=0
test ! -e "/evidence/$TAG"
mkdir -p "$ASCEND_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_PORT_RANGE:-42801-42951}
MASTER_PORT=${MASTER_PORT:-29541}
ONESIDED_PORT=${ONESIDED_PORT:-9481}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}

WORK_DIR="/work/run_dirs/$TAG"
mkdir -p "$WORK_DIR"
ln -sfn /work/source-opt-v1/tests "$WORK_DIR/tests"
ln -sfn /work/source-opt-v1/tilert "$WORK_DIR/tilert"
mkdir -p "$WORK_DIR/static_kernel_compile_outputs/static_kernel_cache"
cd "$WORK_DIR"

if [ "$MODE" = diagnostic ]; then
  MAX_TOKENS=${TILERT_MAX_TOKENS:-448}
  WARMUP=${TILERT_WARMUP:-1}
  REQUESTS=${TILERT_REQUESTS:-1}
elif [ "$MODE" = "formal" ]; then
  MAX_TOKENS=1024
  WARMUP=3
  REQUESTS=3
else
  MAX_TOKENS=${TILERT_MAX_TOKENS:-384}
  WARMUP=${TILERT_WARMUP:-1}
  REQUESTS=${TILERT_REQUESTS:-1}
fi

torchrun --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 --master-port="$MASTER_PORT" \
  tests/glm52/bench_cann_mtp.py \
  --checkpoint /work/model-raw \
  --capture-case /assets/checkpoint-owned/prefix8k-v3/case-00 \
  --source-commit "${SOURCE_COMMIT:?Launch from a Git source snapshot}" \
  --max-tokens "$MAX_TOKENS" \
  --capacity 9344 \
  --warmup "$WARMUP" \
  --requests "$REQUESTS" \
  --width 32 \
  --output "/evidence/$TAG" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --target-graph \
  --compile-target \
  --super-kernel \
  --attention-kernel sfa \
  --fp32-head \
  --lookup \
  --expert-parallel \
  --round-phases \
  --hoist-metadata \
  --hoist-forward-context \
  --rope-graph \
  --packed-host-inputs \
  --plain-moe-combine \
  --onesided-reduce "tcp://127.0.0.1:$ONESIDED_PORT" \
  "$@"
