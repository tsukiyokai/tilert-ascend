export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -e
source /opt/ascend/cann-9.2.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
runtime=/assets/vendor-hook-runtime212-int8
export PYTHONPATH="/work/source-layer0-v1:/assets/vllm-unified:/work/runtime-sfa-token:$runtime/vendor-hook-pinned/vendor-metadata:$runtime/vendor-hook-deps/transformers519:$runtime/vendor-hook-deps/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-12/root/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
original=/work/runtime-sfa-token/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="/opt/kernel-overlay/vendors/custom_transformer:$original${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=ascend,ascend_model HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true
export ASCEND_CACHE_PATH=/evidence/layer0-boundaries-v2-cache/ascend XDG_CACHE_HOME=/evidence/layer0-boundaries-v2-cache/xdg TORCHINDUCTOR_CACHE_DIR=/evidence/layer0-boundaries-v2-cache/inductor TRITON_CACHE_DIR=/evidence/layer0-boundaries-v2-cache/triton VLLM_CACHE_ROOT=/evidence/layer0-boundaries-v2-cache/vllm
mkdir -p "$ASCEND_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
export HCCL_NPU_SOCKET_PORT_RANGE=42601-42751
cd /work/source-layer0-v1
set +e
timeout 900 torchrun --nnodes=1 --nproc-per-node=16 --master-addr=127.0.0.1 --master-port=42501 --no-python /work/receipts/layer0-boundaries-v3-rank.sh --checkpoint /models/glm52 --capture-case /assets/checkpoint-owned/prefix8k-v3/case-00 --teacher-json /work/receipts/teacher-32.json --prompt-json /work/receipts/prompt-8192.json --output /evidence/layer0-boundaries-v3 --layers 1 --steps 1 --query-width 1 --baseline /work/inputs/layer0-baseline-v3 --fp32-head > /work/receipts/layer0-boundaries-v3-driver.log 2>&1
status=$?
echo "$status" > /work/receipts/layer0-boundaries-v3-exit.txt
exit "$status"
