export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -e
source /opt/ascend/cann-9.2.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
runtime=/assets/vendor-hook-runtime212-int8
export PYTHONPATH="/work/source-v12:/assets/vllm-unified:/work/runtime-sfa-token:$runtime/vendor-hook-pinned/vendor-metadata:$runtime/vendor-hook-deps/transformers519:$runtime/vendor-hook-deps/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-12/root/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
original=/work/runtime-sfa-token/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="/opt/kernel-overlay/vendors/custom_transformer:$original${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=ascend,ascend_model HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true
export ASCEND_CACHE_PATH=/evidence/full-fixed-v3-cache/ascend XDG_CACHE_HOME=/evidence/full-fixed-v3-cache/xdg TORCHINDUCTOR_CACHE_DIR=/evidence/full-fixed-v3-cache/inductor TRITON_CACHE_DIR=/evidence/full-fixed-v3-cache/triton VLLM_CACHE_ROOT=/evidence/full-fixed-v3-cache/vllm
mkdir -p "$ASCEND_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
export HCCL_NPU_SOCKET_PORT_RANGE=42601-42751
cd /work/source-v12
set +e
timeout 3600 torchrun --nnodes=1 --nproc-per-node=16 --master-addr=127.0.0.1 --master-port=42501 experiments/cann-precision/run.py --checkpoint /models/glm52 --capture-case /assets/checkpoint-owned/prefix8k-v3/case-00 --teacher /work/receipts/teacher-32.json --reference /work/hub/trt-a3-reference-20261028-2205/complete --source-commit 3fcfd8b4904bc609888150d83d4b9137f4bcae63 --capacity 9344 --gpu-memory-utilization 0.85 --output /evidence/full-fixed-v3 > /work/receipts/full-fixed-v3-driver.log 2>&1
status=$?
echo "$status" > /work/receipts/full-fixed-v3-exit.txt
exit "$status"
