set -eo pipefail
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ACL_OP_INIT_MODE=1 OMP_NUM_THREADS=1
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
mkdir -p /workspace/experiments/glm52/prefix-8k-20261017
exec 9> /workspace/experiments/glm52/prefix-8k-20261017/receiver.lock
flock -n 9
printf '%s\n' "$$" > /workspace/experiments/glm52/prefix-8k-20261017/receiver.pid
exec python -m torch.distributed.run --standalone --nproc-per-node=16 tests/glm52/capture_pd_prefix.py --checkpoint /workspace/models/GLM-5.2-w4a8c8 --capacity 8192 --hixl-host 7.150.10.55 --hixl-port-base 27100 --hixl-library /workspace/experiments/hixl-build/build-sync-contract/libtilert_hixl.so --ctrl-port 5557 --http-port 18081 --timeout 1800 --output /workspace/experiments/glm52/prefix-8k-20261017
