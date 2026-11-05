#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export HCCL_NPU_SOCKET_PORT_RANGE=53600,53699
export HCCL_HOST_SOCKET_PORT_RANGE=53700,53799
cd /workspace/experiments/reduced-head-20261020
port=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
printf '%s\n' "$port" > master-port.txt
timeout --signal=TERM 180 python -m torch.distributed.run \
    --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 \
    --master-port="$port" probe.py > run.log 2>&1
