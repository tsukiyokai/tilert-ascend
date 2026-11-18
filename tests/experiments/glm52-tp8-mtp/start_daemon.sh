#!/usr/bin/env bash
set -eo pipefail
source /assets/env_a3.sh
export TILERT_SHMEM_LIB_DIR=${TILERT_SHMEM_LIB_DIR:-/assets/shmem-lib/lib}
export LD_LIBRARY_PATH="$TILERT_SHMEM_LIB_DIR:$LD_LIBRARY_PATH"
export TILERT_JOB_DIR=/dev/shm/agy_tp8/control
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_NPU_SOCKET_PORT_RANGE=43201-43351
export TILERT_INDEXED_ATTENTION_LIBRARY=/assets/runs/v3x58/build/indexed/whole_layer_indexed_attention_tp8.so
export TILERT_SHARED_ATTENTION_LIBRARY=/assets/runs/v3x58/build/full/whole_layer_full_attention_tp8.so
exec torchrun --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 \
    --master-port=29899 /dev/shm/agy_tp8/scripts/daemon_8rank.py
