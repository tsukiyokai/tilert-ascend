#!/usr/bin/env bash
set -euo pipefail
source /opt/ascend/cann-9.2.0/set_env.sh
mkdir -p /dev/shm/codex-glm52-20261111/shmem-capacity
for lib in /assets/shmem-lib/lib/*; do
    if [[ ${lib##*/} != libtp_push_kernel.so ]]; then
        ln -sf "$lib" /dev/shm/codex-glm52-20261111/shmem-capacity/
    fi
done
/opt/ascend/cann-9.2.0/bin/bisheng -xcce -Dtp_push_kernel=tp_push_kernel_capacity_20261111 -Dtp_push_rmsnorm_quant_kernel=tp_push_rmsnorm_quant_kernel_capacity_20261111 --npu-arch=dav-2201 -std=c++17 -O2 -ffp-contract=off -fPIC \
    -Daclshmemx_sync_vec_all=aclshmemx_barrier_all_vec \
    -I /work/source-opt-v1/tests/glm52/shmem \
    -I /opt/ascend/cann-9.2.0/include \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/adv_api \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/basic_api \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/c_api \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/basic_api/reg_compute \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/simt_api \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/utils \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/include \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/include/adv_api \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/include/basic_api \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/include/c_api \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/include/basic_api/reg_compute \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/include/simt_api \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/asc/include/utils \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/tikcpp/tikcfw \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/tikcpp/tikcfw/impl \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/tikcpp/tikcfw/interface \
    -I /opt/ascend/cann-9.2.0/aarch64-linux/tikcpp/tikcfw/lib \
    -I /assets/shmem-lib/include \
    -I /assets/shmem-lib/src \
    -I /assets/shmem-lib/src/device \
    -c /work/source-opt-v1/tests/glm52/shmem/tp_push_kernel.cpp -o /dev/shm/codex-glm52-20261111/tp_push_capacity.o

c++ -shared -Wl,-soname,libtp_push_kernel.so /dev/shm/codex-glm52-20261111/tp_push_capacity.o -o /dev/shm/codex-glm52-20261111/shmem-capacity/libtp_push_kernel.so \
    -L /assets/shmem-lib/lib -lshmem -lshmem_utils \
    -L /opt/ascend/cann-9.2.0/aarch64-linux/lib64 -L /opt/ascend/cann-9.2.0/lib64 \
    -lascendcl -lruntime -lunified_dlog -lc_sec -lregister -lprofapi -ltiling_api -lplatform
