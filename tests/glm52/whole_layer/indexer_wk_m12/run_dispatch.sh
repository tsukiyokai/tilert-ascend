#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 || ! $1 =~ ^[0-9]+$ || ! $2 =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo 'Usage: run_dispatch.sh PHYSICAL_DEVICE REPORT_NAME' >&2
    exit 2
fi
physical_device=$1
report_name=$2
task_root=${VENDOR_HOOK_TASK_ROOT:-/home/shanshan/tilert-megakernel-20261021}
source_root=$task_root/wk-m12-dispatch-v1
capture_root=$task_root/logs/vendor-hook-decoder212-index-int8-m12-v1
cann_root=${VENDOR_HOOK_CANN_ROOT:-/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0}
image=sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6
docker image inspect "$image" >/dev/null
test -d "$cann_root"
test -f "$capture_root/decoder-index-diagnostic.pt"
test ! -e "$task_root/logs/$report_name"
(cd "$source_root" && sha256sum -c frozen-inputs.sha256)
printf '%s  %s\n' fb187d933ca499bec2ecc938184949775025ccbc90e0de894503e1dba2bae0bb \
    "$capture_root/decoder-index-diagnostic.pt" | sha256sum -c -
mkdir "$task_root/logs/$report_name"
sha256sum "$0" "$source_root/dispatch.py" "$source_root/contract.py" \
    "$capture_root/decoder-index-diagnostic.pt" \
    >"$task_root/logs/$report_name/source-sha256.txt"

docker run --rm --pull=never --name "shanshan-tilert-wk-m12-$report_name" \
    --read-only --network none --workdir /tmp \
    --cpus 2 --memory 12g --pids-limit 128 --shm-size 256m \
    --device "/dev/davinci$physical_device" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=1g \
    --tmpfs /root/ascend/log:rw,nosuid,nodev,size=128m \
    --tmpfs /root/Ascend:rw,nosuid,nodev,size=128m \
    --tmpfs /root/.tvm_test_data:rw,nosuid,nodev,size=128m \
    -v "$source_root:/probe:ro" -v "$task_root/logs/$report_name:/evidence:rw" \
    -v "$capture_root:/capture:ro" \
    -v "$cann_root:/opt/ascend/cann-9.2.0:ro" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    --entrypoint /bin/bash "$image" -lc '
        source /opt/ascend/cann-9.2.0/set_env.sh
        PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache \
        ASCEND_PROCESS_LOG_PATH=/evidence/driver-log \
        ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=0 \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        env -u ASCEND_DOCKER_RUNTIME \
        python3 /probe/dispatch.py --capture /capture/decoder-index-diagnostic.pt \
            --output /evidence
    '
