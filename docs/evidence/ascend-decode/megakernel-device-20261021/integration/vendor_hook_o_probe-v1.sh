#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo 'Usage: vendor_hook_o_probe.sh PHYSICAL_DEVICE REPORT_NAME' >&2
    exit 2
fi
physical_device=$1
report_name=$2
if [[ ! $physical_device =~ ^[0-9]+$ || ! $report_name =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo 'Use an integer device and a simple report name.' >&2
    exit 2
fi
task_root=${VENDOR_HOOK_TASK_ROOT:-/home/shanshan/tilert-megakernel-20261021}
runtime_root=$task_root/vendor-hook-runtime
cann_root=${VENDOR_HOOK_CANN_ROOT:-/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0}
capture_root=$task_root/logs/vendor-hook-forward-diagnostic-v10
image=sha256:9afbed90a0f92e228cc1ba13adf3cee2e89e34f1dd4cad1ed8eb5ce8c091b4af
docker image inspect "$image" >/dev/null
test -f "$runtime_root/tests/glm52/whole_layer/vendor_hook_o_probe.py"
test -f "$capture_root/outer-forward-diagnostic.pt"
test -d "$cann_root"
test ! -e "$task_root/logs/$report_name"
mkdir "$task_root/logs/$report_name"
sha256sum "$0" "$runtime_root/tests/glm52/whole_layer/vendor_hook_o_probe.py" \
    "$capture_root/outer-forward-diagnostic.pt" \
    >"$task_root/logs/$report_name/source-sha256.txt"

docker run --rm --name "shanshan-tilert-o-probe-$report_name" \
    --read-only --network none --workdir /tmp \
    --cpus 2 --memory 4g --pids-limit 128 --shm-size 256m \
    --device "/dev/davinci$physical_device" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=1g \
    --tmpfs /root/ascend/log:rw,nosuid,nodev,size=128m \
    --tmpfs /root/Ascend:rw,nosuid,nodev,size=128m \
    -v "$runtime_root:/workspace:ro" -v "$task_root/logs/$report_name:/evidence:rw" \
    -v "$capture_root:/capture:ro" \
    -v "$cann_root:/opt/ascend/cann-9.2.0:ro" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    --entrypoint /bin/bash "$image" -lc '
        source /opt/ascend/cann-9.2.0/set_env.sh
        PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache \
        ASCEND_PROCESS_LOG_PATH=/evidence/driver-log \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        env -u ASCEND_DOCKER_RUNTIME \
        python3 /workspace/tests/glm52/whole_layer/vendor_hook_o_probe.py \
            --capture /capture/outer-forward-diagnostic.pt --output /evidence
    '
