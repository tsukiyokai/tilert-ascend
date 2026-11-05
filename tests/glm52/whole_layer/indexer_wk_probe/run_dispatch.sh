#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo 'Usage: run_dispatch.sh PHYSICAL_DEVICE REPORT_NAME {full-bf16|full-fp32|split-bf16}' >&2
    exit 2
fi
physical_device=$1
report_name=$2
probe_kind=$3
if [[ ! $physical_device =~ ^[0-9]+$ || ! $report_name =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo 'Use an integer device and a simple report name.' >&2
    exit 2
fi
if [[ $probe_kind != full-bf16 && $probe_kind != full-fp32 && $probe_kind != split-bf16 ]]; then
    echo 'Unsupported dispatch probe kind.' >&2
    exit 2
fi
task_root=${VENDOR_HOOK_TASK_ROOT:-/home/shanshan/tilert-megakernel-20261021}
runtime_root=$task_root/vendor-hook-runtime
cann_root=${VENDOR_HOOK_CANN_ROOT:-/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0}
capture_root=$task_root/logs/vendor-hook-index-diagnostic-bf16-v13
image=sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6
docker image inspect "$image" >/dev/null
test -f "$runtime_root/tests/glm52/whole_layer/indexer_wk_probe/dispatch.py"
test -f "$runtime_root/tests/glm52/whole_layer/indexer_wk_probe/check.py"
test -f "$capture_root/outer-index-diagnostic.pt"
test -d "$cann_root"
test ! -e "$task_root/logs/$report_name"
mkdir "$task_root/logs/$report_name"
sha256sum "$0" "$runtime_root/tests/glm52/whole_layer/indexer_wk_probe/dispatch.py" \
    "$runtime_root/tests/glm52/whole_layer/indexer_wk_probe/check.py" \
    "$capture_root/outer-index-diagnostic.pt" \
    >"$task_root/logs/$report_name/source-sha256.txt"

docker run --rm --pull=never --name "shanshan-tilert-wk-probe-$report_name" \
    --read-only --network none --workdir /tmp \
    --cpus 2 --memory 12g --pids-limit 128 --shm-size 256m \
    --device "/dev/davinci$physical_device" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=1g \
    --tmpfs /root/ascend/log:rw,nosuid,nodev,size=128m \
    --tmpfs /root/Ascend:rw,nosuid,nodev,size=128m \
    --tmpfs /root/.tvm_test_data:rw,nosuid,nodev,size=128m \
    -v "$runtime_root:/workspace:ro" -v "$task_root/logs/$report_name:/evidence:rw" \
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
        python3 /workspace/tests/glm52/whole_layer/indexer_wk_probe/dispatch.py \
            --capture /capture/outer-index-diagnostic.pt --output /evidence --kind "$1"
    ' wk-dispatch "$probe_kind"
