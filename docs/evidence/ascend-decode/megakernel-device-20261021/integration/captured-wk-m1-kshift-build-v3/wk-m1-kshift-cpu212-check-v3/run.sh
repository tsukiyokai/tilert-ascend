#!/usr/bin/env bash
set -euo pipefail
task_root=/home/shanshan/tilert-megakernel-20261021
output=$task_root/logs/wk-m1-kshift-cpu212-v3
mkdir "$output"
docker run --rm --pull=never --name shanshan-tilert-wk-m1-cpu212-v3 \
    --read-only --network none --workdir /tmp --cpus 2 --memory 4g \
    --pids-limit 128 --shm-size 64m --tmpfs /tmp:rw,exec,nosuid,nodev,size=512m \
    -v "$task_root/wk-m1-kshift-v3:/probe:ro" \
    -v "$task_root/wk-m1-kshift-cpu212-check-v3:/check:ro" \
    -v "$task_root/logs/vendor-hook-decoder212-graph-index-int8-m1-v3:/capture:ro" \
    -v "$output:/evidence:rw" \
    -v /home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    --entrypoint /bin/bash \
    sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6 -lc '
        source /opt/ascend/cann-9.2.0/set_env.sh
        env -u ASCEND_DOCKER_RUNTIME PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 python3 /check/check.py
    ' >"$output/run.log" 2>&1
cat "$output/run.log"
