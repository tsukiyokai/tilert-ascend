#!/usr/bin/env bash
set -euo pipefail

task_root=/home/shanshan/tilert-megakernel-20261021
audit_root=$task_root/logs/decoder-extension212-packed-mapping-v1
cann_root=/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0
image=sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449
test -f "$audit_root/cpu_packed_mapping.py"
test ! -e "$audit_root/report.json"
docker run --rm --pull=never --name shanshan-tilert-decoder-extension212-packed-mapping-v1 \
    --read-only --network none --workdir /tmp --cpus 2 --memory 8g \
    --pids-limit 256 --shm-size 128m \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=512m \
    --tmpfs /root/ascend/log:rw,nosuid,nodev,size=32m \
    -v "$task_root:/workspace:ro" -v "$audit_root:/evidence:rw" \
    -v "$cann_root:/opt/ascend/cann-9.2.0:ro" \
    --entrypoint /bin/bash -e "AUDIT_IMAGE=$image" "$image" -lc '
        source /opt/ascend/cann-9.2.0/set_env.sh
        PYTHONPATH=/evidence/overlay:/workspace/vendor-hook-deps/transformers519:/workspace/vendor-hook-deps/python:/workspace/logs/whole-layer-composition-cpu212-v1/vllm-a982-py311-compatible:/workspace/vendor-hook-pinned/vendor-metadata:/workspace/tests/glm52/whole_layer:$PYTHONPATH \
        ASCEND_CUSTOM_OPP_PATH=/workspace/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install/packages/vendors/tilert_bridge_oracle \
        TRITON_ASCEND_ARCH=Ascend910_9392 \
        PYTHONDONTWRITEBYTECODE=1 VLLM_PLUGINS=ascend \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        XDG_CACHE_HOME=/tmp/cache TRITON_CACHE_DIR=/tmp/triton \
        ASCEND_PROCESS_LOG_PATH=/tmp/ascend-log \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python3 /evidence/cpu_packed_mapping.py --output /evidence/report.json
    '
