#!/usr/bin/env bash
set -euo pipefail

task_root=/home/shanshan/tilert-megakernel-20261021
output=$task_root/logs/decoder-extension212-build-v1
cann_root=/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0
image=sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449
test -f "$output/build.py"
test ! -e "$output/build"
docker run --rm --pull=never --name shanshan-tilert-decoder-extension212-build-v1 \
    --read-only --network none --workdir /tmp --cpus 2 --memory 8g \
    --pids-limit 256 --tmpfs /tmp:rw,exec,nosuid,nodev,size=512m \
    --tmpfs /root/ascend/log:rw,nosuid,nodev,size=32m \
    -v "$task_root:/workspace:ro" -v "$output:/evidence:rw" \
    -v "$cann_root:/opt/ascend/cann-9.2.0:ro" \
    --entrypoint /bin/bash -e "AUDIT_IMAGE=$image" "$image" -lc '
        source /opt/ascend/cann-9.2.0/set_env.sh
        python3 /evidence/build.py \
            --vendor /workspace/vendor-hook-pinned/vllm-ascend-8237 \
            --oracle-prefix /workspace/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install/packages/vendors/tilert_bridge_oracle \
            --oracle-manifest /workspace/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/manifest.json \
            --sfa-library /workspace/logs/runtime212-build-v1/sfa/vendor_sfa_adapter.so \
            --cann /opt/ascend/cann-9.2.0 --output /evidence/build \
            --image-id "$AUDIT_IMAGE"
    '
