#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo 'Usage: device_run.sh PHYSICAL_DEVICE REPORT_NAME [--mode int8|bf16]' >&2
    exit 2
fi
physical_device=$1
report_name=$2
shift 2
if [[ ! $physical_device =~ ^[0-9]+$ || ! $report_name =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo 'Use an integer device and a simple report name.' >&2
    exit 2
fi
task_root=${VENDOR_HOOK_TASK_ROOT:-/home/shanshan/tilert-megakernel-20261021}
runtime_root=$task_root/vendor-hook-runtime212
decoder_root=$task_root/vendor-hook-decoder212-full-graph-v1
oracle_root=$task_root/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install
cann_root=${VENDOR_HOOK_CANN_ROOT:-/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0}
image=sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6
docker image inspect "$image" >/dev/null
test -f "$runtime_root/frozen-inputs.sha256"
test -f "$decoder_root/frozen-inputs.sha256"
test -f "$oracle_root/packages/vendors/tilert_bridge_oracle/op_api/lib/libcust_opapi.so"
test ! -e "$task_root/logs/$report_name"
test ! -e "$task_root/logs/$report_name-driver"
test ! -e "$task_root/logs/$report_name-extra-info"
mkdir "$task_root/logs/$report_name-extra-info"

docker run --rm --pull=never --name "shanshan-tilert-decoder212-$report_name" \
    --read-only --network none --workdir /tmp \
    --cpus 8 --memory 64g --pids-limit 1024 --shm-size 1g \
    --device "/dev/davinci$physical_device" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=1g \
    --tmpfs /root/ascend/log:rw,nosuid,nodev,size=128m \
    --tmpfs /root/Ascend:rw,nosuid,nodev,size=128m \
    --tmpfs /root/.tvm_test_data:rw,nosuid,nodev,size=128m \
    -v "$runtime_root:/workspace:ro" -v "$decoder_root:/decoder:ro" \
    -v "$oracle_root:/oracle:ro" -v "$task_root/logs:/evidence:rw" \
    -v "$task_root/logs/$report_name-extra-info:/tmp/extra-info:rw" \
    -v "$cann_root:/opt/ascend/cann-9.2.0:ro" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -e "VENDOR_HOOK_RUNTIME_IMAGE=$image" \
    --entrypoint /bin/bash "$image" -lc '
        set -euo pipefail
        (cd /workspace && sha256sum --quiet -c frozen-inputs.sha256)
        (cd /decoder && sha256sum --quiet -c frozen-inputs.sha256)
        source /opt/ascend/cann-9.2.0/set_env.sh
        mkdir "/evidence/$1-driver"
        PYTHONPATH=/decoder/python:/workspace/vendor-hook-deps/transformers519:/workspace/vendor-hook-deps/python:/workspace/vendor-hook-pinned/vllm-a982-py311-compatible:/workspace/vendor-hook-pinned/vendor-metadata:${PYTHONPATH:-} \
        ASCEND_CUSTOM_OPP_PATH=/oracle/packages/vendors/tilert_bridge_oracle:/workspace/vendor-hook-deps/python/fla_npu/opp/vendors/fla_npu_transformer:/workspace/vendor-hook-deps/python/fla_npu/opp${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH} \
        PYTHONDONTWRITEBYTECODE=1 VLLM_PLUGINS=ascend \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        XDG_CACHE_HOME=/tmp/cache TRITON_CACHE_DIR=/tmp/triton \
        ASCEND_PROCESS_LOG_PATH="/evidence/$1-driver" \
        GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        env -u ASCEND_DOCKER_RUNTIME \
        python3 /decoder/device_full_graph.py --runtime /workspace \
            --output "/evidence/$1" --mode int8 --timeout 600 \
            --extension-sha256 acac5e7220b51bdd24e41352ffc52f88aac5f58dd9d262b0a43aad2ca9efd0fc \
            --adapter /workspace/tilert/models/glm_5_2_ascend/attention_native.py \
            --indexed-library /workspace/native/indexed/whole_layer_indexed_attention.so \
            --shared-library /workspace/native/shared/whole_layer_full_attention.so \
            --sfa-library /workspace/native/sfa/vendor_sfa_adapter.so \
            "${@:2}"
    ' decoder212 "$report_name" "$@"
