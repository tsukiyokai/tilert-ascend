#!/usr/bin/env bash
set -euo pipefail
base=/dev/shm/codex-glm52-20261111
mkdir -p "$base/source"
if [[ -f "$base/glm52-source-20261111.tgz" ]]; then
    tar -xzf "$base/glm52-source-20261111.tgz" -C "$base/source"
fi
mkdir -p "$base/agy_tp8/cases"
if [[ ! -d "$base/agy_tp8/cases/case-01-coding" ]]; then
    cp -a "$base/source/tests/glm52/megakernel_tp8_ep8/case-01-coding" "$base/agy_tp8/cases/"
fi
cp -a "$base/source/tests/glm52/megakernel_tp8_ep8/scripts" "$base/agy_tp8/"
printf 'case-01-coding\n' > "$base/agy_tp8/current_case.txt"
toolkit=${TILERT_TOOLKIT_ROOT:-/tmp/trt-gmm1-20261029-0030/assets}
devices=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
for die in {0..7}; do devices+=(--device "/dev/davinci$die"); done
docker run -d --name codex-glm52-tp8-20261111 --network host --ipc host \
    --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    "${devices[@]}" \
    -v /dev/shm:/dev/shm \
    -v "$base/agy_tp8:/dev/shm/agy_tp8" \
    -v "$base/source:/work/source-opt-v1" \
    -v /mnt/sfs_turbo_new/trt-sync-tmp:/assets:ro \
    -v "$base/trt-sync-local/vendor-hook-runtime212-int8:/assets/vendor-hook-runtime212-int8:ro" \
    -v "$base/trt-sync-local/vllm-unified:/assets/vllm-unified:ro" \
    -v "$base/trt-sync-local/runtime-sfa-token-mlatail:/work/runtime-sfa-token-mlatail:ro" \
    -v "$base/trt-sync-local/kernel-overlay:/opt/kernel-overlay:ro" \
    -v /mnt/sfs_turbo_new/shanshan/GLM-5.2-w4a8c8:/work/model-raw:ro \
    -v "$toolkit/cann-9.2.0:/opt/ascend/cann-9.2.0:ro" \
    -v "$toolkit/nnal:/usr/local/Ascend/nnal:ro" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    --entrypoint sleep e8c21806f1a6 infinity
