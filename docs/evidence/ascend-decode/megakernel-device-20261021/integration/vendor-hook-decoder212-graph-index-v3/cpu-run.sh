#!/usr/bin/env bash
set -euo pipefail
task_root=/home/shanshan/tilert-megakernel-20261021
result_root=$task_root/logs/decoder-graph-index-cpu-v3
mkdir "$result_root"
docker run --rm --pull=never --name shanshan-tilert-decoder-graph-index-cpu-v3 \
 --read-only --network none --workdir /tmp --cpus 2 --memory 4g --pids-limit 128 \
 --tmpfs /tmp:rw,nosuid,nodev,size=256m \
 -v "$task_root/vendor-hook-runtime212:/workspace:ro" \
 -v "$task_root/vendor-hook-decoder212-graph-index-v3:/decoder:ro" \
 -v "$task_root/logs:/evidence:ro" -v "$result_root:/result:rw" \
 -v /home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro \
 --entrypoint /bin/bash sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6 -lc '
 set -euo pipefail
 source /opt/ascend/cann-9.2.0/set_env.sh
 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 python3 /decoder/graph_index_cpu_check.py --decoder /decoder --runtime /workspace \
 --indexed /evidence/attention-wk-m12-build-v1/whole_layer_indexed_attention.so \
 --shared /workspace/native/shared/whole_layer_full_attention.so \
 --failure /evidence/vendor-hook-decoder212-graph-int8-wide-v1/decoder-graph-int8-m1-provided-r2-native.pt \
 --failure-report /evidence/vendor-hook-decoder212-graph-int8-wide-v1/decoder-graph.json \
 --output /result/report.json
 ' > "$result_root/run.log" 2>&1
cat "$result_root/run.log"
