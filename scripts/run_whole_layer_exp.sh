#!/usr/bin/env bash
set -euo pipefail

HOST=${1:?Usage: $0 host variant experiment [fast|formal|diagnostic]}
TAG=${2:?}
EXP=${3:?}
MODE=${4:-fast}
NATIVE_ATTENTION=${NATIVE_ATTENTION:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
BUILD_ONLY=${BUILD_ONLY:-0}
TRACE_ATTENTION=${TRACE_ATTENTION:-0}
TRACE_POSITION=${TILERT_ATTENTION_TRACE_POSITION:-8631}
TRACE_LAYER=${TILERT_ATTENTION_TRACE_LAYER:-0}
TRACE_REQUEST=${TILERT_ATTENTION_TRACE_REQUEST:-0}
MAX_TOKENS_OVERRIDE=${TILERT_MAX_TOKENS:-}
WARMUP_OVERRIDE=${TILERT_WARMUP:-}
REQUESTS_OVERRIDE=${TILERT_REQUESTS:-}
TRT_OVERLAP_SHARED=${TRT_OVERLAP_SHARED:-0}
TRT_FUSED_MC2=${TRT_FUSED_MC2:-0}
TRT_ZERO_COMM_MOE=${TRT_ZERO_COMM_MOE:-0}
PROFILE_LAST=${PROFILE_LAST:-0}
PROFILE_ROUNDS=${PROFILE_ROUNDS:-1}
REPO_ROOT=${TILERT_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
ASSETS=/mnt/sfs_turbo_new/trt-sync-tmp
RUN=$ASSETS/runs/$TAG
cd "$REPO_ROOT"
[[ "$TAG" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]
[[ "$EXP" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]
if [ "${TILERT_LAUNCHER_FROZEN:-0}" != 1 ]; then
  mkdir -p .artifacts/launchers
  launcher=$(mktemp "$REPO_ROOT/.artifacts/launchers/$EXP.XXXXXX.sh")
  cp -- "${BASH_SOURCE[0]}" "$launcher"
  TILERT_PROJECT_ROOT="$REPO_ROOT" TILERT_LAUNCHER_FROZEN=1 exec bash "$launcher" "$@"
fi

if ssh -o BatchMode=yes -o ConnectTimeout=5 "root@$HOST" true 2>/dev/null; then
  rssh() { ssh -o BatchMode=yes "root@$HOST" "$@"; }
  rscp() { scp -q "$@"; }
else
  CREDENTIALS=${TRT_SSH_CREDENTIALS:-${XDG_CONFIG_HOME:-$HOME/.config}/kbenv/passwords}
  PASSWORD=$(awk -F '\t' -v host="$HOST" '$1 == host {print $2; exit}' "$CREDENTIALS")
  test -n "$PASSWORD"
  rssh() { SSHPASS="$PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no "root@$HOST" "$@"; }
  rscp() { SSHPASS="$PASSWORD" sshpass -e scp -q -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no "$@"; }
fi

if [ -f ".artifacts/snapshots/$TAG/manifest.json" ]; then
  python3 scripts/project.py verify ".artifacts/snapshots/$TAG/manifest.json"
else
  python3 scripts/project.py snapshot "$TAG"
fi
SOURCE_COMMIT=$(git rev-parse "refs/snapshots/$TAG")
rssh mkdir -p "$RUN"
if rssh test -f "$RUN/manifest.json"; then
  expected=$(sha256sum ".artifacts/snapshots/$TAG/manifest.json" | cut -d ' ' -f1)
  actual=$(rssh sha256sum "$RUN/manifest.json" | cut -d ' ' -f1)
  [ "$actual" = "$expected" ]
else
  rscp ".artifacts/snapshots/$TAG/source.tar.gz" ".artifacts/snapshots/$TAG/manifest.json" "root@$HOST:$RUN/"
fi
rscp "${BASH_SOURCE[0]}" "root@$HOST:$RUN/launcher-$EXP.sh"
remote_status=0
rssh bash -s -- "$ASSETS" "$TAG" "$EXP" "$MODE" "$SOURCE_COMMIT" "$SKIP_BUILD" "$NATIVE_ATTENTION" "$BUILD_ONLY" "$TRACE_ATTENTION" "$TRACE_POSITION" "$TRACE_LAYER" "$TRACE_REQUEST" "${MAX_TOKENS_OVERRIDE:-0}" "${WARMUP_OVERRIDE:--1}" "${REQUESTS_OVERRIDE:-0}" "$TRT_OVERLAP_SHARED" "$TRT_FUSED_MC2" "$TRT_ZERO_COMM_MOE" "$PROFILE_LAST" "$PROFILE_ROUNDS" <<'REMOTE' || remote_status=$?
set -euo pipefail
assets=$1 tag=$2 experiment=$3 mode=$4 revision=$5 skip_build=$6 native=$7 build_only=$8
trace=$9 trace_position=${10}
trace_layer=${11}
trace_request=${12}
max_tokens_override=${13:-0}
warmup_override=${14:--1}
requests_override=${15:-0}
overlap_shared=${16:-0}
fused_mc2=${17:-0}
zero_comm_moe=${18:-1}
profile_last=${19:-0}
profile_rounds=${20:-1}
run=$assets/runs/$tag
image_tag=$(docker images --format '{{.Repository}}:{{.Tag}}' | awk '/ai-infra-infer-1\.2\.1-arm-openeuler-py311-a3-cann9\.1\.0-pta_v2\.12\.0-ai-infra-op_1\.7\.0/ {print; exit}')
image=$(docker image inspect --format '{{.Id}}' "$image_tag")
if [ ! -d "$run/source" ]; then
  mkdir -p "$run/source"
  tar -xzf "$run/source.tar.gz" -C "$run/source"
fi
mkdir -p "$run/build"
if [ "$native" = 1 ] && [ "$skip_build" != 1 ]; then
  if [ ! -d "$run/vendor" ]; then
    cp -a "$assets/attention_vendor_patched" "$run/vendor"
    cp -a "$run/source/patches/cann-quant-lightning-indexer/." \
      "$run/vendor/attention/quant_lightning_indexer/op_kernel/"
    cp "$run/source/patches/cann-sfa-v3x3/"*.h \
      "$run/vendor/attention/sparse_flash_attention/op_kernel/arch22/"
    tar -czf "$run/build/vendor-source.tar.gz" -C "$run" vendor
  fi
  docker run --rm --network=host --entrypoint /bin/bash \
    -v "$assets":/assets -v "$run":/artifacts \
    -v "$assets/sdk/cann-9.2.0":/opt/ascend/cann-9.2.0 \
    -v "$run/source":/work/source-opt-v1:ro "$image" \
    -lc 'set -euo pipefail; source /opt/ascend/cann-9.2.0/set_env.sh; cd /work/source-opt-v1/tests/glm52/whole_layer; (ATTENTION_INDEXED=1 ATTENTION_HEADS=8 OPS_TRANSFORMER_ROOT=/artifacts/vendor bash attention_build.sh /artifacts/build/indexed) & indexed_pid=$!; (ATTENTION_FULL=1 ATTENTION_HEADS=8 OPS_TRANSFORMER_ROOT=/artifacts/vendor bash attention_build.sh /artifacts/build/full) & full_pid=$!; status=0; wait $indexed_pid || status=$?; wait $full_pid || status=$?; exit $status' \
    2>&1 | tee "$run/build.log"
fi
if [ "$build_only" = 1 ]; then exit 0; fi

common=/dev/shm/trt-sync-local
mkdir -p "$common"
for directory in runtime-sfa-token-mlatail kernel-overlay vllm-unified vendor-hook-runtime212-int8; do
  if [ ! -d "$common/$directory" ]; then cp -a "$assets/$directory" "$common/"; fi
done
occupancy=$(npu-smi info)
mapfile -t hbm < <(printf '%s\n' "$occupancy" | grep -oE '[0-9]+ */ *65536' | cut -d/ -f1 | tr -d ' ')
declare -A occupied=()
while read -r die; do occupied[$die]=1; done < <(
  printf '%s\n' "$occupancy" | awk -F '|' '
    /Process id/ {processes=1; next}
    processes && $3 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
      split($2, ids, " "); print ids[1] * 2 + ids[2]
    }')
[ "${#hbm[@]}" = 16 ]
devices=
for first in 0 8; do
  clear=1
  for ((die=first; die<first+8; die++)); do
    if [ "${hbm[die]}" -ge 5243 ] || [ "${occupied[$die]:-0}" = 1 ]; then clear=0; fi
  done
  if [ "$clear" = 1 ]; then devices=$(seq -s, "$first" "$((first+7))"); break; fi
done
if [ -z "$devices" ]; then
  output=$run/evidence/$experiment
  mkdir -p "$output"
  printf '%s\n' "$occupancy" > "$output/occupancy.txt"
  printf 'No idle eight-die group on %s\n' "$(hostname)" | tee "$output/failure.txt"
  cp "$run/manifest.json" "$output/source.json"
  exit 1
fi
mkdir -p /dev/shm/trt-evidence /dev/shm/trt-run-dirs
[ ! -e "/dev/shm/trt-evidence/$experiment" ]
[ ! -e "$run/evidence/$experiment" ]
metadata=$run/metadata/$experiment
mkdir -p "$metadata"
cp "$run/manifest.json" "$metadata/source.json"
cp "$run/launcher-$experiment.sh" "$metadata/launcher.sh"
if [ -f "$run/build/source.json" ]; then cp "$run/build/source.json" "$metadata/build-source.json"; fi
printf '%s\n' "$occupancy" > "$metadata/occupancy.txt"
sha256sum "$assets"/shmem-lib/lib/*.so > "$metadata/shmem.sha256"
printf '%s\n' "$image" > "$metadata/image.txt"
arguments=()
if [ "$native" = 1 ]; then arguments+=(--native-attention); fi
if [ "$profile_last" = 1 ]; then arguments+=(--profile-last --profile-rounds "$profile_rounds"); fi
trace_env=()
if [ "$trace" = 1 ]; then
  trace_env=(-e "TILERT_ATTENTION_TRACE=/evidence/$experiment/attention.pt"
             -e "TILERT_ATTENTION_TRACE_POSITION=$trace_position"
             -e "TILERT_ATTENTION_TRACE_LAYER=$trace_layer"
             -e "TILERT_ATTENTION_TRACE_REQUEST=$trace_request")
fi
if [ "$mode" = diagnostic ]; then
  trace_env+=(-e SHMEM_LOG_LEVEL=DEBUG -e SHMEM_LOG_TO_STDOUT=1)
fi
if [ "$max_tokens_override" -gt 0 ]; then trace_env+=(-e "TILERT_MAX_TOKENS=$max_tokens_override"); fi
if [ "$warmup_override" -ge 0 ]; then trace_env+=(-e "TILERT_WARMUP=$warmup_override"); fi
if [ "$requests_override" -gt 0 ]; then trace_env+=(-e "TILERT_REQUESTS=$requests_override"); fi
job=$(printf '%s' "$experiment" | cksum | awk '{print $1}')
master_port=$((20000 + job % 10000))
onesided_port=$((10000 + job % 8000))
while [ -n "$(ss -Htan "sport = :$onesided_port")" ]; do
  onesided_port=$((onesided_port + 1))
done
hccl_first=$((44000 + job % 100 * 160))
printf '{"master":%s,"onesided":%s,"hccl_first":%s}\n' \
  "$master_port" "$onesided_port" "$hccl_first" > "$metadata/ports.json"
set +e
docker run --rm --name "trt-$experiment" --privileged --network=host --ipc=host --entrypoint /bin/bash \
  -e "ASCEND_RT_VISIBLE_DEVICES=$devices" -e "SOURCE_COMMIT=$revision" \
  -e "MASTER_PORT=$master_port" -e "ONESIDED_PORT=$onesided_port" \
  -e "HCCL_PORT_RANGE=$hccl_first-$((hccl_first+150))" \
  "${trace_env[@]}" \
  -e TRT_FUSE_AR_NORM=1 \
  -e "TRT_OVERLAP_SHARED=$overlap_shared" \
  -e "TRT_FUSED_MC2=$fused_mc2" \
  -e "TRT_ZERO_COMM_MOE=$zero_comm_moe" \
  -e TILERT_INDEXED_ATTENTION_LIBRARY=/artifacts/build/indexed/whole_layer_indexed_attention_tp8.so \
  -e TILERT_SHARED_ATTENTION_LIBRARY=/artifacts/build/full/whole_layer_full_attention_tp8.so \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /mnt/sfs_turbo_new/shanshan/GLM-5.2-w4a8c8:/work/model-raw \
  -v "$run/source":/work/source-opt-v1:ro -v "$run":/artifacts \
  -v "$common/runtime-sfa-token-mlatail":/work/runtime-sfa-token-mlatail \
  -v "$common/kernel-overlay":/opt/kernel-overlay \
  -v "$assets":/assets -v "$common/vllm-unified":/assets/vllm-unified \
  -v "$common/vendor-hook-runtime212-int8":/assets/vendor-hook-runtime212-int8 \
  -v "$assets/sdk/cann-9.2.0":/opt/ascend/cann-9.2.0 \
  -v "$assets/nnal-src/nnal":/usr/local/Ascend/nnal \
  -v /dev/shm/trt-evidence:/evidence -v /dev/shm/trt-run-dirs:/work/run_dirs \
  "$image" -lc 'exec bash /work/source-opt-v1/scripts/run_exp.sh "$@"' bash \
  "$experiment" "$mode" "${arguments[@]}" 2>&1 | tee "$run/$experiment.log"
status=${PIPESTATUS[0]}
set -e
output=$run/evidence/$experiment
mkdir -p "$output"
if [ -d "/dev/shm/trt-evidence/$experiment" ]; then
  cp -a "/dev/shm/trt-evidence/$experiment/." "$output/"
fi
cp -a "$metadata" "$output/metadata"
cp "$run/$experiment.log" "$output/run.log"
if [ -d "/dev/shm/trt-run-dirs/$experiment/static_kernel_compile_outputs" ]; then
  tar -czf "$output/compiled.tar.gz" -C "/dev/shm/trt-run-dirs/$experiment" static_kernel_compile_outputs
fi
rm -rf "/dev/shm/trt-run-dirs/$experiment" "/dev/shm/trt-evidence/$experiment"
exit "$status"
REMOTE

mkdir -p ".artifacts/builds/$TAG"
if [ "$NATIVE_ATTENTION" = 1 ]; then
  rscp -r "root@$HOST:$RUN/build/." ".artifacts/builds/$TAG/"
  rscp "root@$HOST:$RUN/build.log" ".artifacts/builds/$TAG/build.log"
fi
if [ "$BUILD_ONLY" = 1 ]; then exit "$remote_status"; fi
mkdir -p ".artifacts/incoming/$EXP"
rscp -r "root@$HOST:$RUN/evidence/$EXP/." ".artifacts/incoming/$EXP/"
audit=()
if [ "$MODE" = formal ] && [ -f ".artifacts/incoming/$EXP/summary.json" ]; then
  golden=.artifacts/runs/test-w112-sfa-restored-v7-formal/evidence/rank-0.json
  python3 scripts/check_whole_layer_exp.py ".artifacts/incoming/$EXP" "$golden" \
    --output ".artifacts/incoming/$EXP/audit.json" || remote_status=1
  audit=(--audit ".artifacts/incoming/$EXP/audit.json")
fi
binaries=()
if [ "$NATIVE_ATTENTION" = 1 ]; then
  binaries=(--binary ".artifacts/builds/$TAG/indexed/whole_layer_indexed_attention_tp8.so"
            --binary ".artifacts/builds/$TAG/full/whole_layer_full_attention_tp8.so"
            --artifact ".artifacts/builds/$TAG/vendor-source.tar.gz"
            --artifact ".artifacts/builds/$TAG/build.log"
            --artifact ".artifacts/builds/$TAG/indexed/provenance.json"
            --artifact ".artifacts/builds/$TAG/full/provenance.json")
fi
python3 scripts/project.py register "$EXP" --evidence ".artifacts/incoming/$EXP" \
  --source-ref "refs/snapshots/$TAG" "${audit[@]}" "${binaries[@]}"
exit "$remote_status"
