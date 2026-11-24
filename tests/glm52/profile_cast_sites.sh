#!/bin/bash
# Attribute the round's in-graph Cast tasks to their producers.
#
# 477 of them run per round inside the target graph, about six per layer in
# the verify forward, costing 1.43 ms. A replayed task carries neither an API
# event nor a Python frame, so the only way to see the producer is to run
# without the target graph, where the same operations appear as eager aclnn
# calls with their stacks.
#
# Two things this script gets right that the first attempt did not. The profile
# is copied out before the harness's verdict is consulted, because
# --profile-output is otherwise emitted only on a zero exit and the run that
# fails its equivalence check is still the run whose trace you need. And
# ASCEND_CUSTOM_OPP_PATH is inherited rather than exported, since activate.sh
# already places a patched vendor overlay ahead of the in-tree one and setting
# the variable drops it.
set -u
TREE=${TREE:-/workspace/experiments/snapshot-final}
OUT=${OUT:-/workspace/experiments/cast-attrib-$(date +%Y%m%d-%H%M)}
COMMIT=${COMMIT:?set COMMIT to the source commit}
PORT=$((37000 + RANDOM % 400))

cd "$TREE"
export HCCL_OP_EXPANSION_MODE=AIV TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_CACHE_ROOT="$OUT/cache"
mkdir -p "$VLLM_CACHE_ROOT" "$OUT"

timeout 1800 torchrun --nproc-per-node 16 --master-port "$PORT" \
  tests/glm52/run_cann_mtp_rounds.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --source-commit "$COMMIT" --max-tokens 6 --capacity 9344 --ignore-eos \
  --output "$OUT/out" --profile-output "$OUT/prof" --profile-stacks \
  --draft-graph --draft-projection-tp --attention-kernel sfa --fp32-head \
  --width 4 > "$OUT/run.log" 2>&1
echo "harness exit $?"

# The trace matters whether or not the equivalence check passed.
find "$TREE" /tmp -maxdepth 6 -name '*_ascend_pt' -newermt '-30 minutes' 2>/dev/null |
  while read -r dir; do
    cp -a "$dir" "$OUT/prof-$(basename "$dir")" 2>/dev/null && echo "captured $dir"
  done
ls "$OUT" | head
