#!/usr/bin/env bash
# One arm of the positions A/B. Three things about this environment cost a
# launch each, so they are all explicit here:
#
#   activate.sh unsets PYTHONPATH and exports the vendor search path whose
#   first entry is the barrier-patched owner overlay, so anything set before
#   it is discarded -- the first attempt ran the unpatched in-tree object and
#   read three token streams from its own baseline.
#
#   set_env.sh reads $ZSH_VERSION unguarded, so sourcing under `set -u` exits
#   immediately and both arms finish in zero seconds reporting nothing.
#
#   activate.sh leaves another checkout of this project on PYTHONPATH, and
#   that checkout wins `import tilert` even when this tree is ahead of it on
#   the path. Running from inside the tree is what resolves it, which the
#   cache-off fix makes free: the static-kernel cache that used to make the
#   working directory matter is no longer consulted.
set -uo pipefail
T=/workspace/experiments/phases-20261019
TREE=/workspace/experiments/positions-20261019/tree
E=/workspace/experiments/w4a8-race-20261019
arm=$1
out=$T/$arm

rm -rf "$out" "$T/cache-$arm"
mkdir -p "$out" "$T/cache-$arm"

set +u
source /workspace/environment/activate.sh
set -u
export FIX_ARM=cache-off
export PYTHONPATH=$TREE/tests/glm52/phasehook:$TREE:${PYTHONPATH:-}
# Only the first sitecustomize on the path is imported, so the cache fix
# has to be named rather than placed.
export PHASE_HOOK_CHAIN=$E/fixhook/sitecustomize.py
export VLLM_CACHE_ROOT=$T/cache-$arm
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export HCCL_OP_EXPANSION_MODE=AIV

cd "$TREE"
origin=$(python3 -c 'import tilert; print(tilert.__file__)')
case "$origin" in
  "$TREE"/*) ;;
  *) echo "Error: tilert resolves to $origin, not $TREE"; exit 1 ;;
esac

echo "=== $arm opp[0]=${ASCEND_CUSTOM_OPP_PATH%%:*} $(date -Is)"
timeout 3000 torchrun --nproc-per-node 16 --master-port $((54100 + RANDOM % 400)) \
  tests/glm52/bench_cann_mtp.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --source-commit "phases-$arm" \
  --max-tokens 1024 --capacity 9344 --warmup 3 --requests 3 \
  --width 8 --output "$out" \
  --target-graph --draft-graph --draft-projection-tp --compile-target \
  --super-kernel --attention-kernel sfa --fp32-head \
  > "$out/launch.log" 2>&1
echo "=== $arm exit $? $(date -Is)"
python3 /tmp/which_vendor.py "$out"
python3 /tmp/read_arm.py "$out"
