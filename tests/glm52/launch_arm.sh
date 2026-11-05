#!/usr/bin/env bash
# One arm of a whole-machine timing run, with every known trap encoded.
#
# Three driver scripts written in one evening each rebuilt this recipe and each
# dropped a different piece of it, for ten failed launches and about forty
# minutes of machine time. The traps were all written down already; prose is
# the wrong place to keep them.
#
#   TAG      names the arm; its own cache root and port range come from it
#   WIDTH    verify width
#   TREE     the source checkout
#   OUT      where the records go
#   OVERLAY  prepended to ASCEND_CUSTOM_OPP_PATH, which activate.sh resets
#   remaining arguments go to the bench
#
# Exit 2 means the gate refused, so a driver can wait and retry rather than
# consume the arm.
set -uo pipefail
tag=${TAG:?}
width=${WIDTH:?}
tree=$(realpath "${TREE:?}")
out_root=$(realpath -m "${OUT:?}")
out=$out_root/$tag
overlay=${OVERLAY:-}
want_gib=${WANT_GIB:-47}
utilisation=${UTILISATION:-0.78}

# set_env.sh reads $ZSH_VERSION unguarded, so sourcing it under `set -u` exits
# before anything has been created.
set +u
source /workspace/environment/activate.sh
set -u
export PYTHONPATH="$tree${PYTHONPATH:+:$PYTHONPATH}"
[ -n "$overlay" ] && export ASCEND_CUSTOM_OPP_PATH="$overlay:${ASCEND_CUSTOM_OPP_PATH}"
# CANN's own grouped matmul is the fastest of the four implementations on the
# path at the decode shape -- 66.813 us against 97.362, 104.929 and 123.985 --
# and activate.sh puts two custom packages ahead of it for every engine
# process. STOCK_OPP drops them so the comparison can be made end to end.
[ -n "${STOCK_OPP:-}" ] && unset ASCEND_CUSTOM_OPP_PATH

# A cache root per arm, because a shared one replays another arm's compiled
# graph. A port range per arm, not per pass: two arms of one pass cannot share
# it, the second fails to bind behind the first's TIME_WAIT. And the autograd
# cache asserts on OutputCode against torch_npu's _CompiledFxGraph twelve
# minutes in, after the weights have loaded.
export VLLM_CACHE_ROOT=$out_root/cache-$tag
export HCCL_NPU_SOCKET_PORT_RANGE="${PORT_BASE:?}-$((PORT_BASE + 19))"
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

bash "$tree/tests/glm52/precheck.sh" "$want_gib" || {
  echo "=== $tag refused by the gate $(date -Is)"
  exit 2
}
rm -rf "$out" "$VLLM_CACHE_ROOT"
mkdir -p "$out" "$VLLM_CACHE_ROOT"
echo "=== $tag width=$width util=$utilisation opp[0]=${ASCEND_CUSTOM_OPP_PATH%%:*} hccl=$HCCL_NPU_SOCKET_PORT_RANGE $* $(date -Is)"
md5sum "$tree/tilert/models/glm_5_2_ascend/cann_model.py"

# Static kernel packages are cached below cwd, outside VLLM_CACHE_ROOT.
cd "$out"
timeout "${ARM_TIMEOUT:-3000}" torchrun --nproc-per-node 16 \
  --master-port $((51000 + RANDOM % 2000)) \
  "$tree/tests/glm52/bench_cann_mtp.py" \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --source-commit "$tag" \
  --max-tokens 1024 --capacity 9344 --warmup 3 --requests 3 \
  --width "$width" --output "$out" \
  --gpu-memory-utilization "$utilisation" \
  --target-graph --compile-target \
  --super-kernel --attention-kernel sfa --fp32-head "$@" \
  > "$out/launch.log" 2>&1
status=$?
echo "=== $tag exit $status $(date -Is)"
exit $status
