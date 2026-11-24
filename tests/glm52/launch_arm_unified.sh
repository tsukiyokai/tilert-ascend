#!/usr/bin/env bash
# One whole-machine arm inside the unified image, which carries the engine
# stack and CANN's op compiler itself. It replaces `activate.sh`, which lived
# in the deleted container's writable layer.
#
#   TAG      names the arm; its cache root and port range come from it
#   WIDTH    verify width
#   TREE     the source checkout, mounted read-only
#   OUT      where the records go
#   MODEL    the checkpoint, read-only
#   CAPTURE  the 8K prefix case
#   remaining arguments go to the bench
#
# Exit 2 means the gate refused, so a driver can wait rather than consume it.
set -uo pipefail
tag=${TAG:?}
width=${WIDTH:?}
tree=$(realpath "${TREE:?}")
out_root=$(realpath -m "${OUT:?}")
out=$out_root/$tag
model=${MODEL:?}
capture=${CAPTURE:?}
utilisation=${UTILISATION:-0.78}
attention=${ATTENTION:-sfa}

source /opt/ascend/cann-9.2.0/set_env.sh
# A3's hardware profile reports ATB_EXTENSIONS, so `NPUWorker.__init__` calls
# `_register_atb_extensions()` unconditionally and dies on `libatb.so` when the
# image carries no NNAL. The omniinfer images do not; the ascend vllm image
# does, and its tree is mounted here.
# Its own set_env.sh reads `$ZSH_VERSION` unguarded, so `set -u` aborts the
# source before it exports anything and the failure is still `libatb.so`.
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
# VLLM_TREE and VLLM_ASCEND_TREE override where vllm and `vllm_ascend` come
# from, and they have to move together: `vllm-ascend-unified` reads
# `PassConfig.enable_sp`, which only `vllm-unified` and `vllm-35a49` have.
#
# VLLM_ASCEND_TREE overrides where `vllm_ascend` comes from. The pinned
# `ascend_model` must select its Ascend GLM class: the target loader skips
# the checkpoint's MTP-only `rot.weight`; the MTP class owns `self.rot`.
# `vllm-ascend-unified` is also the source the working `vllm_ascend_C` was
# compiled from. The source trees and plugin metadata must match.
#
# The engine stack comes from the pinned trees under RUNTIME, because the
# omniinfer image ships `omni_npu` rather than `vllm_ascend` and the unified
# image carries only vllm's dist-info -- its version string ends in `.empty`,
# which `importlib.metadata.version` reports happily and `import` does not.
#
# Append, never assign: set_env.sh puts CANN's `tbe` on PYTHONPATH and GE
# imports it, so assigning turns that into `OpCompileProcessor init failed`.
runtime=${RUNTIME:?}
# `vendor-metadata` carries vllm_ascend's egg-info, and the egg-info is the only
# thing that declares `ascend = vllm_ascend:register` under
# `vllm.platform_plugins`. Without it `VLLM_PLUGINS=ascend` selects nothing, no
# platform registers, and the engine dies in `DeviceConfig` on an empty device
# string rather than on anything to do with devices.
ascend_tree=${VLLM_ASCEND_TREE:-$runtime/vendor-hook-pinned/ascend-python-build}
export PYTHONPATH="$tree:${VLLM_TREE:-$runtime/vendor-hook-pinned/vllm-a982-py311-compatible}:$ascend_tree:$runtime/vendor-hook-pinned/vendor-metadata:$runtime/vendor-hook-deps/transformers519:$runtime/vendor-hook-deps/python${PYTHONPATH:+:$PYTHONPATH}"

# `fla_npu` cannot be dropped even though its kernels are not wanted here:
# `vllm_ascend/ops/gdn.py` imports it unconditionally for Qwen3.5. Its
# `libcust_opapi.so` needs GLIBCXX_3.4.29, which neither the image's
# libstdc++ 6.0.28 nor the host's has; the image's gcc-toolset-12 does.
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-12/root/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# `custom_transformer` is not optional. This project had recorded that the
# custom OPP packages need not be rebuilt, because CANN's native grouped
# matmul is the fastest of the four at decode shapes -- but that was about
# speed, not about existence: `aclnnScatterNdUpdateSk` is in this overlay's
# `libcust_opapi.so` and in no `libopapi.so`, so the KV scatter faults without
# it.
# Keep the original vendor present so plugin bootstrap preserves this order.
original_vendor=$ascend_tree/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="/opt/kernel-overlay/vendors/custom_transformer:$original_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
# Select the Ascend platform and model adapters before the first vLLM import.
# The image's omni_npu uses a different vLLM entrypoint layout.
export VLLM_PLUGINS=ascend,ascend_model
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Both belong to an arm's identity rather than to the harness: a deterministic
# reduction fixes the summation order, which moves the logits and so the greedy
# token, and the record this project's best result comes from set neither.
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-true}
export XDG_CACHE_HOME=/tmp/cache TRITON_CACHE_DIR=/tmp/triton
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# A cache root per arm, because a shared one replays another arm's compiled
# graph, and a port range per arm, because two arms of one pass cannot share it.
export VLLM_CACHE_ROOT=$out_root/cache-$tag
export HCCL_NPU_SOCKET_PORT_RANGE="${PORT_BASE:-27000}-$((${PORT_BASE:-27000} + 19))"
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

rm -rf "$out" "$VLLM_CACHE_ROOT"
mkdir -p "$out" "$VLLM_CACHE_ROOT"
echo "=== $tag width=$width util=$utilisation hccl=$HCCL_NPU_SOCKET_PORT_RANGE $* $(date -Is)"

# Static kernel packages are cached below the working directory, which is the
# arm's own by default. A series of arms that differ only in flags can share
# one by setting KERNELS: the package key covers the dumped operator
# parameters and the super-kernel option string, so what differs recompiles
# and the rest is reused. It does not cover the vendor source, so an arm that
# patches a vendor call must keep its own.
# `grep -c "static kernel compile start" launch.log` says which happened.
kernels=$(realpath -m "${KERNELS:-$out}")
mkdir -p "$kernels"
cd "$kernels"
timeout "${ARM_TIMEOUT:-14400}" torchrun --nproc-per-node 16 \
  --master-port $((51000 + RANDOM % 2000)) \
  "$tree/tests/glm52/bench_cann_mtp.py" \
  --checkpoint "$model" \
  --capture-case "$capture" \
  --source-commit "$tag" \
  --max-tokens 1024 --capacity 9344 --warmup 3 --requests 3 \
  --width "$width" --output "$out" \
  --gpu-memory-utilization "$utilisation" \
  --target-graph --compile-target \
  --super-kernel --attention-kernel "$attention" --fp32-head "$@" \
  > "$out/launch.log" 2>&1
status=$?
echo "=== $tag exit $status $(date -Is)"
exit $status
