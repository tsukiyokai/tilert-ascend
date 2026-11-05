#!/usr/bin/env bash
set -eo pipefail
root=/workspace/experiments/m32-full-teacher-20261020/norm-fusion/compiled-request-c29341d
tree=$root/source
overlay=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer
PORT_BASE=29600 GATE_ATTEMPTS=1 bash "$tree/tests/glm52/chain.sh" \
  "$root/request-01" "$tree" "$overlay" \
  'norm-off-w8:8:--draft-graph,--conditioned-graph,--draft-projection-tp,--max-tokens,33,--warmup,1,--requests,2'
