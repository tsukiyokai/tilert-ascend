#!/usr/bin/env bash
# Run a list of arms in one session, gated, with a port range per arm.
#
# Replaces chain_alternating.sh, which hard-coded the directory it looked for
# records in, so any experiment but one checked the wrong place. And replaces
# the per-experiment drivers that each rebuilt the launch recipe and each
# dropped a different piece of it.
#
#   chain.sh <out> <tree> <overlay> <arm>...
#
# An arm is "tag:width:flags", flags comma-separated:
#
#   chain.sh $E $E/tree $OVERLAY pageable-1:8: pinned-1:8: lookup:32:--lookup
#
# Alternate by listing the arms alternately; pairing needs an A beside the B
# that ran under the same conditions, not after it.
set -uo pipefail
out=$1; tree=$2; overlay=$3; shift 3
port=${PORT_BASE:-17600}

for arm in "$@"; do
  IFS=: read -r tag width flags <<<"$arm"
  for attempt in $(seq 1 "${GATE_ATTEMPTS:-40}"); do
    TAG=$tag WIDTH=$width TREE=$tree OUT=$out OVERLAY=$overlay \
      PORT_BASE=$port \
      bash "$tree/tests/glm52/launch_arm.sh" ${flags//,/ }
    status=$?
    port=$((port + 20))
    [ "$status" = 2 ] || break
    echo "=== $tag gated out, attempt $attempt $(date -Is)"
    sleep "${GATE_WAIT:-180}"
  done
  if [ "$status" != 0 ]; then
    exit "$status"
  fi
  if [ -f "$out/$tag/rank-0.json" ]; then
    python3 -I "$tree/tests/glm52/round_clock.py" "$out/$tag"
  else
    echo "=== $tag produced no record"
  fi
done
echo "=== chain done $(date -Is)"
