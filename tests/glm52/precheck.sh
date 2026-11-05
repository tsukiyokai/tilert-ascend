#!/usr/bin/env bash
# Refuse to launch into another tenant's residue. This machine is shared, and
# an arm that launches with too little free HBM dies ninety seconds in with
# the failure buried in a rank record rather than the log.
want_gib=${1:-48}
# The column pads a short value, so "3138 / 65536" carries a space the older
# pattern required not to be there. Matching only the five-digit form reads an
# otherwise idle machine as holding nothing and passes on an empty string.
used=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+ */ *6[0-9]{4}" |
       cut -d/ -f1 | tr -d " " | sort -rn | head -1)
if [ -z "$used" ]; then
  echo "precheck could not read HBM usage from npu-smi"
  exit 1
fi
total_gib=61.27
used_gib=$(awk "BEGIN {printf \"%.1f\", $used/1024}")
free_gib=$(awk "BEGIN {printf \"%.1f\", $total_gib - $used_gib}")
echo "busiest die holds ${used_gib} GiB, free ${free_gib}, need ${want_gib}"
awk "BEGIN {exit !($free_gib >= $want_gib)}"
