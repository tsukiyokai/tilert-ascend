#!/usr/bin/env bash
# Per-card HBM and the name of whoever holds a box, for a list of A3 hosts.
#
# `kbenv` answers a different question and answered it wrongly for the one
# host that mattered: 7.150.10.78 read 占用 with 0 of 16 dies free while every
# card sat at 4 percent, because an idle `vllm` process name decides the
# verdict before HBM is consulted and a die npu-smi did not report counts as
# taken. Three hosts were abandoned on that reading before this scan found the
# machine that was free all along.
#
# The pool rejects key authentication and then reports a correct password as
# "Permission denied", so password auth is forced rather than offered.
#
#   scripts/free_dies.sh 7.150.10.78 7.150.13.51 ...
#
# A line of eight percentages is the eight cards; low everywhere means the
# dies are yours whatever the process list says. `holder` names the tenant --
# which may be you: a rank wedged by an AI core fault survives SIGTERM and
# holds its memory indefinitely, and the next arm's "Free memory on device"
# then reads like somebody else's machine.
set -uo pipefail
password=${A3_PASSWORD:-Hw123456}
options=(-o StrictHostKeyChecking=no -o ConnectTimeout=8
         -o PreferredAuthentications=password -o PubkeyAuthentication=no)

for host in "$@"; do
    printf '%-15s ' "$host"
    timeout 90 sshpass -p "$password" ssh "${options[@]}" "root@$host" '
      for card in $(seq 0 7); do
          rate=$(npu-smi info -t usages -i $card 2>/dev/null |
                 awk -F: "/HBM Usage Rate/ {gsub(/ /,\"\",\$2); print \$2; exit}")
          printf "%s " "${rate:-x}"
      done
      holder=$(docker ps --format "{{.Names}}" | head -1)
      printf "holder=%s ram=%s\n" "${holder:-none}" \
          "$(free -g | awk "/^Mem:/ {printf \"%d%%\", \$7 * 100 / \$2}")"
    ' 2>/dev/null | tail -1
done
