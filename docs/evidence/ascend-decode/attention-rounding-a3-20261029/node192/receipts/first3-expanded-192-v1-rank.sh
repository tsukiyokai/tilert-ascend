#!/bin/bash
set -e
exec /work/tools/strace/usr/bin/strace --seccomp-bpf -f -e trace=openat -o "/work/receipts/first3-expanded-192-v1-rank-${RANK}-strace.log" python -B /work/receipts/first3-device-v1/run.py --source /work/source-layer0-v1 "$@"
