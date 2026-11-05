#!/bin/bash
set -e
exec /work/tools/strace/usr/bin/strace --seccomp-bpf -f -e trace=openat -o "/work/receipts/layer0-boundaries-v4-rank-${RANK}-strace.log" python -B /work/receipts/layer0-boundaries-v4.py --source /work/source-layer0-v1 "$@"
