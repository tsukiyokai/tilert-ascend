#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ATTENTION_EXPORT=1 bash "$here/attention_build.sh" \
    "${1:-$here/attention_export_build}"
