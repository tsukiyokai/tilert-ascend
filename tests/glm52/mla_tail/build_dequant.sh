#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MLA_TAIL_DIAGNOSTIC=dequant exec bash "$here/build_profile.sh" "$@"
