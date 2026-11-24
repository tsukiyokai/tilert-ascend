#!/usr/bin/env bash
set -euo pipefail

exec "${SFA_METADATA_BISHENG:?Set the actual bisheng path}" \
    -I "${SFA_METADATA_OVERLAY:?Set the prepared SFA overlay}" "$@"
