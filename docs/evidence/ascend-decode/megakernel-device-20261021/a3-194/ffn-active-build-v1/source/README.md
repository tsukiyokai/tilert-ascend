# Complete FFN with active-expert enumeration

This isolated variant inserts the accepted active-expert body into the complete
FFN. Its sorted table contains only nonempty experts, retaining matrix calls,
row ownership, arithmetic and all six body barriers. The table occupies one
additional INT32[264] tensor at body slot 20. The bridge and output epilogue
remain unchanged.

```sh
python3 build.py FROZEN_COMPOSED_BUILD_V2 FROZEN_ACTIVE_BODY NEW_OUTPUT
python3 prepare_check.py FROZEN_COMPOSED_CHECKER NEW_HARNESS
```

The checker compares the original 30 boundaries and the entire active table
against both reference paths. The table expectation is independently derived
from reference expert counts, including its zero padding. M=12 with 24 groups
is the integration candidate supported by the standalone body's paired timing;
complete-FFN performance requires a separate paired measurement.
