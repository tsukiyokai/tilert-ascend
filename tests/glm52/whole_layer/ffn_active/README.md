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

The [M12/24 device gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active212-m12-v1-a3-194/summary.json)
passes eight changing-input replays and 16 dual-reference cases. All 496
raw-byte comparisons pass across the original 30 boundaries and the full
264-element active table. Routed weights use NZ storage and shared scales
retain BF16.

The [complete-FFN paired measurement](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-timing-v3-a3-194/summary.json)
at M12/24 with 65 active experts saves 5.659/9.549 us in two allocation-owner
arenas, with 30/30 wins in each. Candidate/base medians are 235.217/240.890 us
and 232.566/242.127 us. Each pair uses the same 32 common scratch addresses,
16 independent weight banks and alternating event order. All 19,200 numerical
boundary checks, 360 event-tail checks and 348 entire-table comparisons pass.
The table starts with a zero count and nonzero padding/ID sentinels before
checked replays. These results cover this local FFN fixture; other widths,
route distributions and model TPOT require their own measurements.
