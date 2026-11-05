# K128 Down tiling in the 609 two-layer chain

This fixture compares N1024/K128 against N512/K128 inside the same frozen
`609c973e…` local-layer kernel. Both arms retain the explicit TopK copy between
the indexed and shared layers. The control uses `c90b3666…` preparation; the
candidate uses `0842cd0f…` preparation. No new device binary is built.

The measurable claim is lower time for the complete indexed/shared pair under
this workload. Standalone FFN improvements are not added to earlier layer
measurements. M12 needs one M tile in both arms: N512 has 12 N tiles, while
N1024/K128 has six. K128 also changes the L0 buffering configuration. The paired
measurement determines their combined effect in the surrounding layer chain.

## Address and numerical contract

The two arms share every writable production address, weight, cache and retained
TopK owner within each arena. Each layer keeps its original candidate preparation
and an independent complete `c90.prepare` owner. Only the latter's read-only
`body[7]` enters the control graph. The unused control scratch remains outside the
timed graph and is retained for its original Torch validator. Both full owner sets
live until graph reset.

The frozen C++ layout checks all 98 configuration fields and 116 address fields.
The 664-byte configurations are identical. The 928-byte address records may differ
only at `body.scratch[7]`, bytes 624–631; padding is also compared. Each prepared
Down tensor must match all 200 bytes before and after execution:

| Arm | Down tile | Full record SHA256 |
| --- | --- | --- |
| Control | M64/N512/K128 | `bb8ef4481b8c63b00dc43940975ee5b172818bcd228e8bc553b22234cbb2a244` |
| Candidate | M32/N1024/K128 | `c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999` |

The source check verifies 49 embedded FFN C++/header files against the frozen 381
parent. Of those, 48 K128 files match after namespace normalization; only the host
tiler differs. Its additional host plan is compiled into the preparation DSO.
The embedded device consumer, all arithmetic, dynamic expert M, synchronization,
and TopK ownership are unchanged.

The existing gates remain: original norm plus quantization against the native
`norm_quant_out` reference, each arm's owning FFN validator, complete 109-boundary
snapshots, cache resets, fresh-write poison including all 264 active-table words,
input/owner checks, and immediate event-tail checks. The two-layer successor
reads the first layer's actual output and residual. Three repeated graph replays
establish idempotence before timing. Norm reference copies occur outside events;
the timed graphs contain neither those copies nor unused control scratch.

## Timing scope

M12, 24 mixed groups, two retained allocation arenas, one weight bank containing
the two actual synthetic Decoder weight sets. Each graph contains 16 two-layer
pairs. Each event measures 30 graph replays, with 30 alternating pairs per arena.
Results are microseconds per indexed/shared two-layer pair. There are no TP
collectives. This does not measure a multilayer weight working set or TP16 TPOT.

`stage.py` derives the immutable fixture from the validated N512/N1024 timing
bundle. The event loop, replay ordering, poison, snapshots and retirement logic
remain unchanged; only candidate preparation identity, tiling bytes and report
labels change. CPU checks reuse the existing 18 contracts and four CLI controls.
The actual ARM preflight must load six DSOs and all 11 schemas without NPU access.
Device execution and performance remain unmeasured for this combination.
