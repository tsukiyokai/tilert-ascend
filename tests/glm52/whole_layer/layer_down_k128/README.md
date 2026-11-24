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
The device comparison passes numerically but does not show a consistent
performance improvement across the two allocation arenas.

## Device result

On a3-194 physical die 4, all 6,660 numerical, 600 event-tail, 844 complete
Down-tiling and 388 complete-record comparisons pass. The paired median savings
are -1.142 us and +0.743 us per indexed/shared pair, with 6/30 and 23/30 wins.
The candidate is not selected for this local-layer workload. Its standalone
FFN gain with 16 weight banks does not establish a gain here with one bank.

Evidence: [device audit and raw measurements](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/layer-down-k128-timing-v1-a3-194/AUDIT.md).

## Frozen CPU gate

The actual AArch64 Torch212 image passed all 18 contracts without skips, four CLI
controls and six-library/11-schema co-load. All 895 fixture files and 7,953 runtime
files matched their frozen hashes. The container used one CPU, no network and no
device mappings; NPU initialization remained false before and after the gate.
The x86 run passed 17 contracts and skipped the AArch64-only host preparation test.

Evidence: [CPU report and source manifest](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/layer-down-k128-timing-build-v1/summary.json).
Fixture manifest: `006088bf725f910bb2578eaed567bcfe4e396a1e349444c3792ee34906972fa8`.
The fixture is frozen at `/tmp/tilert-layer-down-k128-timing-staging-v1` and
deployed to `a3-194:/home/shanshan/tilert-megakernel-20261021/vendor-hook-layer-down-k128-timing-v1`.
`paired-tilings.json` and `k128-compiled-tilings.json` define the current byte
oracles; inherited N1024 records remain historical provenance.

The device runner command, for the root agent's single-die queue, is:

```bash
bash /home/shanshan/tilert-megakernel-20261021/vendor-hook-layer-down-k128-timing-v1/device_local_chain_run.sh \
  4 layer-down-k128-timing-v1 --forward-widths 12 --forward-replays 8 \
  --indexed-library /evidence/attention-wk-m1-build-v1/whole_layer_indexed_attention.so
```
