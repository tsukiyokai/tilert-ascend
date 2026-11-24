# SFA C-template routing, 2026-10-19

## Decision

Retain V1. Dispatching ordinary token indices to C1 is numerically correct
on the tested fixtures but approximately 11x slower in the paired M8 screen.
C4 packing has no general production contract: GLM-5.2 uses RoPE64 and the
standard lightning indexer; index_topk_freq=4 means reuse across layers.
The KPool path rejects nonzero RoPE dimensions.
[Production contract](production-source-audit.json).

## Full host and ordinary indices

A3-187 `trt-asc-187`, die 0. The rebuilt host preserves all 44 installed
operator registrations and their tiling/parse functions and data sizes.
The activated toolkit on both build and execution hosts is 9.2.0 /
V100R001C25B114. The original-header rebuild reproduces the installed BF16
ELF exactly; only the bounds-first header changes the candidate ELF.
[Registry audit](c1-full/registry-audit.json), [build manifest](c1-full/manifest.json).

The frozen synthetic input is M8, four heads, 8192-token prefix, 2048 arbitrary
valid token IDs per query, reversed page table and ordinary index storage.
Input SHA: `d8b00212e619c88383dcb79c487c050fa38b26a29467240e8ac66d7261f5d23a`.
Original C1, bounded C1 and bounded V1 produce identical six A/B/reset
eager/Graph outputs: max_abs=0, NRMSE=0. All outputs are nonzero; A/B differ
in 16,374 of 16,384 elements. Peak Torch reservation is 182 MiB under a
256 MiB allocator quota, excluding runtime context.
[Correctness](c1-full/correctness-summary.json),
[loaded host/device identities](c1-full/loaded-host-device-audit.json).

## Paired screening

Both timing routes use device ELF
`1f4b79fa28c7329237ebfe24f04877a51dc2acb41a85c8dca7489acf02491a3a`.
Only the full host dispatch differs; actual keys are V1=1158 and C1=134.
Each arm uses ten eager warmups, ten capture calls, and thirty measured
replays of a ten-call graph; 1280 calls total, no added pairs.

| Order | V1 event µs/call | C1 event µs/call | C1/V1 |
| --- | --- | --- | --- |
| AB | 90.053 | 998.380 | 11.087 |
| BA | 92.406 | 1002.533 | 10.849 |

Wall medians in A1/B1/B2/A2 order are 103.942/1010.199/1017.764/105.589 µs.
All four graph outputs match eager. Timing peak reservation is 104 MiB.
Before/after snapshots show resident models, HBM 50117–50118/65536 MB and
0% AICore; they do not establish exclusive ownership throughout each arm.
These are hot-input single-operator screening results, not decoder TPOT.
[All samples, hashes and occupancy](c1-full/performance-summary.json),
[raw arm records](c1-full/screen-two-pairs/summary.json).

The installed Cube service walks selected token IDs separately in both MM1
and MM2. With block_size=1, each iteration copies one row after scalar page
lookup and address calculation; V1 instead copies a prepared 128-row tile.
This identifies the serial work retained by C1, without assigning unmeasured
stage times. [Installed-source anchors](c1-full/source-cost-audit.json).

## Bounds and earlier fixtures

The full-K2048 fixture ends on an exact tile boundary and does not exercise
the terminal bounds branch. The separate literal CPU transcription covers
16 cases/46 calls: 28 defined calls preserve outputs/read sequences and
18 invalid initial reads disappear. Device coverage of partial-tail termination
remains absent. [Scalar evidence](c1/bounds-summary.json).

The earlier guarded C1 fixture includes causal future entries, trailing−1,
page crossings and partial tiles; it passes cross-route eager/Graph/reset
comparisons. The C4 fixture proves only its block-closed synthetic selection.
[Guarded C1](c1/single-die-summary.json), [C4](c4/single-die-summary.json).
All patches and builds remain isolated server artifacts; production libraries
were not replaced. The C1 dispatch candidate is not integrated.
