# Three-rank tree reduction

The private tree reducer matches the installed deterministic FP32 HCCL result
byte for byte in all 72 cases. The original serial reducer reproduces the
previous order difference. The comparison ran on a3-194 physical dies 4, 6 and 7
with Torch 2.12, Torch-NPU 2.12 and CANN 9.2.

There are two row counts, M1 and M12, and two input families. Each family has
two eager inputs and four graph-replay inputs per shape and rank. Across the
two native arms this gives 12 captures and 96 explicit graph replays.

The order family retains the three columns `[1, 2^-25, -1]`, `[1, -1, 2^-25]`
and `[1, 2, 3]`, scaled by changing powers of two. The edge family rotates 32
finite BF16 columns without scaling: signed zeros, subnormals, cancellation,
finite extremes that overflow during addition, and final BF16 rounding ties.
NaN inputs are outside this fixture.

All 576 recorded full-tensor byte comparisons match their intended oracle:
the parent matches the serial expression, the candidate and HCCL match the
tree expression, and the parent differs from HCCL. All 72 input-retention and
144 immediate-consumer checks pass. The independent scalar
[audit](audit.py) reconstructs every comparison dictionary and all 72 recorded
FP32 HCCL input and sum prefixes. It also verifies the frozen source and DSO
identities, address separation, device mapping and loaded dependencies.

Runtime logs contain six algorithm selections and 72 FP32 SUM launches using
`AllReduceMeshAivFor91093Executor`, deterministic execution, and no explicit
`HCCL_ALGO` override. These records are retained in
[the complete selector/launch excerpt](hccl-selection-and-launch.log).

The [summary](summary.json), [case audit](case-audit.json),
[launch audit](launch-audit.json), [raw-file manifest](raw-manifest.json),
source snapshots, launch command and two idle-admission samples are retained.
All 20 raw files are archived; compressed logs are checked against the original
93,268,732 bytes. Device reports retain full-tensor comparison metrics and the
first 32 values as raw bits, rather than complete tensor dumps.

Each probe executes two reductions with a dependent CUBE stage between them.
Only its first reduction and immediate consumer are compared here. This
private probe is not a whole-layer, TP16 or performance result.
