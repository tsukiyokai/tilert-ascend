# Fixed-budget copy tree

A copy-only tree has proposals at every visited prefix: all candidates come
from committed context. It avoids the missing off-boundary MTP proposals that
prevented the earlier mixed-tree estimate from establishing a gain.

Policy: shortest1/longest8; select the newest2/4/8 distinct continuations at
the longest matching suffix. Pad each continuation with pending to31 tokens.
Allocate nodes in ascending depth, then continuation recency, merging common
prefixes; stop at31 non-root nodes. Pending is the root, so every verification
has32 rows. Freeze proposals before reading the reference continuation.

| Policy | Rounds | Verified rows | Committed/round | Complete mean round cost allowed vs flat |
| --- | --- | --- | --- | --- |
| Flat32 | 147 | 4704 | 6.959184 | 1 |
| Tree2 | 135 | 4320 | 7.577778 | <1.088889 |
| Tree4 | 132 | 4224 | 7.750000 | <1.113636 |
| Tree8 | 140 | 4480 | 7.307143 | <1.050000 |

Each policy walks its own boundaries and commits1023 tokens, excluding the
common first output. Twelve CPU checks cover duplicate prefixes, allocation
ties, no-match padding, scalar traversal, final bonus clipping and agreement
with the existing reverse scan. Scalar traversal and independent leaf-prefix
matching agree at every round.

Tree4 spends4.975ms constructing proposals over the request versus0.492ms
for flat; index initialization52.54 versus21.43ms, with commit costs separately
recorded. These are single pinned-core samples on a3-84, not paired timing.
The same teacher-unaccepted trajectory is the sole input. The cost ratios
exclude setup and must cover the complete device, commit and host round.

Tree4 improves this CPU acceptance screen but requires sibling exclusion
before indexer top-2048, depth-based RoPE, ancestor-only attention and selected
path KV compaction. Its11.36% allowance does not establish a device gain.
No production tree runtime is enabled.

The [installed-source audit](capability/capability.md) identifies a selector
hook but no arbitrary ancestor mask in the A3 C8 API. Its sorted output alone
does not justify an exact prefix/tail merge: sequence-length-dependent core
partitioning changes tie order. Retain this route as a candidate until accepted
inputs and measured round cost justify the required kernel/state changes.

[Replay](replay.py), [summary and source/input hashes](summary.json),
[per-round nodes, sources and commits](raw.json).
