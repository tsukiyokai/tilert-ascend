# Exact M32 KV sharing: structural screen

For these 21 frozen M32/SFA producers, union plus a query mask has the
lowest aggregate logical KV reads and modeled padding work. Its four-head
padding work is 45.60% of the current full-cache FIA control; plain cascaded
sharing is 49.42%. Union wins this work model at 19 producers; cascade wins
at producers 1 and 22. This supports studying union/tile sharing inside one
kernel. These counts establish neither latency nor measured HBM traffic.

## Counts

Totals below sum one M32 execution for each of 21 producer selections,
covering 32 queries and 2048 selected indices per query. They are not totals
over all 78 model layers. Each QK/PV column applies separately to QK and PV.

| Mechanism | Logical KV row occurrences | Pairs/head, no padding | Four-head pairs, no padding | Four-head padded pairs (16×128 tiles) | Segments |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-query Top2048 | 1,376,256 | 1,376,256 | 5,505,024 | 22,020,096 (10,752) | 672 |
| Union + query mask | 88,382 | 2,828,224 | 11,312,896 | 11,452,416 (5,592) | 21 |
| Cascade 32→16→8→4→1 | 543,942 | 1,376,256 | 5,505,024 | 12,412,928 (6,061) | 987 |
| Full-cache FIA control, 9344 rows | 196,224 | 6,279,168 | 25,116,672 | 25,116,672 (12,264) | 21 |

The effective selected interactions are 1,376,256 pairs per head, or
5,505,024 pairs across four heads, for every exact sparse mechanism. Union
and full-cache rectangles additionally compute masked cells; these are
included in their unpadded matrix work above.

Union has 2937–4964 rows per producer and one segment. Cascade has 47
nonempty segments per producer: 1×32-query, 2×16-query, 4×8-query, 8×4-query
and 32×1-query segments. Each branch can repeat rows occurring elsewhere
in the tree. [table.csv](table.csv) contains all producer counts;
[result.json](result.json) includes the cascade breakdown by group size.

## Work model and exactness

A segment contains `g` neighboring queries and `N` KV rows. The actual
rank has four Q heads: checkpoint configuration is 64 heads / TP16, so the
folded matrix dimension is `M=4g`. For each QK or PV operation:

- Logical pairs per head: `Σ gN`.
- Four-head work without padding: `4Σ gN`.
- Assumed padded work: `Σ ceil(4g/16)×16×ceil(N/128)×128`.
- Tile count: padded work divided by `16×128`.

One logical KV read means one row occurrence per nonempty segment, shared
by its queries and four heads. K and V are not counted twice. A row used
by separate segments is counted again. These are logical counts and an
assumed tile model, rather than observations of the current kernel.
Segments expose scheduling and launch risk; a fused kernel can consume
multiple segments in one launch.

The decomposition follows [QUILT §3.3](https://arxiv.org/html/2610.11134v1#S3.SS3):
emit the intersection of the current group's residual sets, remove it
from each query, and recurse into contiguous subgroups. The levels are
32→16→8→4→1. Every leaf residual is retained. Ancestor subtraction makes
segments disjoint on each query's path. The script reconstructs all
21×32 query sets for all three mechanisms: 2016 exact reconstructions,
with no missing or repeated indices. This is selected-set exactness.

## Evidence and reproduction

The input is the completed SFA rank0 report with 32/32 teacher successors:
`7deccb93e6aeb797e3b41344e8dcf614bcd776a8d929f4c28b0fb71fc38e4eba`.
Each of 21 producer tensor files is checked against its recorded SHA256;
their hashes and canonical selected-set hashes are in [result.json](result.json).
The checkpoint config hash also matches the run report.

[screen.py](screen.py) runs on the a3-194 server CPU with one intra-op and
one inter-op thread, backend autoload disabled, and `torch_npu` absent.
Run `bash command.sh` on a3-194. Results are available at
`/tmp/m32-full-teacher-20261020/exact-sharing` locally and on a3-194;
the container execution files are under
`/workspace/experiments/m32-full-teacher-20261020/exact-sharing`.

## Current wrapper constraint

The installed operator schema takes `actual_seq_kvlen` as `SymInt[]`, not
a device Tensor ([CPU schema receipt](fia-length-contract.json)). A device
union count therefore needs a different execution path or a host readback
with this wrapper. This is not a claim about every CANN entry point. A fixed
capacity must preserve overflow; the observed maximum4964 does not bound
rejected draft suffixes. Gathering also adds source reads and destination
writes before FIA reads the packed cache.
