# Wide-verification expert budget

The current TP16 W4A8 representation assigns128 intermediate columns to
each rank. Per distinct routed expert, each rank holds:

| Projection | W4 bytes | UINT64 scale bytes | FP32 compensation bytes |
| --- | ---: | ---: | ---: |
| Gate | 393216 | 1024 | 512 |
| Up | 393216 | 1024 | 512 |
| Down | 393216 | 49152 | 24576 |
| Total | 1179648 | 51200 | 25600 |

For `S = sum(U_layer)` over75 routed layers, logical expert reads are
`1,256,448 * S` bytes per rank, assuming one read per participating group.
This is already TP-sharded: do not divide by16 again. Reuse the existing
[estimator](../../../../../tests/glm52/estimate_traffic.py) and
[representation audit](../../cann-traffic-source-audit-20261017.json).
The installed W4A8 source still matches that audit's `d06549f` hash prefix.

For M32, each row selects eight distinct experts. Consequently each layer
has256 routed rows, each expert receives at most32 rows, and `8 <= U <= 256`.
Across75 layers, `600 <= S <= 19200`. GMM1's doubled MSD rows per expert
are at most64, below baseM128; this width does not require another M tile.
Counts33/64 in stress checks exceed actual M32 routing multiplicity.

Logical bytes are not measured HBM traffic. Cache reuse can reduce reads;
kernel reloads and scratch can increase them. A necessary W4-only condition,
given an upper bandwidth bound B and C bytes already resident in reusable
on-chip storage, is `max(1,179,648*S - C, 0) <= 0.002*A*B` for a2ms/token
budget and A committed tokens per round. All other work must also fit.

The old A=6.959 trajectory is not an accepted new lookup result. Current
teacher/M1 captures cannot supply the rejected candidate suffixes' routes.
Measure S, per-expert counts and A on the actual lookup session, then price
its chosen widths; neither random routes nor teacher-forced unions replace it.
