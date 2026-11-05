# Native decode composition on a3-194

The experiments use physical die 4 alone, exposed as `npu:0`, after repeated
occupancy checks. The routed/shared comparisons use CANN 9.2.0 and
Torch/Torch NPU 2.12; each report records loaded-library and source hashes.

The [active-expert list candidate](routed-pipeline-front-active-v1-summary.json)
adds one INT32[264] scratch tensor. Its eight-word header stores the count,
followed by sorted active expert IDs and a zeroed tail. Both BF16 and
dispatch inputs produce the table before the existing first barrier.
Projection calls, row ownership, arithmetic and all six barriers match the
selected parent. This candidate does not include the earlier Down stripes
or gate/up K split.

The initial 128 changing graph replays pass 4,864 raw-byte boundary comparisons.
The cases cover M=1,2,12,32, mixed routes, 8-to-256-to-8 active-expert
transitions, 16/24 core groups and both input entries. The complete table
is compared with an independent count-based oracle on every replay.
The [build evidence](routed-pipeline-front-active-v1-build/)
and [frozen harness](routed-pipeline-front-active-v1-harness/source-sha256.json)
record the exact candidate and checks.

Eight timing fixtures use 24 distinct equal-valued weight banks, all six
arm orders, and 30 paired samples. The two native arms share 17 writable
scratch addresses and their synchronization buffer; the experiment repeats
with both original allocation sets. Positive savings below favor the candidate.

| Width | Routed expert union | Paired saving, candidate arena | Paired saving, control arena | Wins |
| --- | --- | --- | --- | --- |
| 12 | Mixed, 81 experts | 5.81 us | 4.63 us | 60/60 |
| 32 | Mixed, 158 experts | -0.88 us | -1.47 us | 0/60 |
| 12 | Same eight experts for all tokens | 22.40 us | 19.72 us | 60/60 |
| 32 | Same eight experts for all tokens | 16.56 us | 19.65 us | 60/60 |

Including these fixtures, all 1,664 numerical replays, 63,232 raw-byte
boundary comparisons and 32 post-timing output checks pass. The measured
scope ends at the two local BF16 outputs, including shared DynamicQuant;
output addition and TP reduction remain outside it. The results support
M=12 integration. Mixed-route M=32 regresses on both address sets, and
M=1/2 have no timing evidence, so a global replacement is not selected.

The [adapter validation](routed-front-active-adapter-v2-summary.json) passes
96 changing graph replays and 3,568 raw-byte comparisons through the actual
`routed_shared.run_body` entry. Its shape selection uses the active variant
only for M=12 with 24 groups, including workspace preparation before capture.
The remaining shapes use the parent. The checks cover mixed and extreme
expert distributions, FP32/BF16 shared scales, and both group counts. This
scope excludes the vLLM runner and TP finalization.

The [callable-body validation](routed-body-v2-summary.json) passes another
96 graph replays and 3,552 raw-byte comparisons against the selected parent.
The extracted body preserves all numerical statements and six internal
barriers while taking a caller-owned pipe. Its standalone wrapper owns the
device initialization. These checks cover M=1,2,12,32, mixed/extreme expert
sets, FP32/BF16 shared scales and 16/24 groups. The containing kernel must
publish and synchronize the final stores before a subsequent phase consumes
them; the selected non-profiled body has no implicit final barrier.
