# What 5x actually requires

TPOT is round time over tokens accepted a round, and the two factors multiply.
At 10.8528 ms measured, 2.x ms is 5x. Neither factor reaches it alone.

## Acceptance is already at its ceiling, and the ceiling has a closed form

Expected tokens accepted a round, for single-token acceptance rate α and draft
length γ, is

    E = (1 - α^(γ+1)) / (1 - α)

which tends to 1/(1 - α) as γ grows. Measured here: 7.1 accepted at width 32,
so α ≈ 0.86, and 1/(1 - 0.86) = 7.14. **The measurement is the asymptote.** The
width sweep that saturated between 32 and 256 was not an implementation
limit; no draft length can beat 7.14 at this α.

Published figures put that α in context. A draft model on coding tasks gives
acceptance length 5.24, falling to 1.84 on creative writing. A general n-gram
curve reads 1.86 accepted at N=1, 3.12 at N=3, 3.86 at N=5 and 4.25 at N=7,
with throughput peaking at N=5 for 1.18x. EAGLE-3 on long outputs grows to
2.5-5.7. So 7.1 is above all of them, which is what an agentic coding model
copying its input buys.

Raising acceptance therefore means raising α, not lengthening a chain: several
candidate branches verified in one pass, where the round takes the best. α of
0.92 gives E = 12.5, a factor of 1.75, and that is about the ceiling of this
lever -- 1/(1 - α) is unforgiving.

## Round time is 4.6 times its own byte floor

Measured 46.277 ms of device time a round against a 10.057 ms floor priced
from the bytes each operator must move. The named pieces, all measured:

| what | measured, ms a round | floor | lever |
| --- | --- | --- | --- |
| launch floor, 3008 calls, 82% on MIX cores | 11.512 | — | one kernel a layer takes 3008 calls to 78-156 |
| routed experts | 12.39 | 3.48 | [the INT8 kernel measures 3.70](../routed-kernel-20261020/README.md) |
| TP reduction, 0% overlap with compute | 5.35 | — | one-sided, fused into the layer kernel |
| SFA attention | 6.973 | 1.382 | 5x off its floor, unattacked |
| `Cast` + `ForeachAddList` | 3.922 | 0.094 | pure overhead; [2.7 already shown removable](../../../probe.md) |
| `QuantBatchMatmulV3` | 3.810 | 0.780 | — |
| `MlaPrologV3` | 2.950 | 2.745 | 1.1x, the in-capture control |

Stacking what is already measured or built -- layer fusion, the INT8 routed
kernel, INT4 weights, an in-kernel one-sided reduce, the fused epilogue --
takes the round from 46.3 to about 15 ms. That is 3x.

## So the arithmetic of 5x

3x on the round times 1.75x on acceptance is 5.25x. Both are required: layer
fusion alone is 3x and tree drafting alone is 1.75x. The residue after both is
the attention kernel at 5x its floor and the KV-cache traffic behind it, which
is where a sixth factor would have to come from.

Sources for the acceptance figures:
- https://arxiv.org/pdf/2601.11580
- https://www.morphllm.com/speculative-decoding
- https://arxiv.org/pdf/2604.26469
- https://docs.vllm.ai/en/latest/features/speculative_decoding/
