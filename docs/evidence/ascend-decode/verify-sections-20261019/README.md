# The verify by section, each against the bytes it actually moves

A bounded width-8 capture, analysed after the fact — `analyse: False` on the
collector, so the raw data needed turning into `trace_view.json` and
`kernel_details.csv` separately. The run passed this harness's gate:
`status: pass`, `serial_speculative_consistency: True`, zero failed checks, so
width 8's speculative decoding emits the same tokens as serial decoding and
the `68f2bc184553` stream is not a speculative artefact.

Scaled to one verify by the ratio of the verify's measured device time to the
capture's, 39.677 / 163.023 = 0.2434. The scaling is checked against a
per-layer count: 61 MoE layers × 90.8 µs = 5.54 ms against the 5.456 the
global factor gives.

| section | share | live bytes | floor | cost | ratio |
| --- | --- | --- | --- | --- | --- |
| MoE | 34% | 4.29 GiB | 2.99 ms | 13.57 | 4.5× |
| attention | 23% | 1.12 GiB | 0.78 ms | 9.20 | **11.7×** |
| the rest | 43% | small | — | 16.91 | higher |

So the section a MoE megakernel would address is the healthiest of the three.

## The shapes, from `kernel_details.csv`

| operator | per call | blocks | shapes |
| --- | --- | --- | --- |
| `GroupedMatmulSwigluQuantV2` | 89.809 static, 98.950 dynamic | 24 / 48 mix | `8,6144 ; 8 ; 256 ; 256,4,384,16,64` |
| `GroupedMatmul` | 58.226 | 24 / 48 | `8,128 ; 256,128,6144` int4 |
| `SparseFlashAttention` | 89.01 | 24 / 48 | q `8,4,512`+`8,4,64`, cache `73,128,1,512`, picks `8,1,2048` |
| `MlaPrologV3` | 37.40 | 24 / 48 | `8,6144` through `6144,2048`, `2048,1024`, `6144,576` int8 |
| `TransposeBatchMatMul` | 12.08 | 24 / **0 mix** | `8,4,512 × 4,512,256` |
| `QuantLightningIndexer` | 42.02 static | 24 / 48 | `8,32,128` against `73,128,1,128` |
| `LightningIndexer` | 62.83 dynamic | 24 / 48 | the same shape in bf16 |

## What each ratio is made of

**The grouped matmuls are at 2.75× and 3.56×.** gmm1's weight tensor is 192
MiB of int4 and gmm2's 96, but batch one routes 8 rows to at most 64 of 256
experts, so 48.00 and 24.00 MiB are live — 32.68 and 16.34 µs against 89.809
and 58.226. Three times, not ten. An earlier claim of mine that these calls
move far too few bytes to account for their time was wrong by that factor.

**Attention's 7.26× has an eightfold redundancy inside it.** The cache is 73
blocks of 128 — 9344 slots, the configured capacity — at 1152 B a slot. One
layer reads 8 rows × 2048 picks × 1152 B = 18.00 MiB, which is 12.3 µs against
89.01. But the eight rows are eight consecutive positions choosing from the
same scores, so their unions should nearly coincide: 2.25 MiB, 1.5 µs, 58×.
Whether they do is being measured rather than assumed.

**`MlaPrologV3` is at 3.3×** — 17.4 MiB of int8 projection weight, 11.3 µs
against 37.40. Healthy, like the MoE.

**`TransposeBatchMatMul` is almost all fixed cost.** `8,4,512 × 4,512,256` is
4.2 MFLOP and takes 12.08 µs on 24 AI_CORE blocks with no vector blocks at
all. Over 256 calls that is 3.1 ms of the capture.

## Two dispatch findings

`GroupedMatmulSwigluQuantV2` appears both as `static` at 89.809 µs and as
`dynamic` at 98.950 — 9.1 µs apart, and `Model ID` 4294967295 (eager) as well
as 49 (the draft graph). So some calls miss the static kernel.
`QuantLightningIndexer` is `static` at 42.02 while `LightningIndexer` is
`dynamic` at 62.83, 49% more, over the 19 calls that are the prompt replay.
The same shape of miss in two places.
