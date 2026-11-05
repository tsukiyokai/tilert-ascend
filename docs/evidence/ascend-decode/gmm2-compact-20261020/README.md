# Compact GMM2 active groups

A3 die15, a3-194/trt-asc, CANN9.2.0 B114. INT8×NZ INT4→BF16,
E256/K128/N6144; speculative width1/8/32 means8/64/256 routed rows.
Saved width8 target profiling attributes6.173ms/75 calls to this family.
No whole-model speedup is established.

## Mechanism and correctness

Runtime key69325233410 selects API-dequant MSD. AIV0 publishes ascending
(original expert ID, cumulative row end) records once; AIC/AIV consume them
after the existing mixed barrier. The table occupies2080B beyond the6MiB
half ring, within the96MiB user workspace. NZ addressing, per-core tile order,
FP16 partials and FP32 epilogue remain unchanged. See [source audit](audit.md)
and [independent review](adversarial.json).

- Server CPU:102 routing cases,86,652 tile assignments/arm, all24 cores exact.
- [Factory](official-v2.json) and [candidate](candidate.json):24 changing-route
  Graph cases,16,121,856 BF16 outputs bit-exact against independent CPU MSD
  arithmetic and across binaries. Sparse expert255 and full256 are covered.
- [Real checkpoint](real-weights.json): layer4/position8192/rank0, four eager/
  Graph calls match both the independent arithmetic and frozen CANN down output.
- [Binary access](candidate-access.json) binds candidate SHA0fe4dbd8…;
  the release-builder ELF is byte-identical. Global CANN installation is unchanged.
- Final package: all26 factory variant hashes remain valid; an unrelated BF16
  ND projection passes. The [release overlay](release-real.json) repeats allfour
  real-weight checks, with its own [binary access receipt](release-access.json).

## Paired timing

[Raw samples](paired-v2.json): one shared die, eight independent full weight
banks,128 calls/Graph, input restored before every call,10 warmup pairs,
30 ABBA samples, profiler off, NPU events, unlocked clocks, no discarded
samples. Peak reserved2386MiB. Both paths run in one test ELF; a32-byte
group-list address offset selects dense traversal with identical numeric data.
The switch is absent from the release patch. [Access receipt](paired-v2-access.json).

| Width / routing | Dense p50 µs | Compact p50 µs | Paired median delta µs | Wins/30 |
| --- | ---: | ---: | ---: | ---: |
| 1 / random top8 | 62.424 | 34.860 | −27.577 | 30 |
| 8 / random top8/token | 91.266 | 72.614 | −18.655 | 30 |
| 32 / random top8/token | 149.682 | 149.875 | +0.293 | 5 |
| 32 / shared8 | 64.587 | 41.403 | −23.123 | 30 |
| 32 / all256 | 214.628 | 214.949 | +0.339 | 2 |

The [unmodified-ELF address control](alignment-control.json) has paired median
deltas−0.310/+0.007/−0.283/−0.335/−0.435µs for these cases. This bounds the
pointer-switch confound well below the M1/M8/shared8 effects. M32 random and
full-union results provide no useful speedup. Price real expert unions before
combining this with lookup or shared FIA.

## Reference correction

The original CPU oracle used all FP32 scale bits. A3 Fixpipe requires the low13
bits cleared before dequantization; the [official source receipt](fixpipe-official-receipt.json)
states this contract. On one frozen synthetic8-row call, this removes all3548
mismatches out of49,152 outputs. The corrected oracle then passes all24 cases.
Commit2624589 includes a four-output CPU regression check.

The eight real checkpoint experts sampled here already have zero low13 scale
bits. This correction does not explain the historical teacher divergence.

## Reproduce

Activate the unified container environment, then run
`tests/glm52/build_gmm2.sh NEW_OUTPUT --compact-experts` and prepend its printed
vendor path to `ASCEND_CUSTOM_OPP_PATH`. Omitting the flag rebuilds the control.
The builder pins B114 kernel/tiling identities and preserves other factory
GroupedMatmul variants. Source and harness copies are under [source](source/).
Code commit2adbb31; the reference correction is separate commit2624589.

Server artifacts: `/home/shanshan/trt-asc-194/experiments/gmm2-compact-20261020`.
The first timing harness stopped before measurement because cloning packed NZ
weights changed their format; v2 builds each bank through production weight
postprocessing. All failed-run logs remain on the server. No NPU kernel failure.
