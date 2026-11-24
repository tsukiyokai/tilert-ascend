# Compact active experts inside W4A8 GMM1

Commit `6292794`: opt-in CANN kernel patch for GLM-5.2 TP16: K6144, N256, 256 experts,
count-form groups, one workspace chunk and at most256 dispatched rows.
M1/M8/M32 verification produces8/64/256 such rows. The default build is
unchanged; full-model teacher and TPOT validation remain pending.

## Why this path

The saved width8 trace contains7 ranks and2 hot rounds per rank. Target
main-stream task unions are40.096ms median: routed up/down13.117ms,
shared-expert core2.306ms, router/permute3.303ms, SFA7.037ms,
MLA prolog/output5.965ms and indexer2.645ms. Separate-stream communication
is excluded from these sums. These are old-profile attribution numbers,
not current M32 timing. [Task groups](profile/target-summary.json)

All8 head tails total1.011ms/round. Removing their gather/layout/selection
would remove about0.43ms, or0.095ms/token using the old acceptance4.507;
native distributed argmax is deferred. [Head attribution](profile/summary.json)

## Mechanism and invariants

AIV0 bulk-loads the256 counts, writes an ascending list of `(expert, end)`
into the unused second pre workspace, and publishes it through the existing
pre→mid barrier. Each consumer invalidates its scalar DCache before reading.
The table uses at most2080B of the28,760,064B bank. Multiple chunks retain
the original pipeline.

Mid preserves active-expert order, tile order and core assignment. MSD uses
`m=2*count`, `xBase=2*previous_end*K`, `yBase=2*previous_end*N`.
The actual source already indexes weight/scale by expert ID; its unused
`wBaseOffset` is not a reason to scan empty experts. Post finds the first
`end > starting_row`, then advances the compact cursor at each expert boundary.
Unused AIVs never dereference that cursor. [Source audit](cpu/source-audit.md)

## Verification and timing

a3-194, trt-asc, CANN9.2.0 B114, die15, foreign Ray workers resident.
CPU invariants:63 cases,3425 tile records and6888 post rows,0.107s.
[CPU result](cpu/cpu-invariant-summary.json)

Same-ELF ±0 switch, identical arithmetic, one process,32 calls per graph,
two weight banks,10 warmup pairs and30 ABBA samples, profiler off.
All five shapes win30/30 pairs; no samples are discarded. Clocks are unlocked.
Numbers are device event microseconds per call; host replay is below1.5µs.

| Verification width / expert union | Dense p50 | Compact p50 | Paired delta p50 |
| --- | ---: | ---: | ---: |
| M1 / 8 | 90.895 | 49.359 | −41.473 |
| M8 / random top8 per token | 111.561 | 83.914 | −27.660 |
| M32 / random top8 per token | 163.276 | 141.164 | −22.019 |
| M32 / shared8 | 96.553 | 59.977 | −36.560 |
| M32 / all256 | 211.161 | 192.990 | −18.273 |

[Raw paired samples, distributions, p90/p99/std](paired.json),
[full-union samples](paired-full-union.json). Full-union samples are noisier;
these are shared-load operator screens, not an end-to-end TPOT gain.

Two independent processes, identical hashed raw inputs:24 changing-group
graph replays have bit-identical codes/scales between the original production
ELF and the production candidate. Peak reserved memory is722MiB.
[Comparison](cross-binary.json), [control](control.json), [candidate](production.json).
File-open/access events confirm reads of the candidate ELF in its isolated
validation process. [Production access](production-kernel-access.json),
[paired access](paired-kernel-access.json)

Real checkpoint, frozen layer4 dispatch at8K, rank0 shard: four calls match
independent CPU arithmetic exactly for every INT8 code and FP32 scale;
repeat equality and amax=127 also pass. [Numerical replay](real-weights.json)

The first screen compared capture outputs before replay; its M8 assertion is
discarded. Explicit replay before comparison passes with the same ELF.

## Build and reproduce

```bash
source /workspace/environment/activate.sh
bash tests/glm52/build_gmm1_barrier.sh \
  /workspace/src/vllm-ascend-unified NEW_OUTPUT --compact-experts
```

The builder applies patches rooted at either `csrc/` or its parent and copies
every changed header into the OPC input overlay. Default output remains
`baaa18e24c44695737b414d3196414152f1324caf0087c00cfc102162cf16bc5`.
The production candidate is
`3124d442068fbd09badfce88bafca57f5a7c6bf5940da35bebdea92b8b2f7e7e`;
the paired ±0 binary is
`20561bcca03dc7fcf3fd4880ea8e5ca36dc6343bd256c90dd06c7df5f40a71e5`.
[Build manifest](production-artifacts.sha256)

Server artifacts and commands: a3-194
`/home/shanshan/trt-asc-194/experiments/gmm-compact-20261020`.
The [validation](source/validate.py), [paired screen](source/bench.py) and
[input builder](source/gmm1_inputs.py) use the installed vLLM-Ascend layout.
Production uses no ±0 switch. Keep the overlay explicit until full-model
frozen-teacher, serial/graph consistency and request timing pass.
