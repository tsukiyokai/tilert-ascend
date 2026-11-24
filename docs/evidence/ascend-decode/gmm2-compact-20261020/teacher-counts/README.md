# GMM2 fixed-wide results on teacher count vectors

Forced-wide improves the three sampled fallback count vectors by
15.88–20.03 µs per call in paired medians. Shared eight regresses by
5.49 µs, with all 30 pairs losing. This supports a bounded guard experiment
and rejects always-wide for M32.

| Count vector | Active / hot / max | Tasks compact → wide | Arm medians, µs | Paired wide−compact, µs | Wide wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher layer 29 | 89 / 1 / 17 | 1068 → 540 | 101.073 → 85.645 | −15.884 | 30/30 |
| Teacher layer 60 | 88 / 2 / 30 | 1056 → 540 | 99.265 → 82.258 | −16.546 | 28/30 |
| Teacher layer 21 | 94 / 3 / 19 | 1128 → 582 | 100.560 → 80.497 | −20.029 | 30/30 |
| Teacher layer 17, all-small | 101 / 0 / 16 | 1212 → 606 | 105.376 → 83.469 | −21.922 | 30/30 |
| Shared eight | 8 / 8 / 32 | 96 → 96 | 46.619 → 52.214 | +5.493 | 0/30 |

Hot means more than 16 rows. Each ABBA sample computes
`mean(wide₁, wide₂) − mean(compact₁, compact₂)`. All 150 paired samples and
their summary statistics were independently recomputed from
[raw records](device-result.json); see [audit](audit.json).

## Geometry, traffic and latency

For active count `c≤32`, compact 64×512 uses 12 tasks; wide 32×1024 uses
six for `c≤16`, and 12 otherwise. The three fallback vectors nearly halve
task count while adding 1.12%, 2.27% and 3.19% logical W4 reads. Their arm
median reductions are 15.26%, 17.13% and 19.95%. These are separate measures:
task reduction does not determine latency, and logical reads do not measure
HBM traffic. The source changes B's L2 hint when a second M tile appears.

Across the complete reconstructed 75-layer prior, forced-wide adds 49
expert matrix reads: 18.375 MiB, or 0.6575% of total logical W4 bytes
(1.5322% within the 35 fallback layers). Scale/bias traffic is excluded.
Shared eight has no task reduction and doubles logical W4 reads; its
measured 12.0% arm-median regression is the concrete adverse case.

The smallest further falsification is a legal top-eight distribution with
eight experts at 30 rows and eight at two rows. It has 16 active experts,
eight hot experts and max30: tasks fall 192→144 while logical W4 reads rise
50%. The [CPU construction](audit.py) checks 32 unique-eight selections.
This unmeasured case can challenge a maximum-count-only relaxation that
passes the measured layer-60 vector. Reuse the same ELF and compare this
case with layer 60 before choosing a production predicate.

## Identity and correctness

Both arms use ELF SHA256
`ae1a0296f8f051c0701a21f84d83769f902c6fe6c7321a08f7c262745483922d`,
with compact/forced-wide selected by count-pointer bit 5. The isolated
source preserves the original barriers and applies the production
[temporary-buffer declaration fix](buffer-declaration.patch).
[Access receipt](device-access.json) records OPEN/ACCESS and exit0. The
CPU audit rehashes all 23 build inputs/artifacts and the ELF descriptor.
The older adaptive pointer test short-circuits a header read in one arm;
its timings are not used here.

All 50 corresponding output hashes match. The bound device script also
checks byte equality and finite outputs; the five case hashes are distinct
and differ from the all-zero tensor hash. Raw output tensors were not
retained, so the CPU audit validates the digests and bound checks rather
than re-reading output bytes.

## Scope and reproduction

This is A3-194 die0, eight weight allocations, ten calls per graph,
ten warmup pairs and 30 ABBA samples per case. Event timing includes input
restoration; profiler is off and clocks are unlocked. The host remains
shared. [Two preflight samples](screen-01-preflight.json) show AIC0 and
6518 MiB free. Peak allocated/reserved is 1,359,755,264/1,784,676,352 bytes,
under a 2 GiB allocator quota.

Count vectors come from frozen rank-0 M1 residuals reconstructed with
current M1 kernels. They are not saved historical routes, live M32 routes
or rejected lookup tails. Weights, activations and scales are seeded
synthetic tensors with production W4 NZ conversion. This screen isolates
tile choice, without a complete adaptive-policy or request-TPOT result.

Artifacts remain at
`a3-194:trt-asc:/workspace/experiments/gmm2-compact-20261020/teacher-counts-screen/`.
CPU audit command from that container:

```bash
python /workspace/experiments/gmm2-compact-20261020/teacher-counts-screen/audit.py \
  /workspace/experiments/gmm2-compact-20261020/teacher-counts-screen \
  /tmp/gmm2-teacher-counts-audit.json
```

[Benchmark](bench_counts.py), [launch](launch.py),
[preparation identity](build-manifest.json) and [archive hashes](manifest.json)
retain the executable protocol and provenance. No production source was
changed by this experiment.
