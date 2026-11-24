# Production GMM2 guard: four-process screen

The five newly eligible count vectors improve in both neighboring arm
comparisons. The three controls with the same source-predicted tile choice
reverse sign between comparisons. This screen shows gains on the widened
cases and process-to-process drift; it does not establish zero added cost
or a stable regression on the controls.

All values below are event-time medians in µs per call, including INT8 input
restoration. A uses the previous production adaptive ELF; B uses the new
guard release. Processes run A1→B1→B2→A2.

| Count vector | A1 | B1 | B2 | A2 | B1−A1 | B2−A2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher layer 29 | 95.874 | 84.697 | 78.780 | 97.765 | −11.177 | −18.985 |
| Teacher layer 60 | 94.222 | 82.058 | 75.990 | 96.237 | −12.164 | −20.247 |
| Teacher layer 21 | 98.576 | 82.995 | 77.366 | 99.487 | −15.581 | −22.121 |
| Teacher layer 17 | 80.735 | 85.955 | 80.870 | 82.938 | +5.220 | −2.068 |
| Active64 / hot8 / max17 | 79.279 | 71.342 | 66.081 | 80.945 | −7.937 | −14.864 |
| Active64 / hot8 / max25 | 78.974 | 73.751 | 69.378 | 80.729 | −5.223 | −11.351 |
| Active32 / hot4 / max32 | 55.709 | 57.929 | 55.110 | 57.763 | +2.220 | −2.653 |
| Shared eight | 40.116 | 44.104 | 40.171 | 41.001 | +3.988 | −0.830 |

Hot means more than 16 rows. A2 is 0.885–2.203 µs slower than A1 across
the eight cases; B2 is 2.819–6.068 µs faster than B1. Host submission medians
also differ and are retained in the raw records. The observations do not
identify the cause of this drift. All 960 timing samples remain; no outlier
was removed. [Summary](summary.json) and [CPU audit](audit.json) retain the
four arm medians, forward/reverse differences, pooled statistics and raw
ordinal-balanced differences. Samples at the same ordinal in different
processes are not adjacent in time.

## Policy and identity

The new source retains wide tiles when every expert has at most 16 rows.
It additionally allows max32, total256, active≥64 and hot×8≤active. For
these fixtures, source predicates predict one wide case in A and six in B.
Teacher layer 17 remains wide; Active32/hot4 and Shared eight remain compact.
`source_predicted_wide` is a source expectation, not a recorded device
branch observation.

The dispatcher and MSD/Fixpipe header are identical between builds. The
[source difference](predicate-only.diff) is confined to the count publisher
and its final guard. [ELF/source identities](identity.json) bind:

- A: `16e8bea5f250b048da83cbcecac5f0e77f0dd66886a0fc1412144c1d6f47a114`.
- B: `a40750d33709c588d284a37b2419b35e41a4536c70e2a6c44f78765147f8c404`.

Each process has an OPEN/ACCESS receipt for its expected ELF and exits0.
The CPU audit rehashes the binaries, descriptors, benchmark, bank builder
and native source files. This compares complete production ELFs without
the earlier pointer-selected test branch.

## Correctness, memory and load

All 80 corresponding call positions have equal output hashes across all
four arms. Counts, repeated-route, activation and scale hashes also match;
all eight concatenated output hashes match. The device script checks finite
outputs and byte equality between eager and the first captured Graph output.
Raw output tensors were not retained; this audit checks the recorded hashes
and the bound device checks.

Each arm uses eight independent weight allocations, ten calls per graph,
ten warmup replays and 30 timing samples per case, with a 2 GiB allocator
quota. Peak allocated/reserved is 1,359,755,264/1,549,795,328 bytes in every
arm. Profiler is off and clocks are unlocked on shared A3-194 die0.
Each actual launch follows two AIC0/free≥6000 MiB snapshots. Two earlier
preflights reject A1 and B2 on transient AIC activity before probe launch;
their complete records remain in `rejected-01/` and `rejected-02/`.

Four count vectors come from frozen M1 residuals reconstructed with current
M1 kernels; four are bounded synthetic distributions. Activations, weights
and scales are seeded synthetic values using production W4 NZ conversion.
The result concerns a complete GMM2 call on one shared die.

## Reproduce the audit

Server artifacts remain under
`a3-194:trt-asc:/workspace/experiments/gmm2-compact-20261020/guard-release-screen/`.
From that container, the CPU-only audit is:

```bash
python /workspace/experiments/gmm2-compact-20261020/guard-release-screen/audit.py \
  /workspace/experiments/gmm2-compact-20261020/guard-release-screen \
  /tmp/gmm2-guard-release-audit.json
```

[Benchmark](bench.py), [arm launcher](run-arm.sh), [initial launch](launch-abba.py),
[resumed launch](resume-abba.py), [CPU import](cpu-import.json) and
[archive hashes](manifest.json) preserve the executed protocol. The companion
`../validation/` evidence is maintained separately.
