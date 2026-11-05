# GMM1 K512 screen: rejected

A3 a3-194 die15, CANN9.2.0 B114, E256/K6144/N256. Three arms share one
ELF: original N256/K256, N128/K256, and N128/K512. BaseM128 and order1
remain fixed. At N128, K512 changes stepKa/stepKb4→2 and depthA1/depthB1
8→4: A/B L1 stay131072B each, six outer transfers remain, and inner K
iterations fall24→12. GM bytes do not decrease.

INT32 accumulators cannot overflow for these INT4 inputs; the complete K
reduction still precedes one Fixpipe conversion. L0A/B double buffers each
grow to65536B. The two L1 slots alternate across six outer iterations.
[CPU arithmetic](cpu-n128-k512-contract.json),
[independent audit](independent-audit/result.json),
[cache slots](independent-audit/cache-contract.json).

All24 changing-route Graph cases match codes/scales bit-for-bit. Timing:
32 calls/Graph,5 warmups,10 ABBA pairs, two weight banks, profiler off,
unlocked clocks under shared load; peak reserved898MiB.

| Case | N256/K256 µs | N128/K512 µs | Paired delta µs | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| M1 random | 45.081 | 41.298 | −3.699 | 10/10 |
| M8 random | 85.242 | 83.499 | −1.721 | 10/10 |
| M32 random | 141.969 | 142.455 | +0.514 | 0/10 |
| M32 shared8 | 56.638 | 52.857 | −3.763 | 10/10 |

At fixed N128, M32 random saves4.665µs, but it remains slower than the
original N256 arm. Close this GMM1 tiling branch without a production
switch or further screening. Actual lookup expert unions must precede
another policy decision. [Summary](nsplit-k512-measured-summary.json),
[raw pairs](nsplit-k512-bench.json), [correctness](nsplit-k512-validate.json).

Loaded ELF SHA256 `f056a35df8cbbd937aec5584d09d8c040f07b53031cd7fd3057aeb63424844f5`:
[runtime access](nsplit-k512-bench-access.json),
[patch](nsplit-k512-three-arm.patch). Server artifacts remain under
`/workspace/experiments/gmm-compact-20261020/kloop-audit`.
