# GMM1 N splitting: no general gain

A3 a3-194 die15, CANN9.2.0 B114, E256/K6144/N256. One experimental
ELF compares N256, N128 and N64 tiles, retaining baseM128/baseK256,
order1, stepKa/stepKb4 and depthA1/depthB1=8. Full output stride256,
MSD row pairs and the final FIX completion barrier remain unchanged.

All24 changing-route Graph cases match codes/scales bit-for-bit. Timing
uses32 calls/Graph,5 warmups and10 ABBA pairs, two weight banks, no profiler
and unlocked clocks under shared load. Peak reserved is898MiB.

| Case | N256 µs | N128 µs | Paired N128 delta µs | N128 wins | Paired N64 delta µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 random | 48.486 | 45.259 | −3.216 | 10/10 | +0.674 |
| M8 random | 87.373 | 84.223 | −3.144 | 10/10 | +14.275 |
| M32 random | 137.001 | 143.423 | +6.490 | 0/10 | +56.250 |
| M32 shared8 | 56.227 | 52.198 | −4.064 | 10/10 | +0.672 |

N64 loses every pair in all four cases. N128 helps small expert unions but
regresses M32 with152–174 active experts; no production policy is selected.
[Measured summary](nsplit-measured-summary.json), [raw pairs](nsplit-bench.json),
[correctness](nsplit-validate.json), [CPU address coverage](cpu-nsplit-coverage.json).

Loaded ELF SHA256 `d3f5a551b485d87025d5b9000c1255d945a72a60a9b4d09f5e91d8d874a08c45`:
[key2/24-block selection](nsplit-launch-audit.json),
[runtime open/access](nsplit-bench-access.json),
[experimental patch](nsplit-three-arm.patch).
The contiguous conversion retains the address tags selecting each arm;
[post-conversion descriptors](nsplit-contiguous-witness.json) confirm all three.
Server artifacts remain under `/workspace/experiments/gmm-compact-20261020/kloop-audit`.
