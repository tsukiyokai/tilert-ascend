# GMM1 full-A screen: rejected

A3 a3-194 die15, CANN9.2.0 B114, INT8×NZ-INT4 MSD,
E256/K6144/N256. One experimental ELF compares compact128, compact64
with streamed A, and compact64 with all K of A resident in L1.
BaseK256, arithmetic and pre/post stages are unchanged. Group buffers have
identical values; their aligned address tags select the three arms.

The full-A choice changes six disjoint K transfers into one; total A bytes
are unchanged. B still streams in six batches and Cube still executes24
K blocks. This tests transfer scheduling rather than reduced HBM traffic.
[CPU resource/address checks](cpu-validity.json), [API tiling](cpu-tiling.json).

All24 changing-route Graph cases match codes/scales bit-for-bit across the
three arms. Timing uses32 calls/Graph,5 warmups,10 ABBA pairs per comparison,
no profiler, unlocked clocks and a resident foreign job. Peak reserved is
898MiB. These are synthetic single-die screening results.

| Case | Compact128 µs | Full-A64 µs | Paired delta µs | Full-A wins |
| --- | ---: | ---: | ---: | ---: |
| M1 random | 46.382 | 46.873 | +0.543 | 1/10 |
| M8 random | 84.072 | 88.256 | +4.215 | 0/10 |
| M32 random | 144.541 | 145.951 | +1.348 | 1/10 |
| M32 shared8 | 59.747 | 60.733 | +1.505 | 2/10 |

Against streamed-A64, full-A paired deltas are −0.621,+2.587,+2.083,+1.471µs
respectively. No complete-package benefit survives; retain compact128 and
do not add a production flag. [Raw timing](bench.json), [correctness](validate.json).

Loaded ELF SHA256 `7f0ed40760a7a36bc738232d1b1bc1ee6defdb3d8fff167069048f7a2794c291`:
[key2/24-block selection](launch-audit.json), [runtime open/access](bench-access.json),
[source identities](source-identities.json), [experimental patch](full-a-three-arm.patch).
Server artifacts remain under `/workspace/experiments/gmm-compact-20261020/kloop-audit`.
