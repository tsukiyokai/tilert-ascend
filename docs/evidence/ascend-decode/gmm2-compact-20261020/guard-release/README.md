# Dispersed M32 GMM2 selection

Commit `8e58cff` changes only the opt-in adaptive tile publisher. The original
all-counts≤16 branch remains. An additional branch accepts 256 routed rows,
counts≤32, at least 64 active experts, and at most one eighth with counts>16.
The default build is unchanged. No installed CANN package was replaced.

The [fixed-tile teacher comparison](../teacher-counts/README.md) showed that
a few hot experts need not force the whole layer back to 64×512. The
[concentration screen](../guard-counts/README.md) rejected a hot-fraction-only
rule: 16/32-active cases were slower, while 64-active cases improved.
This empirical boundary expands selection from 40/75 to 75/75 reconstructed
teacher windows. Actual rejected lookup suffixes remain unmeasured.

## Correctness and identity

- [Actual C++ publisher](publisher/result.json): 18 CPU cases validate the
  decision, packed records, padding and same-address wide/fallback/reset.
  CPU storage stubs do not model hardware synchronization.
- [Released kernel](validation/device-result.json): 58 changing-route Graph
  replays match independent CPU MSD arithmetic byte for byte; 6.657s,
  peak reserved 736100352 bytes. Includes newly eligible hot groups and
  same-shape transitions across the one-eighth boundary.
- [ELF access](validation/device-access.json): new SHA
  `a40750d33709c588d284a37b2419b35e41a4536c70e2a6c44f78765147f8c404`.
  The [source comparison](publisher/source-diff.json) confirms that the
  header outside Publish and the compute/synchronization sources are unchanged.

## Complete release comparison

A3-194/trt-asc, CANN9.2.0 B114, shared die0. A is the previous production
adaptive ELF `16e8bea5…`; B is the new release. Four separate processes run
A1/B1/B2/A2, eight weight banks, ten calls per Graph, ten warmups and30
samples. NPU-event time includes input restoration. Profiler off, clocks
unlocked, no samples removed. Peak allocated/reserved memory is
1359755264/1549795328 bytes under a2GiB quota. All80 output positions match
across all four processes, with identical input, scale and route hashes.

| Counts | A1 µs | B1 µs | B2 µs | A2 µs | B1−A1 / B2−A2 µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher layer29 | 95.874 | 84.697 | 78.780 | 97.765 | −11.177 / −18.985 |
| Teacher layer60 | 94.222 | 82.058 | 75.990 | 96.237 | −12.164 / −20.247 |
| Teacher layer21 | 98.576 | 82.995 | 77.366 | 99.487 | −15.581 / −22.121 |
| Teacher layer17, unchanged wide | 80.735 | 85.955 | 80.870 | 82.938 | +5.220 / −2.068 |
| 64 active /8 hot /17 rows | 79.279 | 71.342 | 66.081 | 80.945 | −7.937 / −14.864 |
| 64 active /8 hot /25 rows | 78.974 | 73.751 | 69.378 | 80.729 | −5.223 / −11.351 |
| 32 active /4 hot /32 rows, fallback | 55.709 | 57.929 | 55.110 | 57.763 | +2.220 / −2.653 |
| Shared8, fallback | 40.116 | 44.104 | 40.171 | 41.001 | +3.988 / −0.830 |

Both adjacent comparisons improve for the five newly eligible cases.
All three unchanged-path controls reverse delta sign. Their overhead is
unresolved amid observed process-to-process variation. Two activity checks
rejected launches before NPU execution; later checks allowed the recorded
arms. These are single-die screens using reconstructed teacher counts and
seeded synthetic operands, not whole-model or request-TPOT measurements.

[Raw four-arm evidence and independent audit](screen/README.md).
Remote root: `/workspace/experiments/gmm2-compact-20261020/`.
Keep the overlay opt-in until full-model teacher and actual lookup-session
timing establish its value on the 2ms path.
