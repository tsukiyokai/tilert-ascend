# GMM2 concentration boundary

A3-194/trt-asc, CANN 9.2.0 B114, shared die0. Same paired ELF
`ae1a0296…` as the [teacher-count screen](../teacher-counts/README.md).
Both arms use compact expert metadata and identical synchronization; the
count-pointer selector changes 64×512 to 32×1024 tiles. No installed
package was changed.

Eight synthetic M32/top8 distributions each contain 256 routed rows.
Server CPU checks cover every output exactly once and preserve MSD pairs.
The device comparison uses eight weight banks, ten restored-input calls
per Graph, ten warmup pairs, and 30 ABBA samples per case. Timing uses NPU
events, profiler off, clocks unlocked, with no samples discarded. All 80
complete output pairs are byte-exact. Peak allocated/reserved memory is
1,359,755,264 / 1,784,676,352 bytes under a 2 GiB quota.

| Active / hot / hot rows | Compact p50 µs | Wide p50 µs | Paired delta µs | Wide wins |
| --- | ---: | ---: | ---: | ---: |
| 16 / 2 / 32 | 46.272 | 51.685 | +5.425 | 0/30 |
| 32 / 4 / 17 | 55.166 | 60.740 | +5.585 | 0/30 |
| 32 / 4 / 32 | 54.882 | 59.333 | +4.469 | 0/30 |
| 64 / 8 / 17 | 78.850 | 67.942 | −10.850 | 30/30 |
| 64 / 8 / 25 | 78.389 | 70.301 | −8.039 | 30/30 |
| 64 / 9 / 17 | 78.979 | 68.363 | −10.591 | 30/30 |
| 32 / 8 / 17 | 54.910 | 60.361 | +5.480 | 0/30 |
| 16 / 8 / 30 | 44.559 | 51.962 | +7.576 | 0/30 |

Hot means more than 16 rows. With counts at most 32, tile count is
12A versus 6(A+H), and logical W4 references are 393216A versus
393216(A+H) bytes. A hot fraction of one eighth bounds extra references
to 12.5%, but does not ensure a speedup: the 16/32-active cases disprove
that inference. Actual HBM traffic was not measured.

The candidate retains the previous all-counts≤16 path and extends it only
to 256 total rows, counts≤32, at least 64 active experts, and H/A≤1/8.
This is an empirical selection rule for the opt-in overlay. The threshold
is bracketed by the measured 32/64-active cases; it is not an optimum.
The 64/9 case deliberately falls back despite its fixed-wide improvement.
Full release timing must include the adaptive publisher and header read.
These single-die measurements do not establish request TPOT.

Raw samples, output hashes, source, preflight, exit status and ELF access
receipt are beside this file. Remote root:
`/workspace/experiments/gmm2-compact-20261020/guard-counts-screen`.
