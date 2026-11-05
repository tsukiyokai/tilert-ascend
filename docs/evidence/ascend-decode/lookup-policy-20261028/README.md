# Lookup policy and CPU cost

`bench_lookup.py` now passes the same shortest and longest tail bounds to
both implementations and records them. Its reverse scan previously fixed
the longest tail at 8 while the current `LookupDrafter` defaults to 32.
For `A = [1, ..., 32]`, the context `A + [111] + A[-8:] + [222] + A`
proposes 111 at length 32 and 222 at length 8. Three CPU regression checks
cover this counterexample, both bounds at widths 2/8/32 and the real CLI.
Historical records retain their measured source identities and policies.

## Packed reverse scan

[Probe](../../../../experiments/lookup-index/rfind_probe.py) uses native
uint32 tokens and C `bytearray.rfind`, validates token alignment, and keeps
earlier aligned matches that overlap an unaligned byte hit. It preserves
longest-tail then latest-occurrence ordering on the tested inputs. The
independent Python reverse scan is outside the timed arms.

Local arm64 macOS, Python 3.11.15; four synthetic 8192-token prompts and
1024-token outputs; width32, shortest2/longest32; two warmups and 30
alternating paired samples per workload. No CPU affinity or clock locking.
Setup includes the common first output; steady cost sums proposal and
commit calls. These costs exclude target inference, orchestration and I/O.

| Workload | Rounds | Current setup p50, ms | rfind setup p50, ms | Current steady p50, ms | rfind steady p50, ms | Current total p50, ms | rfind total p50, ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact copy | 32 | 42.469 | 0.304 | 5.699 | 0.092 | 48.173 | 0.397 |
| Copy with 51 edits | 173 | 42.399 | 0.301 | 7.384 | 49.209 | 49.929 | 49.515 |
| Ambiguous bigram | 1023 | 49.022 | 0.317 | 15.774 | 439.042 | 64.830 | 439.356 |
| No match | 1023 | 48.199 | 0.307 | 15.348 | 249.347 | 63.560 | 249.653 |

The four workloads and all 240 measured samples match the independent
oracle's complete proposal/acceptance signatures. Another 1600 incremental
checks and two byte-alignment cases pass. Every raw duration sum, summary
quantile and measured source hash was checked after the run. See
[summary](rfind-summary.json) and [full raw samples](rfind-cpu.json.gz).

The pure-copy gain is large, but short ambiguous tails and absent matches
repeat scans over the context. This implementation remains an experiment;
the recorded current arm is the hash index at Git `1ef8a8d`. The subsequent
[radix index](../lookup-radix-20261028/README.md) has its own paired records.
No NPU or TPOT result follows from these CPU measurements. Reproduce this
historical comparison from `1ef8a8d`.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 tests/glm52/check_lookup_benchmark.py
PYTHONDONTWRITEBYTECODE=1 python3.11 experiments/lookup-index/rfind_probe.py \
  --root "$PWD" --output /tmp/lookup-rfind-cpu.json
```
