# Lookup contracts and CPU cost

## State and acceptance

Lookup owns the target cache only. Its context is prompt plus committed output,
including the pending token. MTP KV stays at the injected P−1 length. The round
state and measurement record now report that actual ownership; the previous
validator would reject a correct lookup request by requiring MTP KV to advance.

Replay includes pending-token padding and clips the final correction/bonus to
the requested output count. CPU checks exercise the actual rank engine with
independent scalar target/draft models, serial output and KV-prefix comparison,
false cache-length rejection, and longest/latest lookup ordering. Incremental
index checks compare 1280 proposals with an independent backward scan, including
empty commits and multi-token commits.

## Server CPU measurement

A3-84 `trt-asc`, Python 3.11.12; one 8192+1024 trajectory; two warmups followed by
30 paired AB/BA samples per width. The timer includes proposal and commit work;
the total also includes request-local index construction. Target inference is
excluded. Every pair has identical proposal/acceptance signatures.

| Width | Rounds | Scan total p50, ms | Indexed total p50, ms | Index setup p50, ms | Indexed query p50, µs |
| --- | --- | --- | --- | --- | --- |
| 8 | 253 | 1031.537 | 20.591 | 17.776 | 2.090 |
| 16 | 198 | 1034.633 | 20.734 | 18.023 | 2.201 |
| 32 | 174 | 1029.860 | 20.895 | 18.228 | 2.250 |

[Raw paired records](cpu-paired.json) bind inputs and measured source hashes.
The frozen server source is under `/workspace/offline/lookup-cpu-20261019/`.
Later local docstring/line-wrap changes do not alter the measured algorithm.
This is a CPU proposer improvement; no end-to-end TPOT improvement is claimed.

## One-token fallback

With longest suffix 8 unchanged, shortest 2→1 changes width 32 acceptance from
5.8793 to6.9592 (174→147 rounds). [Policy summary](policy-summary.json) binds the
input, source and raw audit hashes. Proposals use only the visible prefix;
future recorded output scores the fixed proposal afterwards. The actual indexed
implementation independently reproduces 228/171/147 rounds against the reverse
scan ([three paired samples per width](unigram-paired.json)); its total CPU
work is21.98–22.54ms per request, including construction. Wrong branches
can change the MoE expert union, which this replay cannot measure.

The prepared device comparison is `/workspace/experiments/context-copy-20261019/go.sh`;
it has not launched. Each arm has separate output/cache/ports and checks occupancy.
The previous waiting queue was stopped before any model launch. This source
snapshot includes pinned staging throughout and matched MTP controls. Compare
complete emitted tokens before accepting latency results.

A causal width policy was screened on CPU: use32/16 only when the current
8-token suffix has at least31/15 known continuation tokens, otherwise8. It
requires175 rounds and2240 verification rows, against fixed32's147 rounds and
4704 rows. Under shared bucket costs it wins only if
`120T8+30T16 < 122T32`; rejected-branch expert costs remain unknown. The rule
still selects32 for seven rounds with no correct draft. It is not enabled.
[Policy record and concrete counterexample](adaptive-summary.json).
