# M16 shared FIA

The existing shared-KV implementation supports the intermediate width16.
The runtime width gate now permits8/16/32; single-token calls retain SFA.

Server CPU constructs a frozen four-layer fixture at M16/C9344, four heads,
top-2048, with independent Q/KV per layer. A/B lengths are8208/8207; causal
thresholds follow each query's logical position. FP64 dense-mask versus
selected-KV maximum difference is2.50e−16. Future-row mutation leaves the
reference unchanged. [CPU contract](contract/result.json).

a3-194 die15, CANN9.2.0 B114, shared residency, unlocked clocks; two launch
snapshots show AIC0% and6936MiB headroom. The actual production binder matches
four rebuilt FIA calls in12/12 A/B/reset comparisons. Eager versus Graph
A/B/reset matches24/24. SFA versus FIA is a separate diagnostic:
NRMSE0.002070–0.002171, max_abs0.001953125. These fixtures do not establish
78-layer teacher acceptance.

Thirty ABBA pairs, one four-layer Graph replay/sample, profiler off; inputs
are hot. Each pair averages its two samples per arm before the reported
medians. Raw samples and the actual loaded library/ELF identities are retained.

| Four-layer group | SFA | Shared FIA | Paired difference | Wins |
| --- | --- | --- | --- | --- |
| Device event (µs) | 373.365 | 328.620 | −44.190 | 29/30 |
| Wall with events/sync (µs) | 443.856 | 397.591 | −45.499 | 29/30 |

Median paired event ratio is0.88135. The pooled raw event medians are
372.820/329.530µs. Peak allocated/reserved memory is128.81/234MiB under a
512MiB Torch quota;604 attention calls complete, process exits0, and die15
headroom returns to6936MiB. No whole-model TPOT is measured.

The modified constructor also passes eight checks using the real installed
vLLM config with only worker/device allocation replaced: M16, INT8 indexer,
max_num_batched_tokens16, NPU uninitialized. Three existing mask/fallback
checks pass on server CPU. [Constructor](integration/config-check.json),
[mask checks](integration/mask-check.log).

[Measurement summary](measured-summary.json), [120 raw samples](device/result.json),
[launch](device/launch-audit.json), [screen](screen.py),
[CPU fixture generator](cpu_contract.py), [inventory](inventory/summary.json).
