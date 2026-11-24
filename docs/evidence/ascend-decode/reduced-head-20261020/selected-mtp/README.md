# Selected conditioned MTP head

Decision: close implementation work for this round. The old width8 conditioned phase floor is 2.080 ms out of 52.857 ms per round (3.94%). Its entire head accounts for a median 219.986 µs of device tasks in the saved seven-rank profile. A selected M1 head would remain, after host acceptance processing. These records do not measure removable latency or current M4/M32 cost.

The split is structurally possible: compute all conditioned hidden/KV rows before the existing host read, resolve acceptance, then project row `committed-1`. Preserve the installed FP32 head and logits processor. Recycle hidden is already normalized; passing it through `compute_logits` would normalize it twice. The scalar counterexample changes top1 from 0 to 1. A separate head graph would require stable input storage and its own lifecycle; the existing M1 graph also writes KV.

Files:

- `budget.json`: phase/profile values, calculation, source hashes and evidence limits.
- `trace-summary.json`, `trace-head-tails.json`: unchanged existing trace-accounting receipts.
- `counterexample.json`: server CPU scalar semantic result; no CANN simulation.
- `audit.md`: exact source anchors and the minimum interface if reopened.

No product code or NPU experiment was added.
