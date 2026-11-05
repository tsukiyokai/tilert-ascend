# Frozen overlapping target calls

Runner `24049d1` replays the first two measured M8 target calls from the
historical chain. Positions are8192–8199 and8195–8202; logical commits retain
8195 and8196. Rejected physical KV remains available for the next overwrite.
Only rows0/1/2/8 have trustworthy teacher labels:382/262/576/2524.

[CPU checks](cpu/result.json) cover48 contracts, including bad read lengths,
accepted-prefix mutation and rejected-tail clearing. On a3-194/trt-asc,
CANN9.2/B114, the first8 layers pass on all16 ranks under a2GiB/die Torch
quota. The [independent audit](smoke-8/audit.json) checks128 artifact hashes,
layer handoffs, causal indices, actual SFA/indexer lengths and block tables.
Peak layer allocation is942,355,456 bytes/die. No final logits or timing is
claimed by this8-layer run.

[Trusted-row comparison](smoke-8/teacher-comparison.json): all8 layers have
bit-identical hidden, residual and new KV against the accepted M32 teacher.
Index order first differs at layer1/row1; selected sets remain identical.

Full78-layer replay completed on2026-10-20, exit0. All16 ranks pass4/4
trusted labels. The [raw audit](full-78/audit.json) checks1,264 artifacts;
all78 layers' trusted hidden/residual/new KV match accepted M32 exactly.
Selected sets agree throughout; only their order differs. Final normalized
hidden and FP32 logits are [bit identical](full-78/final-comparison.json)
to the four corresponding M32 rows and across ranks. The old divergent
position produces2524, with6500 second and margin0.0026168823.
Raw output is under
`a3-194:trt-asc:/workspace/experiments/m32-full-teacher-20261020/two-rounds/full-78/`.
Next isolate the [three historical norm/quant fusions](../norm-quant-fusion/README.md).

This is layer-major eager execution. It does not cover compiled Graph,
conditioned MTP, request reset or their asynchronous buffer lifetimes.
Finite new KV is not proof that every cache byte was written.
[Launcher](run.sh), [raw auditor](audit.py),
[teacher comparison](compare_teacher.py),
[GMM receipt](smoke-8/gmm-access.json), [SFA receipt](smoke-8/sfa-access.json).
