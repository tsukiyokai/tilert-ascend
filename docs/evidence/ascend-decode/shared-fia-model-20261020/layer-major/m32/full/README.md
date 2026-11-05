# Full M32 teacher validation

A3-194, CANN 9.2/B114, source `8e58cff`, TP16, W4A8C8, one frozen
8192-token prefix. All 78 layers execute one M32 teacher chunk, with FP32
TP accumulation and FP32 final head. The SFA run completed on 2026-10-20
at 07:33:28 CST, exit 0.

The independent [CPU audit](sfa-audit.json) verifies 32/32 frozen successor
labels on every rank and byte-identical final FP32 logits across all 16
ranks. It checks 1,264 artifact hashes, 1,232 layer handoffs, causal unique
indices and producer reuse. Peak layer Torch allocation is 942,552,064 bytes
per die under a 2 GiB quota. This validates one teacher in eager layer-major
execution; token-major Graph/reset, eight prompts and request TPOT remain.

[Rank-0 receipt](sfa-rank-0-receipt.json), [all-rank summary](sfa-summary.json),
[mapped libraries](sfa-loaded-libs.json) and the [GMM](sfa-gmm-access.json) /
[SFA](sfa-device-access.json) file-access receipts bind the run. File events
do not attribute execution to a particular rank. Full reports and tensors
remain in `a3-194:trt-asc:/workspace/experiments/m32-full-teacher-20261020/`.

The host remains shared with resident Ray workers. Two physical-HBM samples
require at least 6000 MiB free on every die before launch; utilization is
recorded, not used as a zero-activity requirement for this bounded functional
run. Torch's quota excludes runtime and HCCL allocations. No timing is taken.

The paired shared-FIA run completed at08:01 CST with30/32 labels on every
rank; queries3 and21 fail. Its409 source files match the SFA snapshot byte
for byte; only the attention backend changes, with separate cache/output.
The [independent raw audit](shared-result/paired-diagnostic.json) validates
2,528 artifacts across both arms, all layer handoffs, causal unique indices,
prefix checks and identical logits across ranks within each arm. Only the
teacher gate fails. Peak shared-FIA layer allocation is942,589,952 bytes/die.

| Query | Frozen label | SFA top1 / runner-up / margin | Shared-FIA top1 / runner-up / margin |
| --- | ---: | --- | --- |
| 3 | 2524 | 2524 /6500 /0.002617 | 3797 /2524 /0.031149 |
| 21 | 13 | 13 /26 /0.071281 | 26 /13 /0.032093 |

Rank0 merged-hidden NRMSE grows from0.00364 at layer0 to0.09949 at31 and
0.16866 at47; it ends at0.12303 before final norm,0.15993 after norm.
[All78 layers](shared-result/layer-curves.csv), [curves](shared-result/layer-curves.svg)
and per-query diagnostics are
retained. This is accumulated divergence, with the teacher gate unchanged.
A [fixed-input replay](../../../fia-row-correction/README.md) at layers66–73
finds FIA invalid-row correction0→2 bit identical across524,288 outputs.
The flag-only route is closed for those inputs; the teacher gate still fails.

Query3 is the same logical prefix as the historical chain's first wrong
output6500. Its SFA margin0.002617 makes arithmetic a competing explanation
alongside replay/cache state; the two-round replay must save FP32 logits.
No whole-request speed is inferred from this functional experiment.

Against the historical M1 run, [582/660 boundary/cache pairs](m1-comparison/README.md)
are bit identical. Hidden/residual agree through layer67; layer68/query13
is the first output difference. Its inputs and new KV agree, while two
selected IDs exchange order from producer66 onward. Final normalized NRMSE
is0.0239812; this diagnostic does not invalidate the 32/32 labels or prove
serial equivalence beyond them. The independent layer58/query16 INT8 cache
difference cannot cause the earlier query13 through causal attention.
A [fixed-Q/KV four-call experiment](layer68-first-difference/README.md)
can separate index order from query-width arithmetic if needed.

The checkpoint directory is on the root filesystem, but its actual shard
symlinks resolve into a read-only NFS mount. Bounded server-CPU prefetch
warms at most eight layers ahead with four 8 MiB buffers. Loading and
prefetch time are outside decode performance claims.

## Reproduce

The preserved [SFA launch](run-sfa.sh) and [shared-FIA launch](run-shared-fia.sh)
use the original source snapshots and output paths. The independent auditor
is [audit_sfa78.py](audit_sfa78.py), with the hash-bound [helper](audit.py).
Run it inside the recorded container against the preserved raw output;
the report records its exact source hashes.
