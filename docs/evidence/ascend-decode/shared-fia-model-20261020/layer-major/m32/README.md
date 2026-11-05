# M32 with one resident decoder layer

A3-194, CANN 9.2/B114, TP16, frozen 8K prefix and 32 teacher inputs.
The runner executes M1 or one complete M32 chunk through each layer,
including embedding and the final head. Hidden and residual remain separate;
index consumers restore all query rows. One shared-FIA mask owner survives
layer replacement. M32 requires all 32 inputs; intermediate chunk widths
would require separate mask ownership and are not exposed here.

The [CPU schedule check](cpu/result.json) executes the actual runner and
binder with substituted device arithmetic. It checks producer/consumer
handoff, causal slots, row order, final finite gates and argument rejection.

The [device audit](sfa-8-audit.json) verifies eight real M32/SFA layers on
all sixteen ranks: three dense layers, five MoE layers, and the C8 index
producer at layer 6 followed by its consumer at layer 7. Hidden, residual
and indices are bit identical across ranks. All prefix, ownership, causal
index and FP32 reduction checks pass. Peak Torch allocation is 942,552,064
bytes/die under a 2 GiB quota. This is partial execution, with no final
teacher labels or TPOT measurement. External Ray workers remained resident.

[Mapped libraries](sfa-8-loaded-libs.json) bind the process host libraries;
[GMM](sfa-8-gmm-access.json) and [SFA](sfa-8-sfa-access.json) receipts record
device-binary file accesses. File events do not identify the executing rank.
Source snapshots, report hashes and raw tensors remain under
`a3-194:trt-asc:/workspace/experiments/layer-major-m32-20261020/`.

The [old M1 comparison](m1-comparison/README.md) checks rank0 across
layers0–7: all68 boundary/cache tensor pairs are bit identical, including
3,145,728 hidden/residual BF16 values. Selected-token sets agree for
every query; index order differs at layers1–5. The server CPU rerun
reproduces the complete result JSON byte for byte. Runtime/source
differences are recorded with the comparison.

The subsequent [full78-layer SFA run](full/README.md) passes all32 frozen
labels with an FP32 head on16 ranks. Its shared-FIA counterpart completes
30/32; queries3 and21 fail. The full paired audit retains this failure.
The earlier eight-layer shared-FIA attempts stopped at activity admission;
the full run uses bounded shared-host functional admission and no timing.

## Faster validation loading

The [actual loader check](loader/loader-actual-result.json) reads the same
36 checkpoint tensors, 58,097,664 bytes, with one handle per shard rather
than one per tensor. Single-thread CPU ABBA times are
136.475/15.968/16.866/135.243 ms. Names, bytes, shape, dtype and stride agree;
returned tensors own their storage. Normal completion, early close and a
failing model consumer release all handles and mappings. This measures
loading only; neither full-layer loading time nor decode improves by an
established factor yet. The device result above predates this loader change.

An initial committed-loader retry stopped at the activity gate. A
[12-sample follow-up](activity-followup.json) records external activity peaks
of6–7% between zero samples. The full78-layer teacher result now covers this
loader; its effect on full-model loading time remains unmeasured.
