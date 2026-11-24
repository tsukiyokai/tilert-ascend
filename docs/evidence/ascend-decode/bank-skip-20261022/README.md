# The vendor kernel already skips an empty expert, so the narrowing is void

The bank narrowing was this project's largest lever and its only measured one.
It does not exist. The grouped matmul charges nothing for an expert whose group
is empty, so the engine already pays only for the experts a round touches, and
subsetting the weight tensor to those experts removes a cost that was never
spent.

## The measurement

Three arms share one routing total of 96 rows over a sixteen-expert bank at the
checkpoint's down-projection shape, `[16, 6144, 2048]` bf16 at 402 MB. Seven
repeats of 50 iterations each, timed with NPU events on one idle die of a3-194.

| arm | weight tensor | nonzero groups | median us |
| --- | --- | ---: | ---: |
| spread | all sixteen | 16 | 250.53 |
| concentrated | all sixteen | 5 | 87.23 |
| narrowed | the five | 5 | 87.10 |

`concentrated` and `narrowed` differ by 0.13 microseconds, or 0.15 percent,
against a spread between repeats of about 0.6. Keeping eleven unused experts in
the tensor costs nothing measurable. What the narrowing would have bought is
already in hand.

Both arms sit at bandwidth: `spread` reads 402 MB in 250.53 us for 1607 GB/s and
`concentrated` reads 126 MB in 87.23 for 1443 GB/s, against roughly 1600 GB/s a
die. So cost follows the bytes actually read, which follows the nonzero group
count.

## Why the earlier probe looked like a lever

The 2026-10-21 probe varied the expert count by slicing the weight tensor, and
gave every remaining expert a nonzero count. It therefore moved the tensor size
and the active group count together and attributed the whole effect to the
tensor. The `spread`-to-`concentrated` ratio of 2.87 is that probe's curve; it
is real, and it is not a lever, because the engine is already at the
concentrated point. A round touches 5.07 of a rank's sixteen experts, which the
routing fixes and the routed/shared fixture's own 81 distinct of 256 confirms.

## The kernels say the same thing

The engine's own fused gmm1,
`grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h`,
computes each group's row count and leaves the loop body before touching the
weights:

    SetMNConfig(splitValue, mnConfig);
    if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
        continue;
    }
    mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);

`grouped_matmul_antiquant_a8w4_msd.h:273` has the same `continue`, as do
`grouped_matmul_antiquant_a8w4.h`, `..._a8w4_msd_new.h`, `..._a16w8_msd.h` and
the generic `grouped_matmul_kernel.h`. This project's own hand-written routed
kernel does it too, at `routed_int4_kernel.cpp` with
`if (counts.GetValue(expert) == 0) continue;`.

That last point also dissolves a coincidence. The native whole layer measured
574.15 microseconds and the vendor whole layer 575.22, and those were read as
two independent paths agreeing. They agree because on this term they are the
same mechanism: both skip the empty expert, so neither can be ahead.

## Scope

The timed arms use the plain bf16 `npu_grouped_matmul`, not the engine's W4A8
NZ fused variant. The skip is read from that variant's source and measured in
its family. A paired device comparison on the deployed operator would close the
gap, but the burden was on the lever and it has not been met.

[`probe.py`](probe.py) is the measurement and [`result.json`](result.json) the
seven repeats.
