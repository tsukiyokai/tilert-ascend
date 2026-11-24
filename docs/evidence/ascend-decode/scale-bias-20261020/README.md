# The term both my kernel and my reference leave out

2026-10-20. Read off the checkpoint on the CPU, with every machine's dies
held by another tenant.

## What the checkpoint says

`quant_model_description.json` carries 235,935 entries, of which 117,311 name
a `weight_offset` or a `scale_bias`. Every expert projection is marked
`W4A8_DYNAMIC` and has four tensors, not two:

    model.layers.30.mlp.experts.0.gate_proj.weight         W4A8_DYNAMIC
    model.layers.30.mlp.experts.0.gate_proj.weight_scale   W4A8_DYNAMIC
    model.layers.30.mlp.experts.0.gate_proj.weight_offset  W4A8_DYNAMIC
    model.layers.30.mlp.experts.0.gate_proj.scale_bias     W4A8_DYNAMIC

And the bias is not zero. Reading the shard directly:

    model.layers.3.mlp.experts.0.gate_proj.scale_bias F32 [2048, 1]
      first 16 values, 16 nonzero:
      [-0.001393, 0.076904, 0.000794, 0.017286, 0.001723, 0.025291, ...]

## What it does and does not make of the zero-mismatch results

My first reading was that the kernels and the host reference shared the
omission, so their agreement was vacuous. That was wrong, and the correction
is the useful part.

`reference_weights.py` carries the term and documents the convention:
`"w4_scale_bias": "Checkpoint correction added once after 16H+L"`, read as
`prefix + ".scale_bias"`, and applied in `w4_msd_product` as

    result = 16 * high_scaled + low_scaled
    result = result + correction
    return (result * activation_scale).to(output_dtype)

-- after the MSD recombination, before the activation scale. The INT4 kernel
carries it too: `gu_bias` and `down_bias` are in its launch signature and it
loads them. So the INT4 zero-mismatch results stand, and the convention never
needed a device run to settle; it was in the tree.

What lacks the term is the INT8 kernel alone, and the binding I had just
written called it. That is the finding: the integration was wired to the one
kernel of the three that cannot serve this checkpoint.

## Where that leaves the integration

On the INT4 kernel, which is the better target anyway for a reason that has
nothing to do with the bias. The INT8 path needs the packed codes widened,
302 MB a layer, 22.65 GB over 75 layers against a 64 GB part already holding
50 -- which is what made it a per-layer experiment. The INT4 kernel reads the
checkpoint's codes as they are, so the memory blocker retires with the
rewiring, and it was already 4.0 to 4.6% faster.

`routed_int4_binding.cpp` and its `build_so.sh` now target it, with the two
bias tensors in the operator signature and the MSD scratch -- `stacked`,
`stacked_low`, `witness` and the Matmul workspace -- resident per row count.
It builds.

## The shape this also settled
`replay_cann_gmm1.py` asserts the frozen trace's `group_counts` is `(256,)`
with every entry 0 or 1, and `dispatch_a8` is `(8, 6144)`. So at M1 a rank
sees one row per active expert and eight active experts -- a single token
routed to eight of the 256, 16 of which live on this rank. The kernel's
measured shapes, 16 experts of 2 rows and up, are heavier than the decode
round they were meant to serve.
