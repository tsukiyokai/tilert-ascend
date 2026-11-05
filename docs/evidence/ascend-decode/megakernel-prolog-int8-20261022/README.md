# Prolog's two projections at int8: 26.87 microseconds a layer, and untaken

The engine's Prolog projects `weight_dq` and `weight_dkv_kr` from the same int8
`token_x`. Concatenating them into one matmul invocation saves 26.87
microseconds a layer at the engine's dtype, the result is exact, and the vendor
does not already do it.

## The comparison

One idle die of a3-194, twelve rows, two alternating passes an arm, eight
changing inputs each, five timing repeats.

| dtype | level | fused us | split us | saving us | max difference |
| --- | --- | --- | --- | ---: | ---: |
| int8 | prolog | 23.91, 24.20 | 51.07, 50.40 | **26.87** | 0 |
| int8 | query_norm | 28.03, 28.29 | 27.10, 27.10 | −1.19 | 0 |
| bf16 | prolog | 25.51, 25.74 | 56.29, 55.65 | 30.55 | 5.95e-4 |
| bf16 | query_norm | 31.38, 31.21 | 29.64, 30.07 | −1.31 | 2.29e-4 |

The int8 arms match an integer reference exactly, which is a stronger statement
than bf16's 5.95e-4 against magnitudes of 357: there is no accumulation slack to
hide behind. Both arms' product digests are identical at every input.

The saving transfers across the dtype -- 26.87 at int8 against 30.55 at bf16 --
and so does the second level's regression, −1.19 against −1.31. Neither
conclusion was an artefact of the wider operand.

## The vendor leaves it on the table

This is the check that killed the expert-bank lever, applied here. The engine's
Prolog is one fused vendor operator, `MlaProlog::MlaPrologVecS1CubS2`, which
takes `weight_dq`, `weight_uq_qr`, `weight_uk` and `weight_dkv_kr` together, so
it might already concatenate them. It does not. In
`mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h` the two
down-projections are separate matmuls over the same activation:

    796:  mmCqResGm_[...], tokenXGm_[tokenXOffset], weightDqGm_[...], mmCqParam_, ...
    804:  mmCkvKrResGm_[...], tokenXGm_[tokenXOffset], weightDkvKrGm_[...], mmCkvKrParam_, ...

They carry separate parameter structures with separate `needSetOrgShape` flags
and write separate results. Both read `tokenXGm_` at the same offset. So the
invocation this stage removes is one the vendor still pays.

## Scope

One die, no tensor-parallel reduction, no model. The weights here are int8 while
the deployed ones are W4A8 with channel scales and an MSD split, so the bytes
differ again; the saving counts invocations, and the engine's own figure needs
the quantized weight path. The concatenated weight layout is a load-time step.

[`run.sh`](run.sh) is the alternating runner; the JSON files are its output.
