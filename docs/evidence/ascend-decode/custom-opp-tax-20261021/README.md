# CANN's own operator is the fastest; three custom packages sit on top of it

2026-10-21, a3-84, one die, one process, 96 rows, sixteen experts. The same
`grouped_matmul_swiglu_quant_v2` call, varying only
`ASCEND_CUSTOM_OPP_PATH`.

    CANN only, no custom opp     66.813 us
    owner overlay only           97.362 us   +46%
    vllm-ascend package only    123.985 us   +86%
    both custom packages        104.929 us
    plus the gmm-fastpath overlay (measured separately)  100.604 us

`activate.sh:18` sets the two custom packages for every engine process, and
`launch_arm.sh:34` prepends a third. So no engine measurement has ever run
CANN's own implementation of this operator.

## This is the state difference

The 61.6 us clean against 153.3 us mixed gap was chased through nine
candidates -- loading a kernel library, running that kernel, building a
communication domain, a second resident fractal-NZ bank, the full two-stage
chain, constant versus random codes, graph capture, weight bank size, and the
TP16 gate-up width -- all innocent, because none of them touched the operator
path. A single-die probe does not source `activate.sh`; the engine does.

## What it is worth and how to take it

66.813 against 104.929 is 38 us a layer, about 3.0 ms a round. Unsetting the
variable wholesale is the wrong way to collect it: the custom packages exist
for other operators and other shapes, and one of them is vllm-ascend's own.
The right form is to let this one operator fall back -- an opp path that does
not carry `grouped_matmul_swiglu_quant_v2` while keeping the rest.

## The shape of the mistake

Three layers of custom operator packages, each added to make something
faster, together costing 57 percent on the operator that dominates the
round's largest line item. None of them was ever measured against the CANN
implementation at the decode shape.

## Not a general rule: the package wins on the other operators

Same two paths, the operators the package also overrides:

    rms_norm + dynamic quant   CANN 56.049 us   custom 43.038 us   custom 23% faster
    moe_gating_top_k_softmax   CANN 33.732 us   custom 26.455 us   custom 22% faster
    rms_norm                   CANN 26.527 us   custom 27.076 us   even

So there is no systematic conclusion, only a per-operator fact. Unsetting the
path wholesale collects 38 us a layer on the grouped matmul and gives back
about 20 on the norm and the gating -- net 1.4 ms a round rather than 3.0.

## Trimming the package does not work

Removing the grouped matmul's kernel directory and its config json from a copy
of the package left the operator at 116.376 us -- still not CANN's 66.813 --
and pushed the norm and the gating to 50.875 and 30.032, between the two
states. So deleting files does not deregister an operator; dispatch finds it
elsewhere, probably `binary_info_config.json` or the proto, and a missing
binary falls back to something slower than either.

Collecting the full 3.0 ms needs a correct deregistration or a call-site
bypass. The queued end-to-end arms use the wholesale unset, which understates
the lever -- if its 1.4 ms shows up, the precise form is worth building.
