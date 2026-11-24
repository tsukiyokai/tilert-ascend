# The custom operator overlay costs 43 percent at the decode shape

2026-10-21, a3-84, one die, one process, 96 rows, sixteen experts.

    without ASCEND_CUSTOM_OPP_PATH    70.450 us
    with the overlay active          100.604 us

The overlay at `gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer`
contains its own `grouped_matmul_swiglu_quant_v2` -- the proto headers and the
ai_core kernel for ascend910_93 are both there -- so an engine arm that sets
that variable runs a different operator than a bare probe does.

Every engine arm sets it: `launch_arm.sh:34` prepends the overlay to
`ASCEND_CUSTOM_OPP_PATH`.

## What it is worth

30 us a layer over 78 layers is about 2.3 ms a round, collectable by not
setting the variable. No kernel to write.

## What it does not explain

The gap being chased is 61.6 us clean against 153.3 in some mixed state, and
this accounts for about half of it. Eight other candidates were ruled out
first: loading a custom kernel library, running that kernel, building a
communication domain, a second resident fractal-NZ bank, the full two-stage
vendor chain, constant versus random codes, graph capture, and weight bank
size.

## The irony worth recording

The overlay was built to make this operator faster. At the decode shape --
twelve tokens, 96 sorted rows -- it is 43 percent slower, and it has been
active in every engine measurement since it was introduced. Including the
measurements that set the round's budget.
