# Constant tiling for the routed gate/up projection

This independent candidate starts from token-terminal library `276352762139244a1ddfc8184065a856c2ae75ee2ccad5961af2af93f0dd5e00`. Only the GU `Matmul` template receives a `MatmulApiStaticTiling` configuration. Its `cfg` remains `CFG_NORM`; BasicBlock, MDL and static padding stay disabled. Down, shared experts, all arithmetic, active ownership, five body joins and terminal completion retain the parent's implementation.

The constant fields match the retained host tiler for both ND and NZ weights: base M/N/K is 64/256/256, N is 256 and K is 6144. M and singleCoreM remain dynamic tiling fields. Each expert still supplies `SetOrgShape(2*count,256,6144)` and `SetSingleShape(2*count,256,6144)`, including changing counts within the same graph. The host tiler, tensor validators and original 33 scratch addresses are unchanged. The 408-byte descriptor, 112-byte shared-input record and 116480-byte resident UB budget are unchanged.

The intervention makes fixed tiling values visible to the compiler. It does not reduce selected weight bytes or the 24 K256 matrix steps per expert. `SetTensorA/B`, `IterateAll` and `End` remain in the source; assembly comparison must identify any eliminated loads, divisions or branches before attributing an effect to constant propagation. Device numerical and matched-address performance checks are pending.
