# Communication that is not a standalone kernel

> More importantly, communication in TileRT is not implemented as a standalone
> kernel. It is integrated directly into the execution pipeline, allowing
> compute and communication to continuously overlap.
> — [the AMD write-up](../tilert-ascend/amd-article-reread-20261018.md)

The one-sided reduction measured so far is 2.84× the vendor's, but it is still
a standalone kernel: it pays its own launch and overlaps the compute it
follows by exactly nothing, which is the structure that sentence rejects.
[one-sided all-reduce](../onesided-allreduce-20261018/README.md)

`fused_proj_reduce_kernel.cpp` puts them in one body. **Both passes compile,
exit 0**, and the cube object carries a real `SET_CROSS_CORE` instruction with
78 KB and 164 KB of code on the cube and vector sides.
[kernel](../../../../tests/glm52/shmem/fused_proj_reduce_kernel.cpp)

## The structure

One `KERNEL_TYPE_MIX_AIC_1_2` body, each half guarded:

The cube side copies its `TCubeTiling` from global memory, runs
`Matmul<AQuant, BQuantNz, CPartial, BiasNone>` — int8 activation against a
FRACTAL_NZ int8 weight, which is the layout production actually holds — and
raises `CrossCoreSetFlag<0, PIPE_FIX>` when the partial is in global memory.

The vector side waits on that flag and then pushes **per tile**: for each
32 KiB chunk of the partial, `aclshmemx_mte_put_nbi` sends that chunk to one
peer, so a tile is in flight while the next is still being written. That is
the whole difference from a collective kernel, which cannot start until the
entire matmul has finished. Then `aclshmem_quiet`, `aclshmemx_sync_vec_all`,
and the ordered local sum over the sixteen slots.

One-sided operations stay on the vector side because
`include/device/gm2gm/shmem_device_cc.h:23` places the cube scalar unit
outside the visibility guarantee, and every one-sided kernel in the SDK is
declared vector-only. All three UB buffers come from a single `TPipe`, since
two allocators dividing one UB is what silently corrupted an earlier version.

## What compiling establishes, and what it does not

It establishes that `Matmul<>` with NZ weights and `aclshmemx_mte_put_nbi`
can share one MIX kernel on `dav-c220-cube` and `dav-c220-vec` — which was
genuinely open, since the cube and vector halves have different visibility
guarantees and compete for the same UB budget. The compile also found the
include roots: the device `.hpp` files live under the source tree's
`src/device/`, not the install tree, which is what
`examples/CMakeLists.txt` adds for every kernel target.

It does not establish a number. The kernel has no host harness yet, so
nothing is claimed about whether per-tile pushing beats the standalone
22.3 µs. The comparison to make is this kernel against the two kernels it
replaces — a vendor `npu_quant_matmul` plus the standalone reduction — at
sixteen ranks with the payload check.

## Why this is the shape, not an optimisation

The round spends 11.29 ms on collectives and overlaps them with compute by
0.000000 ms, measured. A faster standalone collective can at best shrink that
11.29; only a collective inside the compute pipeline can hide it behind work
that has to happen anyway. With the weight stream's floor at 3.56 ms of a
6.87 ms budget, hiding is the only route that closes.
