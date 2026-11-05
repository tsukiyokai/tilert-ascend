# HyperMPMD dissected: the one-sided recipe and the O1 scheduler, read from source

Shallow clone of `mindspore-ai/hyper-parallel` at `/tmp/hyper-parallel-study/`
on the workstation, read only, nothing executed. Paths below abbreviate
`hyper_parallel/core/multicore/` as `M/`. This is the reference the project
was pointed at for "scheduling sunk into AICore" and working symmetric-memory
communication on Ascend+Torch.

## What it actually is

Not a bespoke symmetric memory: a thin, disciplined wrapper over the official
`aclshmem` SDK (gitcode `cann/shmem`, locked at `d30702bb55d0` in
`M/_build/dependencies.lock.json`). The heap is local device memory exported
by runtime IPC name and opened by peers into their own VA — the symbol patch
(`M/shmem/cann_runtime_symbols.patch`) names the machinery:
`rtIpcSetMemoryName / rtIpcOpenMemory / rtSetIpcMemorySuperPodPid`. Not
`aclrtMallocPhysical`, not an HCCL window. Bootstrap is a TCP config store
(`HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT`), subgroups by unique-id broadcast
over torch.distributed (`M/shmem/_lifecycle.py:114-125`).

## The device-side PUT, verbatim discipline

`M/ops/hyper_mega_moe/op_kernel/put_mem_signal/put_mem_signal_kernel.cpp` is
the whole recipe. `aclshmem_ptr(local_symmetric_addr, pe)` translates to the
peer's heap alias; a UB ping-pong (`aclshmemi_copy_gm2ub` on MTE2,
`SetFlag/WaitFlag<MTE2_MTE3>`, `aclshmemi_copy_ub2gm` on MTE3 to the
translated remote address) moves the bytes — a remote write is an ordinary
MTE3 copy to a mapped address, no special instruction. Ordering:
`WaitFlag<MTE3_MTE2>` drains the MTE3 pipe (:108-109), `aclshmem_fence()`
(:113-117) orders SHMEM operations, then `aclshmemx_signal_op(remote_signal,
value, ADD/SET, pe)` (:126-128). Data visibility before signal visibility is
the SDK contract (`M/shmem/ccsrc/data_plane/rma.h:75`). **The sender never
issues DCCI against a remote address.** The waiter spins only on a flag in
its own heap (`aclshmem_signal_wait_until`), with rate-limited DCCI.

The 507015 we died on is named in their source:
`M/shmem/ccsrc/bindings/torch_bindings.cpp:706-708` — "aclshmemx_barrier_on_
stream spins without a timeout and deadlocks under interconnect contention
(aicore 507015 'cross-device memory access times out')", elevated to protocol
in `M/docs/shmem.md:128-131`. Their replacements: host HCCL barrier plus
local `zero_()` before any remotely visible write (`workspace.py:213-222`);
in-kernel cross-rank convergence by generation-counted radix-2 dissemination
(`M/ops/runtime/worker_kernel.h:201-237`), never a device barrier; 64-byte
isolation per signal slot (`data_plane/sync.h:23`). Against our TP16 one-shot
push that died at a DCCI after remote MTE3 output with two-PE surviving: this
codebase never does that DCCI, and "unbounded spin amplified by interconnect
contention" matches the 2-alive/16-dead signature.

## O1 is a static plan plus event counters, not a resident scheduler

No device task queue, no ffts+, no CrossCoreSetFlag between AIC and AIV. The
host compiles one `RuntimeConfigC` blob (64 B header, trigger-threshold
table, 576 B TaskDescs, per-core static task index arrays —
`M/ops/runtime/runtime_config.hpp:190-414`); each core strides its own slice
(`ProcessFast`, `worker_kernel.h:138-163`, `#ifdef __DAV_C220_CUBE__` for the
cube path, odd AIVs only for vector). Two primitives: TriggerEvent = one 32 B
`SetAtomicAdd<int32_t>` DataCopy into a GM counter array (:309-340);
WaitForDependency = threshold read with per-cacheline
`DataCacheCleanAndInvalid` + `PipeBarrier<PIPE_ALL>`, DCCI rate-limited to
2500 cube / 7500 vector cycles (:397-424). The elegant part: a cross-rank PUT
completion is `aclshmemx_signal_op(ADD)` **onto the same event-counter array**
on the target rank (`worker_kernel.cpp:224-267`), so intra-kernel scheduling
and inter-chip communication share one event semantics. Cube→vector handoff
is GM plus these counters; no UB or L2 hand-off.

RATR is fifteen lines offline (`M/scheduler/scheduler.py:26-52`): dispatch
tasks reordered so at step k every rank r targets `(r + k) mod ep` — rotation
de-incasts, zero runtime cost.

## Relation to CANN mega_moe, and a re-check we owe ourselves

`HyperMegaMoe` (vendor `hyper_parallel_multicore_nn`,
`torch.ops.hyper_parallel.mega_moe`) and CANN `mc2/mega_moe` are different
families: the former inlines `gmm/grouped_matmul` and `swi_glu` sources as
building blocks over aclshmem; the latter is inference-oriented over
`utils/hccl_shmem.hpp`. But ops-transformer master's `mc2/mega_moe` carries
A3 + W4A8 dispatch+GMM+SwiGLU+GMM+combine incl.
`block_epilogue_w4a8post_pertoken_swiglu.hpp` — MoE-dispatch semantics, not
our TP reduce, yet our "no vendor fusion serves us" record should be
re-verified against that directory specifically.

## Constraints, located

T % 128 == 0 is the offline plan granularity (`spec.py:28`,
`module.py:372-378`), incidental — the kernel handles ragged actual sends.
BF16-only is packaging (single-dtype binary json; quantised template branches
exist but are empty at the entry, `hyper_mega_moe.cpp:32-53`). push×graph
capture exclusion (`module.py:495-496`) is intrinsic to online heap rebuild,
not to PUT; fixed-capacity buffers avoid the rebuild but the guard is
unconditionally wide. Four-row decode fails the constructor outright.

## What transfers to our fused-kernel epilogue

Three MoE-orthogonal layers: the PUT + anti-507015 discipline (map: each rank
PUTs its partial sum into per-peer exclusive slots plus signal-add; consumer
waits for count == 15 locally, then vector-reduces); the two self-contained
operator-agnostic headers under `M/ops/runtime/` as the cube/vector
cooperation skeleton; the RATR rotation for any all-to-all-shaped step. Costs
acknowledged: host-side `aclshmemx_init_attr` must precede the kernel, the
kernel links the SDK device library (include recipe at
`M/cmake/vendor/op_kernel/CMakeLists.txt:37-44`), and graph capture over
fixed-capacity symmetric buffers is unverified by that repo.
