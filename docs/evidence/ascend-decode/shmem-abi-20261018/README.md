# One ABI defect, three failures: the SHMEM barrier never worked here

Two broken vendor examples and my own deadlocked kernel turned out to be the
same defect appearing three times. Found by a background investigation on
`a3-84`; `.orig` copies of every edited file are kept beside them.

## The defect

In an AscendC `__global__ __aicore__` signature, an eight-byte parameter
placed **after** a four-byte one is written by the host launch stub at a
non-eight-aligned compact offset, while the device prologue reads it from the
next eight-aligned offset — which is already past the end of the args buffer.
The parameter is not slightly wrong; it is something else entirely.

The library commits it itself:

```c
aclshmemi_barrier_on_stream_kernel(aclshmem_team_t team, uint64_t ffts_config)
```

`aclshmem_team_t` is `int` (`include/host_device/shmem_common_types.h:120`),
so `ffts_config` sits at offset 4 with `argsSize=12` and the device reads from
offset 8. The plog is unambiguous: `args(0 to 0) after execute:
0xfffb900000000000` is `team=0` concatenated with the low half of
`ffts_config`.

**So `aclshmem_barrier_all()` has never worked on this machine.** Every call
logs 507015 at `src/host/data_plane/shmem_host_cc.cpp:85` — 964 entries across
sixteen ranks in one run, about one per barrier — and the function returns
`void`, swallowing `ACLSHMEM_INNER_ERROR`. A silent no-op. The stock
`examples/barrier_all` fails at 2, 4 and 8 PEs.

The fix is a parameter reorder on the two on-stream kernels plus their call
sites, leaving `shmemi_device_cc.h` and the hardware `SyncAll` path untouched.
Four alternatives were compared; two others also pass but only by routing
around FFTS, and one of those downgrades every user kernel's
`aclshmemx_sync_vec` from hardware cross-core sync to GM polling — which this
project cannot accept. After the reorder `examples/barrier_all` passes at 2
and 8 PEs with zero 507015.

## This was my deadlock

My `tp_push` plog names `_Z34aclshmemi_barrier_on_stream_kernelim`,
`blockDim=1`, `argsSize=12` — the library defect, not my kernel. The failure
happens in the host barrier **before my first launch**. Three of my plogs
(19:35 and 21:22) are all this one entry, and on the fixed library my kernel
produces no plog at all.

Two of my hypotheses are refuted by it. The SDMA/QP precheck from PR 897 is
irrelevant: no QP, link or SDMA resource error appears anywhere in the logs.
And the seven-argument GM-to-GM `aclshmemx_mte_put_nbi` needs no
initialisation the five-argument UB form does not — both reach the same
chunking at `src/device/gm2gm/engine/shmem_device_mte.hpp:183-212`, and 96 KiB
is one block, under my own 129024-byte staging budget. `combine` pushes
h=6144 at a time because that is its tiling granularity, not a limit.

**And my stage isolation was backwards.** On the fixed library at 2 PEs,
`stage 1` (publish and wait, no reduce) returns 0 with `wrong 0` at
**9.681 µs**, while `stage 2` (reduce, no communication) hangs. I had measured
the opposite — through a broken barrier, so not measuring what I thought.

Three concrete faults in my `TpPushReduce`, none of which produce an error
code: hand-filling `LocalTensor::address_` to bypass TPipe allocation and
pairing it with explicit `SetFlag`/`WaitFlag`, where a `WaitFlag` on an event
never set is exactly this kind of silent permanent stall; `REDUCE_UB_BASE =
1024` landing inside ACLSHMEM's reserved `mte_config` UB window `[0, 16384)`
(`src/host/init/shmem_init.cpp:42,78`); and `GetBlockIdx()` in a MIX kernel
returning the **cube** block index, so with `cores = GetBlockNum() *
GetSubBlockNum()` two sub-blocks share one `core`, half the stripes are never
computed and two sub-blocks publish the same peer. The correct index is
`GetBlockIdx() * GetSubBlockNum() + GetSubBlockIdx()`.

Also: the SDK's own one-sided kernels are all `ACLSHMEM_GLOBAL_VECTOR`
(`[[bisheng::core_ratio(0, 1)]]`), and
`include/device/gm2gm/shmem_device_cc.h:23` states the cube scalar unit is
outside the visibility guarantee — so publish and wait must be guarded by
`if ASCEND_IS_AIV`.

## ACL graph capture is cleared, and that was the open risk

The verdict is reached by same-binary polarity rather than inference. With
capture bypassed via a new `ACLGRAPH_DEMO_NO_CAPTURE` switch the failure is
identical, only the status changes from 507011 (the model-layer wrapper) to
507015 — **capture is not in the causal chain**. Three magic-parameter
variants settle the rest: the original unaligned layout fails both captured
and immediate, passing the value by host read passes both, and moving the
pointer before the `int` passes both.

After the fix, 400 graph replays across sizes and 2/4/8 PEs ran 800 kernels
containing `aclshmemx_mte_put_nbi`, `mte_get_nbi`, `signal_op`,
`signal_wait_until`, `aclshmem_ptr`, `AscendC::SyncAll` and
`aclshmemx_sync_vec_all`, bit-correct every round, pure replay from round two
with `magic` written by `aclrtMemcpy` and read by the graph.

**One-sided symmetric memory survives ACL graph capture and replay on A3.**
Our production decode path is entirely graph-captured, so this was the single
largest unknown in the whole one-sided programme. A mechanism note supports
it: `ffts_addr` is a per-device constant obtained once in `aclshmem_init`
(`src/host/sync/shmemi_sync.cpp:22-41`), so the value baked into a graph's
task args cannot go stale.

## The 0x91 and the -3 were both wrappers

`retCode=0x91` is `model_execute_task.cc:163` wrapping
`ACL_ERROR_RT_MODEL_EXECUTE` (507011). The real fault is four lines above:
`fftsplus aivector error`, `The address for the scalar to access the internal
buffer of AICore is out of bounds`, `subErrType:4` — AI Core error bit 121
`ccu_mpu_err`, a scalar MPU window check, **not** a UB overflow, which would
report 124/125 or 135. UB overflow is excluded arithmetically: peak offset
194624 against `ub_size=196608`.

`aclError:-3` is not `ACL_ERROR_INVALID_PARAM`. `main.cpp:558` is
`ACL_CHECK(aclshmem_finalize())` and `-3` is `ACLSHMEM_SMEM_ERROR` from a
different enum. It is also downstream noise: the number of ranks reporting it
varies run to run, depending on which ranks escape the device deadlock far
enough to reach finalize. The profiling hypothesis is experimentally dead —
`--perf --warmup 0 --loops 1` passes while `--perf --warmup 59 --loops 1`
hangs, and so does a configuration where no rank allocates a prof buffer at
all.

## combine's second, independent defect — and our own number

With the library fixed, `combine_classic` still hung: its status slots stride
32 bytes, so two signal slots share one 64-byte scalar cache line, and a
signal set is a cached store plus a writeback of that line
(`src/device/gm2gm/shmem_device_so.hpp:380-389`). Two concurrent senders
overwrite each other's SET and the receiver's wait never completes. ACLSHMEM
treats its own sync bits as `ACLSHMEM_SYNCBIT_SIZE = 64`. Fixed by striding
the slots to 64 bytes, carrying the launch's `magic` as the flag value instead
of a constant 1, and deleting a cached-store zeroing loop that dirtied the
line.

That non-perf mode passed was luck: it forces warmup 0 and loops 1, so there
is never a second round to race.

Then the measurement we wanted, at our decode shape:

```
-pes 16 -bs 4 -h 6144 --perf --warmup 10
host per-reduction us pe=0 reductions=50
  min=4817.30 p50=4996.55 mean=4998.39 max=5188.07
combine_perf_rank0.csv: full_op 4936.59 us, comm_only 18.5104 us
  (633.09 GB/s global, 39.57 GB/s per PE)
```

**`comm_only` is 18.51 µs** for a sixteen-rank combine at four tokens and
hidden 6144. The 4937 µs `full_op` is the MoE arithmetic around it, and the
4998 µs host figure differs from it by launch plus sync.

## Nine more latent instances, and the rule

A script recomputing natural-alignment offsets over every kernel signature in
`examples/`, `src/` and `tests/` found nine further cases of an eight-byte
parameter after a four-byte one, in `udma_demo`, `hccs_sio_link` (twice),
`rdma_perftest_demo`, `notifywait` (three) and `sdma` (two). The library's
`src/` has no other `__global__` entry points, so the list is complete.

The rule for our own kernels: **every eight-byte kernel parameter must land on
an eight-byte-aligned compact offset, and the simplest way is to put all
eight-byte parameters first.**

Operationally: device-side builds do not track header dependencies, so after
editing anything under `src/device/` the dependent `.cpp` must be touched or
the measurement is of the previous `.so`.
