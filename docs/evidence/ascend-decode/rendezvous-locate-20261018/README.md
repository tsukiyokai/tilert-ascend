# Locating the reduction's cost: it is none of transfer, hops or topology

Before building a one-sided push, four measurements on `a3-84` to find what
the 63.5 µs in-graph (126 µs eager) reduction is actually made of.
[world/hop probe](../../../../tests/glm52/bench_collective_shape.py).

## It is not the message

Eager float32 `all_reduce` over sixteen ranks costs 157.43 µs for four bytes
and 109.72 µs for six mebibytes
([size sweep](../reduce-floor-20261018/README.md)). Flat across 1.5 million
fold.

## It is not the number of hops

A 64-byte reduction by world size: 2 ranks 45.98 µs, 4 ranks 153.23,
8 ranks 107.00, 16 ranks 158.07. Not monotonic, noise larger than any trend —
so neither a ring's O(N) nor a tree's O(log N). **Two ranks exchanging
64 bytes already costs 45.98 µs.**

## It is not the topology

Pairwise `all_reduce` from rank 0 to each of the fifteen peers: 68.90 to
95.93 µs, ratio 1.39. A3 pairs two dies per module, so rank 0 ↔ 1 should be
the cheapest hop available; it measured 94.38 µs, among the slowest. Peer
identity does not matter.

## And the other vendor path is worse

`tensor.copy_()` across dies costs ~160 µs for every size from 4 B to 1.5 MB,
and sixteen back-to-back `non_blocking=True` copies still cost 158.25 µs
each, so they do not pipeline at all — each call is a full serialisation.
`can_device_access_peer` returns true for every peer and contexts open fine,
so reachability is not the limit; torch's cross-device copy simply is not a
peer DMA.

## What is left

Three independent vendor paths — HCCL reduction at 63.5 µs in-graph, HCCL at
45.98 µs for two ranks and 64 bytes, torch cross-device copy at 160 µs — all
carry a fixed per-operation cost of the same order that depends on nothing
about the data. That is the "separate synchronization stage" the AMD write-up
describes removing, and it matches this project's earlier finding that A3's
HCCL executor re-runs on device-side AICPU and recomputes link requirements
per call.

The wire itself is still unmeasured, and no vendor path on this machine can
measure it. Only a kernel writing into a mapped peer address can.

## The mechanism, read properly

The AMD description removes two things, and the second is the larger: "This
removes multiple rounds of data transfer **as well as a separate
synchronization stage**." A one-shot push is not a faster collective — it is
the absence of a collective. The producing kernel's epilogue store is aimed
at sixteen symmetric buffers instead of one; the consumer reads sixteen local
buffers and reduces them in a fixed order, which is a local vector op; the
synchronisation is one flag per peer that the consumer spins on locally, and
the producer never waits for an acknowledgement, so the latency paid is
one-way rather than a round trip. The fixed order is also bit-reproducible,
which matters here: EP16 diverged the trajectory precisely by changing the
summation's association.

## The primitives exist on this machine

No `aclshmem` SDK is installed, but two routes are:

PTO-ISA's `pypto` ships the complete surface —
`pypto.distributed.create_shmem_tensor(group_name, n_pes, dtype, shape)`,
`create_shmem_signal`, `shmem_put`, `shmem_get`, `shmem_signal`,
`shmem_wait_until`, `shmem_barrier_all`, `shmem_clear_signal`,
`my_symbolic_pe`, plus
`pypto.experimental.shmem_store(src, offsets, dst, dst_pe, put_op=AtomicType)`
whose own docstring example is our use case verbatim: a `pypto.matmul`
result stored straight into a peer's symmetric buffer. `shmem_wait_until`
is the local-flag spin, so no barrier is needed. PTO already runs bit-exact
on this machine ([stack selection](../stack-selection-20261018/README.md)).

The lower route is the runtime's own IPC mapping, whose symbols are present
in `libascendcl.so`: `aclrtDeviceEnablePeerAccess`, `aclrtMallocPhysical`,
`aclrtMapMem`, `rtIpcMemGetExportKey`, `rtIpcMemImportByKey`,
`rtIpcMemSetImportPid`. That is the recipe HyperMPMD's symbol patch revealed
([dissection](../hyper-mpmd-20261018/README.md)).

So the gate is open. The next number is a sixteen-rank one-sided push plus
ordered local reduction, measured against 63.5 µs.
