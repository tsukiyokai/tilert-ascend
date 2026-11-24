# A one-sided TP16 all-reduce: 22.3 µs against HCCL's 63.5

`a3-84`, dies 0–15, `[4, 6144]` float32 — 96 KiB, the decode shape — on the
SHMEM library with the barrier parameter reorder in place
([ABI defect](../shmem-abi-20261018/README.md)). Every measurement carries the
correctness check: each rank contributes its own index, so every element of
the sum must equal `pe_size*(pe_size-1)/2 = 120` and a wrong summation order
cannot pass.

| rounds | per reduction |
| --- | --- |
| 1 | 109.825 µs |
| 25 | 25.933 |
| 100 | 23.144 |
| 400 | **22.344** |

Three repeats at 400 rounds: 22.265, 22.382, 22.402 — 0.6% spread, every run
`wrong 0 untouched 0 of 24576 expected 120.0 got 120.0`. The single-round
figure is almost entirely launch, the same constant-product artefact that cost
four wrong explanations earlier
([per-call cost](../fixed-cost-20261018/README.md)).

**22.3 µs against HCCL's 63.5 µs in-graph is 2.84×, and against 126 µs eager
is 5.64×.**

| cores | per reduction |
| --- | --- |
| **16** | **22.339** |
| 20 | 22.666 |
| 24 | 22.799 |
| 32 | 23.482 |
| 48 | 25.049 |

Sixteen is both the optimum and the floor, since `blocks` must be at least
`pe_size` for the fan-out to complete — and the check catches a short launch
rather than letting it pass silently: `blocks=8` at TP16 returns `got 84.0`
with rc=1, and 84 is exactly the sum of the eight ranks satisfying
`(r + c) % 16 == 0` with `c < 8`.

Halves at 16 cores and 400 rounds: **push plus sync 17.523 µs, local sum
3.421 µs**, so communication is 78% of the cost. At two PEs: 6.233 total,
5.422 push, 0.907 sum.

## The structure, and why two core groups was wrong

Each core `c < pe_size` pushes the whole partial to peer
`(my_pe + c) % pe_size` — a fan-out of `pe_size` puts rather than one small
put per core per peer — then `aclshmem_quiet()` with
`aclshmemx_sync_vec_all()`, then every core sums its stripe over the
`pe_size` slots in rank order. No flags, no polling, every core doing both
phases.

The producer/consumer split that `allgather` needs is wrong for a reduction,
and it was the main cause of the 3.8× penalty I had measured at 48 cores: half
the cores idle in each phase, and the flag poll issues an
`aclshmem_int32_get_nbi` per (peer, slice). Removing it left a mild 12% rise
from 16 to 48 cores. `allgather` needs the split because its consumers pull
peers' *payloads* through a staging window; after a push every rank's local
window already holds all the partials, so no core needs to fetch for another.

The kernel is declared `[[bisheng::core_ratio(0, 1)]]`, as every one-sided
kernel in the SDK is. That makes `GetBlockIdx()` the AIV index directly
instead of needing `GetSubBlockIdx()`, keeps cube cores from issuing one-sided
operations — `include/device/gm2gm/shmem_device_cc.h:23` places the cube
scalar unit outside the visibility guarantee — and matches
`aclshmemx_sync_vec_all()`'s vector-only contract.

Two planes with one sync per round, and the safety argument is that reaching
round r+2 requires passing r+1's sync, while any peer passing r+1's sync has
already completed r's reduction in program order — so r+2 cannot overwrite a
plane still being read. That saves one sync per round against the obvious
two.

## What I had got wrong, beyond the deadlock

**My protocol never computed an all-reduce.** `TpPushPublish` passed `my_pe`
as the target PE, and `put_nbi`'s `pe` argument decides *whose* window the
data lands in — so it was a local copy and the data never crossed a rank. The
other fifteen slots stayed at their memset zero while the reduction read
exactly those slots. My design premise, that every rank's data is already
local after publish, was a misreading; the original
`peer = (my_pe + core) % pe_size` was right and I had edited it away. It was
not nearly working: it was measuring something that did not exist.

**And my stage isolation was fake.** `stage 2` disabled the publish while
keeping the consumer's flag poll, so consumer cores waited on a flag nobody
would ever raise by construction. That was not a reduction stall. Both my
earlier stage-2 hang measurements are void.

Three further real faults, independent of the stall: `slices = cores / 2`
overran `MAX_SLICES = 32`, since a MIX kernel at 48 blocks gives
`cores = 96` and `half = 48`, so publisher cores 32–47 wrote flags outside the
flag region and into plane 1's flags and payload. A hand-picked
`FLAG_UB_BASE` and `STAGE_UB_BASE` alongside a `TPipe` is two allocators
dividing one UB without knowing about each other — "the TPipe allocates on
top" is not something a TPipe can do. And moving UB above ACLSHMEM's reserved
`[0, 16384)` was unnecessary: that `mte_config` window is only used by the
higher-level RMA path, and `aclshmemx_mte_put_nbi` with an explicit user
buffer never touches it, as `examples/allgather`'s `tmp_buff = (__ubuf__
T*)(64)` demonstrates.

## Two environment misreadings of mine

`examples/barrier_all` is not broken — it takes positional arguments
`<n_pes> <n_pes_per_node> <node_id> <tcp://ip:port> [first_device_id]`, so my
`-pesize 2 -gnpus 2` hit the usage branch and never ran. Called properly it
reports `[PASS]` on both Device and Host lines at 2 and 8 PEs, which also
confirms the parameter reorder fixed the in-kernel `aclshmem_barrier_all()`.

And my `bash scripts/build.sh` without `-examples` reset the CMake cache to
`USE_EXAMPLES:BOOL=OFF`, removing every example target while leaving the
previous `build/bin/` executables in place — so builds did not fail, they
simply stopped updating. Recovering it in place with
`cmake -DUSE_EXAMPLES=ON …` avoids another full reset.

## The next lever, with the experiment that decides it

Per rank per round the outbound traffic is 96 KiB × 16 = 1.5 MiB, and push
plus sync is 78% of the cost. Reduce-scatter plus all-gather would move
2 × 15 × 6 KiB ≈ 180 KiB, **eight times less**, at the price of two cross-PE
syncs per round instead of one.

A stage that only does `aclshmem_quiet()` and `aclshmemx_sync_vec_all()` and
moves nothing settles it. At TP16, 16 cores, 400 rounds:

| stage | per reduction |
| --- | --- |
| **sync only** | **5.078 µs** |
| push plus sync | 17.655 |
| local sum only | 3.368 |
| full | 22.403 |

**The sync floor is 5.078 µs, well under 17.5**, so the push half carries
12.58 µs of actual data movement. Reduce-scatter plus all-gather divides those
bytes by eight and pays a second sync:

```
12.58 / 8 + 2 × 5.078 = 11.73 µs      against today's 17.655
```

That is about 5.9 µs off the communication half and takes the whole reduction
from 22.4 to roughly 16.5 µs — a further 26%, and 3.85× HCCL's 63.5 µs. The
experiment returns a go.

Two caveats on the arithmetic. It assumes the 12.58 µs scales linearly with
bytes, which the earlier size sweep suggests is optimistic at these sizes
([rendezvous](../reduce-floor-20261018/README.md) found the vendor collective
entirely size-independent, though that was rendezvous rather than transfer).
And it assumes the two syncs cannot overlap with their own transfers, which a
careful implementation might partly defeat.
