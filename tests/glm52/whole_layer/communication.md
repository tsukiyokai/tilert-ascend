# Device TP reduction

`communication_device.h` provides an in-kernel BF16 all-reduce for an
Ascend 910B/C MIX kernel. Each rank sends its partial to every rank through
SHMEM MTE, sums received BF16 values in ascending rank order in FP32, and
rounds the result once to BF16. Two calls can separate attention, FFN, and
the following layer without returning to the host. This numerical order is
a candidate model contract; HCCL may use another reduction order.

The real device probe in `communication_probe.cpp` uses one MIX launch for
all-reduce, a rank-local CUBE projection of 16x16 tiles, BF16 conversion,
and a second all-reduce. It tests communication dependencies and graph
reuse. It does not implement a GLM layer or measure model TPOT.

## Required execution conditions

- All ranks use the same HCCS/MTE-mapped SHMEM world, tensor width, capacity,
  and collective order. This implementation does not support a ROCE fallback.
- Shapes are BF16 physical ND `[M,6144]`, `1 <= M <= capacity <= 32`, with
  `2 <= ranks <= 16`. Both partial and output contain `M*6144` elements.
- Every resident AIC and AIV enters every call. Launch MIX 1:2 with
  `__schedmode__(1)`, initialize SoC state, set the synchronization base and
  Matmul system workspace, and limit blocks to `min(AIC, AIV/2)` for the
  actual device. Concurrent kernels must not prevent these blocks residing.
- Finish preceding Matmul operations, hard events, and cross-core flags
  before entering the reduction. The caller owns `SyncAll<false>` flags
  11..13. Communication uses hard-event slot 0 in its MTE/vector domains;
  those domains may be reused only after the preceding phase drains them.
- `CommunicationBuffers::Init` uses 20 KiB of AIV UB from the caller's
  `TPipe`. Explicit MTE staging avoids SHMEM's default UB scratch. There is
  no permanent UB reservation during another phase; reinitialize the
  buffers after resetting the pipe.
- One window belongs to one serialized stream and its graph executions.
  Different streams or concurrently replayed graphs require different
  windows. All ranks allocate windows in the same order before capture.
  Release them only after every rank has finished every use and before
  SHMEM finalization. Keep the runtime device context alive throughout.

The host allocator uses real `aclshmem_calloc`: it zeroes local storage
before its host control-plane rendezvous returns. Its window is 64-byte
aligned and contains one local epoch counter, two ready planes, and two
payload planes. Ready words for different senders occupy separate cache
lines. A capacity-32 TP16 window consumes 12,585,024 bytes per rank.

## Ordering and reuse

The device counter starts at zero and advances once per collective. It is
persistent across layers and graph replays; the host does not rewrite it.
Each invocation performs these phases on every local MIX group:

1. AIV 0 increments the epoch. A full pipeline/DDR fence and local
   `SyncAll<false>` publish the preceding producer's output and the epoch.
2. AIVs divide peers among themselves and call `aclshmemx_mte_put_nbi` with
   explicit UB storage. Each writer calls `aclshmem_quiet`. A second full
   local barrier completes every writer before the ready publisher starts.
3. AIV 0 publishes this rank's ready word to every peer with
   `aclshmemx_signal_op(..., ACLSHMEM_SIGNAL_SET, peer)`. Each reducing AIV
   waits for an exact epoch match in the appropriate ready plane before
   reading each sender's payload. AICs issue no SHMEM operations.
4. A final full local barrier completes all payload reads and output writes
   before any core can begin the next collective.

Both payload and ready use `epoch & 1`. A writer reaching epoch `r+2` has
finished `r+1`, which requires every rank's ready signal for `r+1`. A rank
can publish that signal only after its local readers finished `r`. Thus
the old plane cannot be overwritten while a reader still needs it. A
single ready plane fails: a fast rank can replace ready `r` with `r+1`
before a slow rank observes `r`, leaving an exact-match wait stuck.

`communication_protocol_check.py` enumerates asynchronous sender,
publisher, and consumer transitions. The checked cases are two ranks/six
rounds and three ranks/four rounds; both include repeated plane reuse. It
also rejects single-payload-plane and single-ready-plane controls. This
checks the protocol state machine, not device visibility or runtime
scheduling. Missing ranks, divergent call counts, a reset live epoch, and
concurrent use of one window violate the execution conditions and can
deadlock.

## Source and build

The initial build uses SHMEM commit
`24c15d21c26972404b0bb0e9cfd08d03930d15c6`. Its relevant implementations are
`src/device/gm2gm/engine/shmem_device_mte.hpp:247` (explicit-UB put),
`src/device/gm2gm/shmem_device_mo.hpp:17` (quiet),
`src/device/gm2gm/shmem_device_so.hpp:262` (MTE signal SET), and
`src/device/gm2gm/shmem_device_p2p_sync.hpp:388` (wait). Signal SET writes a
mapped peer address and flushes its cache line; polling invalidates the
line before reading. No SHMEM barrier or collective API is called.

Build the matching source and libraries in an independent writable
directory. This configuration builds only the MTE library and bootstrap
dependencies, with two CPU compilation jobs:

```bash
source /opt/ascend/cann-9.2.0/set_env.sh
cmake -S "$SHMEM_SOURCE" -B "$SHMEM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_TYPE=Ascend910B \
  -DUSE_EXAMPLES=OFF -DUSE_UNIT_TEST=OFF -DBUILD_PYTHON=OFF \
  -DACLSHMEM_UDMA_SUPPORT=OFF -DACLSHMEM_RDMA_SUPPORT=OFF \
  -DENABLE_CANN_BUILD=ON
cmake --build "$SHMEM_BUILD" --parallel 2 \
  --target shmem aclshmem_bootstrap_config_store aclshmem_bootstrap_uid
SHMEM_LIBRARY="$SHMEM_BUILD/lib/libshmem.so" \
  bash tests/glm52/whole_layer/communication_build.sh
```

`SHMEM_SOURCE` must be exported or passed to the build command. The build
records compiler identity, source SHA256, SHMEM header SHA256, and linked
library SHA256 beside `communication_probe.so`. The checked CANN 9.2 build
compiles the real MIX entry, allocator, tiler and C ABI and passes `ldd -r`.
The corrected entry passes the two-rank device probe at M=1,12,32 with
12 changing-input replays per width. Both reductions match the ordered
oracle and installed HCCL byte for byte. The
[device reports](../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-84/README.md#two-rank-communication)
record the accepted binary and the failing predecessor. TP16 and whole-layer
acceptance remain required.

## Two-rank device check

Run on two reserved, MTE-connected dies, visible to the same environment.
The SHMEM endpoint must differ from the torchrun rendezvous. The driver
owns one SHMEM context and keeps all storage live through graph deletion.

```bash
LD_LIBRARY_PATH="$SHMEM_BUILD/lib:$LD_LIBRARY_PATH" \
  torchrun --standalone --nproc-per-node=2 \
  tests/glm52/whole_layer/communication_devicecheck.py \
  --library tests/glm52/whole_layer/communication_build/communication_probe.so \
  --endpoint tcp://127.0.0.1:28765 --rows 1 12 32 --blocks 4 \
  --warmup 3 --replays 12 --iterations 10 --output communication-results
```

Each replay changes the input and alternates the delayed rank. The driver
requires exact agreement with its explicit ordered-sum/dyadic-projection
oracle, and records the separate difference from the installed
`torch.distributed` HCCL BF16 SUM. HCCL runs outside the candidate kernel
as a numerical comparison. Timings include both reductions and CUBE work
inside the captured probe; they are not a standalone collective or layer
latency. Raw per-rank JSON contains loaded library hashes, source hashes,
Torch/CANN/device identity, warmup, replays, individual samples, and errors.

The reduction loop waits on `MTE3_MTE2` after each UB-to-GM output copy.
The next tile's MTE2 load reuses the same narrow UB buffer, so waiting only
on `MTE3_V` leaves an overwrite race. Widths 12 and 32 exercise multiple
tiles per AIV and detect this dependency; width 1 alone does not.

## Three-rank reduction-order result

The unchanged `1be5fa37` probe is not byte-equivalent to installed HCCL
FP32 SUM on a three-rank cancellation fixture. On physical dies 4/6/7,
M1/M12 and 24 MIX groups, all 36 rank/shape/input cases match the native
ascending-rank FP32 oracle and its immediate consumer. HCCL instead matches
the alternative grouping `(x0+x2)+x1`; 4096 elements per row differ after
BF16 rounding. Two eager cases and four graph replays per shape show the
same distinction. The exact integer control column remains equal.

For epsilon = 2^-25, rank vectors `[1,1,1]`, `[epsilon,-1,2]` and
`[-1,epsilon,3]` produce native `[0,epsilon,6]` versus HCCL
`[epsilon,0,6]`, with six common power-of-two input scales. Both paths read
the same BF16 partial allocation; HCCL receives a fresh FP32 conversion of
that input and runs outside the native graph. Its 36 INFO launch records
confirm rankSize 3, counts 6144/73728, float32 SUM and deterministic mode.
All six algorithm-selection records identify
`AllReduceMeshAivFor91093Executor`. The diagnostic's communication header
is byte-identical to the C2 whole-layer header `cf9e9a84`.

This result rejects general byte equivalence between ascending-rank
accumulation and HCCL. It compares only the probe's first reduction and
does not establish HCCL's complete tree or TP16 behavior. The
[independent audit and complete compressed logs](../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/communication-order-three-rank-v1-a3-194/AUDIT.md)
retain every case, library identity, algorithm/launch line and original
log hash. The 20 raw files total 81,185,055 bytes; gzip decompression is
checked against the original byte hashes.
