# BF16 transport with an FP32 tree

This is a protocol design, not an implementation or performance result. It
changes the communication ownership and window layout; it is separate from
the tree-only candidate `a0bd2462`. The current C2 all-to-all path sends each
rank's entire BF16 partial to every rank. A reduce-scatter/all-gather path can
send each element to one reduction owner, evaluate the same FP32 tree there,
round once to BF16, and distribute the finished BF16 bits.

The arithmetic cannot use BF16 partial sums between intermediate tree nodes.
That would introduce extra rounding. All original rank operands for an owned
element must remain BF16 until individually widened; all internal sums stay
FP32 in the five-slot program. All-gather performs no arithmetic.

## Partition and line ownership

Let `E=M*6144`, `L=256` BF16 elements per 512-byte payload line, `B=E/L`,
and `P` be the rank count. Define `q=B//P`, `r=B%P`. Owner `o` receives
`q+(o<r)` consecutive lines, starting at `o*q+min(o,r)`. This partitions all
elements without padding an actual reduction or splitting a payload line.
For the supported `M>=1`, `2<=P<=16`, every owner has at least one line.

At TP16/M1, a uniform 384-element slice would make adjacent owners share
512-byte lines. The proposed partition instead assigns 512 elements to
owners 0–7 and 256 to owners 8–15. M12 gives every owner 4608 elements;
M32 gives every owner 12288. Local reduction work is split into at most
2048-element tiles, with a final multiple-of-256 tail. Each tile belongs to
one AIV, including its BF16 store and publication. Output copy tiles also
have one AIV writer and boundaries on 256-element lines.
The output tensor's base must satisfy the same 512-byte alignment guard;
offset-aligned tiles alone do not isolate physical lines of a shifted view.

Window payload bases and plane strides must be 512-byte aligned, not merely
the parent's 64-byte flag alignment. Reusing the parent's layout unchanged
would violate this: its payload offset at P16 is 2112 bytes, or 64 modulo 512.
A new allocation must either prove 512-byte base alignment or use the
recorded SHMEM `aclshmem_align(512, bytes)` API. The latter requires explicit
local zeroing, stream completion and an all-rank host rendezvous before first
use. Allocation, zeroing and rendezvous remain outside graph capture. Retain
the original allocation owner through every graph and queued operation.

## Symmetric storage

Capacity `C` is fixed for a window. Set `S=ceil((24*C)/P)*512` bytes. Each
scatter plane has `P` sender slots of stride `S` on every rank. Sender `s`
writes owner `o`'s input slice to remote slot `s`; every peer uses the same
offset because the stride is based on capacity, not its actual owned length.
For a call, only that owner's actual line count is valid. Padding is never
read as an operand. A zero-padded numerical sum is not a valid substitute.

Each gather plane holds the complete `2*C*6144`-byte BF16 output. Owner `o`
writes only its actual global slice at every peer. Slices are disjoint in
both bytes and payload lines. Its local result may occupy that same slice
in its own gather plane; skip the self broadcast to avoid a source/destination
overlap. Scatter still copies the local input to its distinct sender slot.

Keep two planes of each payload region, plus separate scatter-ready and
gather-ready arrays for both parities. Each sender/owner flag retains its own
64-byte slot. Round the whole flag region up to 512 before the first payload.
At `C=32,P=16`, each scatter and gather plane is 393216 bytes; the aligned
header is 4608 bytes, giving 1577472 bytes per rank. The current C2 window is
12585024 bytes. This size reduction requires the new layout and does not
follow from swapping the reducer alone.

## Five-phase conservative schedule

All resident AIC and AIV cores execute the same five mixed-core joins. AICs
participate in those joins but issue no communication or vector arithmetic.

1. One AIV advances the existing local epoch. Publish it and join; all AIVs
   read the same epoch and select its parity. The window is bound to one
   serialized stream. Bound its lifetime before uint32 epoch wrap.
2. AIVs copy the input slices to their reduction owners, using the explicit
   narrow UB buffer and `quiet` after each put. Drain and join before AIV0
   sends `scatter_ready[s]=epoch` to every owner. A ready value must follow
   completion of every writer for that sender, not merely submission.
3. Each owner waits for all sender flags, evaluates the accepted FP32 tree
   for its assigned tiles, and stores its BF16 result in its gather slice.
   Every load/cast/add reuses the tree candidate's fences; every tail is a
   multiple of 256 elements. Drain GM stores and join before broadcast reads
   any other AIV's reduced tile.
4. AIVs broadcast their local owned slice to the other ranks. Each peer's
   gather location is symmetric and line-disjoint from other owners. Drain
   all puts, join, then AIV0 publishes `gather_ready[o]=epoch` to all ranks.
5. Each local AIV waits for the gather flags covering its output tiles, copies
   them to the actual non-SHMEM output tensor, drains its writes and joins.
   The caller may then reset the communication arena or begin its consumer.

This first schedule has two ready waves and five mixed joins, compared with
one wave and three joins in the accepted all-to-all path. Per-tile
reduction/broadcast overlap could remove a join only after a separate
producer/publication proof; it is not assumed here.

The five FP32 slots and one narrow buffer still occupy 44 KiB. No compute
arena remains live when the phase begins. Scatter, reduction and broadcast
use that allocation sequentially. A GM store must be drained before its
source is read by a put; a final put must finish before its UB is reused.
The accepted MTE3→MTE2 tile fence remains necessary even with smaller tails.

## Reuse and teardown

Two planes do not by themselves prove safety. The protocol's causal edges do:
rank `s` can publish scatter-ready for epoch `e+1` only after its local output
copy for `e` completed. Thus an owner that sees every scatter-ready for
`e+1` knows every rank finished reading gather plane `e`. A sender cannot
complete `e+1` until every owner published its gather-ready for `e+1`; those
publications require all owner reductions for `e+1` to have completed. Hence
a sender starting `e+2` cannot overwrite scatter plane `e` while any owner is
still reading it. This also excludes an unbounded fast-rank lead. Ready words
for the next use of a parity cannot replace an epoch still being waited on
without the lagging rank having passed the preceding necessary publication.

Every rank must call all reductions in the same order, with identical
capacity, row count, partition geometry and epoch count. Otherwise a single
ready flag cannot identify the intended data layout. Graph replay updates
inputs and metadata but never resets epochs. Destroy all graphs, synchronize
the bound stream, rendezvous all ranks, then release symmetric windows in
the same order and finalize SHMEM once. A second stream requires a distinct
window or explicit cross-stream serialization.

## Payload and costs at TP16

The average off-rank payload per rank is `4*E*(P-1)/P` bytes: BF16 scatter
plus BF16 gather. The current path uses `2*(P-1)*E`, eight times as much at
P16. For unequal slices, rank `o` sends `2*E + 2*(P-2)*owned[o]` remote bytes;
the average formula must not be presented as every rank's exact M1 traffic.

| Rows | Current bytes/rank | RSAG average bytes/rank | Owner lengths | Reduction tiles/owner |
|---|---:|---:|---|---|
| 1 | 184320 | 23040 | 8×512, 8×256 | 1 |
| 12 | 2211840 | 276480 | 16×4608 | 3 |
| 32 | 5898240 | 737280 | 16×12288 | 6 |

M1 ranks owning 512 elements send 26624 remote bytes, while those owning 256
send 19456. M1 also leaves only one AIV reduction tile per owner; M12/M32 use
three/six. Network payload falls, but the extra publication wave, two joins,
uneven M1 load, and gather-plane-to-output copy may dominate small messages.
The gather copy adds a full local BF16 read and write. Sender traffic alone
therefore cannot establish a latency gain.

The next falsifiable gates would be partition/line-address ownership for all
supported geometries, rank-skewed replay with a single non-reset window,
the original serial/tree witnesses and edge columns, and all production
boundaries under the actual FP32 HCCL baseline. The current three-rank tree
gate does not test NaN-producing four-way overflow trees or TP16 topology.
No RSAG source or device candidate is created by this design note.
