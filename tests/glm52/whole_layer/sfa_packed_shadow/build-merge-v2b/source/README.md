# Packed shadow cache for indexed SFA

This candidate combines the latent and RoPE cache reads in the AIV gather. It
retains the vendor's physical-pair order, causal compaction, planar `kvMergeGm_`
layout, 512-token softmax blocks, CUBE computation, and cross-core flags. The
integrated v1 passes the recorded byte checks but is slower than the ordinary
indexed kernel on both measured widths and indexer modes. It is not promoted.
The complete mixed SFA instantiation in `compile_probe.cpp` compiled with
CANN 9.2.0 for `dav-2201`; its object SHA256 is
`83f280ed9cfc5983179ed720e95f7a97a5cf298adf8cb0d2f3809cb2c3fac0eb`.
Six CPU tests cover raw-byte address/stride obligations. The frozen probe
sources, overlay, compiler identity and logs are in `build-v1/`; this probe
has no host launcher and has not run on an NPU.

Create an independent include overlay:

```bash
python3 tests/glm52/whole_layer/sfa_packed_shadow/prepare.py \
  --source /path/to/recorded/attention_vendor \
  --output /path/to/new/sfa-packed-overlay
```

Place `-I /path/to/new/sfa-packed-overlay` before the ordinary vendor include
root and define `SFA_PACKED_SHADOW_KV`. The overlay contains all four SFA arch22
headers so quoted sibling includes resolve to its patched vector service. The
other three headers are byte-identical to the baseline. `prepare.py` checks
their input hashes, refuses an existing destination, applies a zero-fuzz patch,
and writes the input/output identities to `manifest.json`.

The original vector header is
`attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h`,
SHA256 `681ff5278aa25abaa2741fbbd9528aedba19394548f9d98be069f71046b341bd`.
The patched header is
`f5620780b20befe34f493c34ab3fc409272a36f1af5c1d1c7e567d0848291cd8`.
The patch touches class guards/constants, `InitBuffers`, `CopyInSingleKv`,
`CopyInKv`, and `CopyOutMrgeResult`. `MergeKv` and the subsequent vector
arithmetic are byte-identical to the original source. No CUBE or tiling
implementation is changed.

## Data contract and integration

The supported configuration is BF16 main caches, one KV head, 512 latent
channels, 64 RoPE channels, page size 128, sparse block size 1, `TND` queries,
`PA_BSND` caches, `V_TEMPLATE`, `FLASH_DECODE=false`, and `preLoadNum=2`.
The wrapper must enforce these runtime dimensions. C8/BF16 indexer modes both
use these BF16 main caches.

The separately owned shadow has ND storage `[physical_pages,128,576]`.
For physical row `p`, its first 512 elements equal the original CKV row and
its last 64 elements equal the original RoPE row, including their raw BF16
bits. The byte address of channel `d` is `shadow + 2*(576*p+d)`.
Only the private SFA invocation receives `key=shadow`; `value=original CKV`
and `keyRope=original RoPE` remain valid original pointers. Prolog receives
the original caches. The V-template CUBE reads the planar merge workspace;
its C-template direct-cache path is excluded by the overlay's static checks.

Initialize the entire shadow from the current physical main caches outside
capture. Retain its Tensor owner for every graph using it. Prefill writes,
page reuse, or cache restoration require the corresponding shadow refresh.
For this indexed prototype, after Prolog's existing `EndPhase`, AIVs update
the current physical slots during the first indexer-front projection phase:

1. Assign source rows by `row = AIV_index; row < M; row += 2*blocks`.
2. Read the current INT64 main-cache slot through `GlobalTensor<int64_t>`;
   skip negative slots, as the original
   Prolog scatter does. Read the final original CKV and RoPE rows into one
   independent 1,152-byte UB row. There is no arithmetic or cast.
3. Wait `MTE2_MTE3`, store the complete packed row, then wait `MTE3_MTE2`
   before reusing the UB row. Keep all AIC/AIV participants in the front's
   existing first `EndPhase` (`PIPE_ALL`, DDR, mixed `SyncAll`, `Reset`).

Prolog's preceding barrier publishes both original cache segments. The front's
existing barrier publishes the shadow before the selector and SFA. The packed
row's temporary UB ends at that barrier; no UB value survives `Reset`. This
placement requires no new global barrier or cross-core flag. It is not an
implementation for the 57 shared-indexer layers, which bypass the front.

## Gather order and byte layout

`GetKeyGmOffset` retains the original page-table mapping and causal limit.
The original `CopyInKv` branch predicate is unchanged: duplicate/negative
strides, original stride overflow, and either logical row reaching the causal
tail use two scalar-address copies in original pair order. The ordinary fast
path emits valid rows in ascending physical-address order. These orders can
differ when the page table is permuted; sorting every pair would change the
attention reduction sequence.

The packed fast path uses the same chosen order and reads 1,152 bytes per row.
Its GM stride is `(abs(p1-p2)-1)*1152` bytes. If this larger stride exceeds the
vendor's signed 32-bit limit while the original 1,024-byte stride did not,
two packed reads retain the original fast path's physical order. A single
valid row uses zero stride. Original fallback copies remain in original pair
order. `mte2Size`, 32-row flushes, AIV half ownership, zero padding, valid-size
publication, and all `MergeKv` events remain unchanged.

Each ping/pong UB block holds 32 packed rows: 36 KiB. `CopyOutMrgeResult`
deinterleaves them without vector arithmetic:

| Segment | UB source | Block length | UB source gap | GM destination |
| --- | --- | --- | --- | --- |
| CKV | `ping*32*576` BF16 elements | 1,024 B | 4 data blocks = 128 B | original contiguous `[512,512]` region |
| RoPE | `ping*32*576+512` BF16 elements | 128 B | 32 data blocks = 1,024 B | original contiguous `[512,64]` region |

Both copies use `blockCount=mte2Size-mte3Size`, zero GM destination stride,
and the original `loop%4` plane and AIV row offsets. All source starts and
row strides are 32-byte aligned. In CANN's A3 UB-to-GM `DataCopyPad` contract,
the UB gap is measured in 32-byte blocks and the GM gap in bytes. See
`asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.md:103-115`.

## UB and synchronization budget

| Buffer | Bytes per AIV |
| --- | ---: |
| `inputBuff1`, two packed 32-row blocks | 73,728 |
| `inputBuff2`, retained original allocation | 16,384 |
| `outputBuff1` | 32,768 |
| `outputBuff2` | 4,096 |
| `tmpBuff1` | 32,768 |
| `v0ValidSizeBuff` | 8,192 |
| `nValue`, `cofValue`, `aMlaSum`, `softmaxMax`, `softmaxExp`, `softmaxSum`, each 2,048 | 12,288 |
| default softmax max/sum, each 1,024 | 2,048 |
| Total | 182,272 (178 KiB) |

This adds 8 KiB to the original 170 KiB and stays below the composed kernel's
188 KiB lower-UB budget. The separate 1,152-byte front-update buffer belongs
to an earlier phase and is reset before SFA. It is not added to this total.
`inputBuff1` remains the later Vec1/Vec2 scratch, whose original 0/32-KiB
offsets are unchanged. `MergeKv` drains both `MTE3_MTE2` ping/pong events
before returning; its strided output copies finish before later vector reuse.
Existing vector-to-MTE3 completion points still precede the next gather.

Let `q=mte2Size-mte3Size` before a pair. The original flush test ensures
`q<=30`; one pair adds at most two valid rows. Thus the write ends at or
before row 32. Counts 31 and 32 flush immediately because `q+2>32` after
the pair. Zero-valid pairs and a negative first-index sentinel cannot
increase the count. The final byte of either packed ping/pong is at most
`2*32*576*2=73728`. The two AIVs retain separate UB and disjoint 256-row
output halves. `keyGm.SetGlobalBuffer` in the original kernel takes only
the pointer, without a 512-channel element limit; `InitVec0GlobalTensor`
copies that descriptor without narrowing its bounds.

The patch adds no event IDs: the gather retains `MTE2_MTE3(0)`,
`MTE3_MTE2(0/1)`, its zero-fill events, and valid-size `S_MTE3(1)` /
`MTE3_S(6)`. The outer four-plane CUBE release flag 3 and ready flag 6
remain unchanged, as do compute flags 4 through 9. No `SyncAll` is added.

## Required device gates and measurement

Before the first launch and after every changing-input replay, compare the
entire shadow's raw bytes with `concat(original CKV, original RoPE)` over
all physical pages. Include permuted page tables, causal partial tails,
negative slots, changed slot positions, and zero-input signed-zero values.
Then compare all existing indexed production boundaries, top-k entries,
main/indexer caches, latent output, and final BF16 output with the accepted
ordinary kernel. The shadow comparison must remain independent of SFA output.

Use independent graph captures and at least 30 paired event samples per arm.
The event scope includes the per-replay shadow update and all original indexed
phases; one-time full-shadow initialization is reported separately. Match
weight-bank count, original scale dtype, sequence length, widths and arena
placement. Retain both baseline and candidate artifacts and all source hashes.

At 8K, the extra shadow is 9 MiB per physical 64-page cache. Direct
`torch.cat((CKV, KR), out=shadow)` has a logical read-plus-write payload of
`2*S`, or 18 MiB; a temporary `cat` followed by `shadow.copy_` has `4*S`,
or 36 MiB. Here `S=physical_pages*128*576*2`; the 68-page numerical fixture
uses 9.5625 MiB of shadow and 19.125 MiB for direct packing. The recorded
device gate supports direct `out=shadow` and checks its complete raw bytes.
These are logical copy payloads, not measured HBM traffic. Updating M rows
transfers `2304*M` bytes (27 KiB at M=12).
A full 2,048-token selection uses 1,024 packed pair reads
instead of 2,048 separate CKV/RoPE reads, without reducing selected payload
bytes or planar merge writes. The maximum exposed benefit is the unhidden
gather/ready wait, not the entire SFA phase. Current five-stage intervals
include barriers and do not isolate this term.

## Recorded v1 result

The integrated library SHA256 is
`32eba988e4356074a77ca863a24552ac5a23f0adb635d12b895cf6d3f2ebd430`.
[Numerical checks](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-shadow-numerics.json)
cover M=1/2/12/32, 16/24 core groups, both indexer cache modes and eight changing
replays. Six additional [negative-slot cases](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-shadow-negative-slots.json)
pass. Gates include all production boundaries, all original caches, and the
entire packed shadow. Numerical cases use one weight bank; the eight-bank
timing graph retains the final call's boundaries, not every call's outputs.

The same-arena comparison recaptures ordinary and candidate on each of two
arenas with identical scratch, tiling, output and cache addresses. It uses
30 paired event samples per arena. Negative values below mean the candidate
is slower; each case has zero candidate wins out of 30.

| M | Indexer mode | Paired median ordinary minus candidate, arena A | Arena B |
| ---: | --- | ---: | ---: |
| 12 | C8 | -17.525 us | -14.867 us |
| 12 | BF16 | -21.654 us | -20.856 us |
| 32 | C8 | -13.596 us | -11.120 us |
| 32 | BF16 | -16.518 us | -14.386 us |

Raw [M12](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-shadow-m12-bank8.json)
and [M32](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-shadow-m32-bank8.json)
reports retain independent-arena comparisons, same-arena samples, source
identity and pre/post byte gates. Reduced input DMA count does not establish
a net benefit; these measurements do not isolate which transfer caused the
regression.

## One-time packing probe

`pack_check.py` uses two original BF16 ND caches and one separately allocated
ND shadow. It verifies full raw bytes, unchanged sources and output addresses
before timing, checks changed graph inputs, and verifies the full result again
after timing. Signed-zero and NaN payload witnesses test copying without a
floating-point equality shortcut.

```bash
python3 tests/glm52/whole_layer/sfa_packed_shadow/pack_check.py \
  --device npu:0 --pages 68 64 --samples 30 --warmup 5 \
  --output /workspace/logs/sfa-shadow-pack.json
```

Each independent event contains one direct `cat(out=shadow)` or one captured
graph replay, in alternating order. Allocation, compilation, capture, warmup
and byte checks are outside the event. The report contains every sample,
p50/p90/p99/std, runtime/device identities and the Torch-NPU library hash.
Both arms reuse warm, fixed addresses; this measures the prewarmed operation,
not cold initialization, allocator cost or per-decode updates. A cache reused
for N decode calls costs `T_pack + N*T_candidate`; an amortization claim needs
measured `T_pack`, a positive same-run per-call saving and the actual cache
lifetime. Prefill writes and page reuse can require additional refreshes.
