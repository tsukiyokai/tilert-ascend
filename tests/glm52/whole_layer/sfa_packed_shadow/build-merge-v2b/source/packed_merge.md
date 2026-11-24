# Packed merge workspace variant

This variant removes the packed-gather candidate's strided UB-to-GM
deinterleave. It keeps both the shadow cache and the SFA merge workspace in
row-major BF16 `[row,576]` storage. It tests whether that transfer/layout
change improves the complete indexed chain; v1's measured regression does
not establish the deinterleave as its cause.

`prepare_merge.py` applies the unchanged `packed_gather.patch` and then the
independent `packed_merge.patch` to a fresh private overlay. The input vendor
hashes, v1 output hashes, v2 output hashes, both patches and both preparation
scripts are recorded. The v1 source files and `build-v1/` remain intact.

```bash
python3 tests/glm52/whole_layer/sfa_packed_shadow/prepare_merge.py \
  --source /path/to/recorded/attention_vendor \
  --output /path/to/new/packed-merge-overlay
```

The compile macro remains `SFA_PACKED_SHADOW_KV`. The five-cache native ABI,
shadow initialization and per-call current-row update remain identical to v1.
The ordinary and v1 libraries must retain their original overlays.

## Transfer contract

For plane `p`, selected row `r` and channel `d`, the new merge address in BF16
elements is `p*512*576 + r*576 + d`. The four planes keep their original size
and ownership. `CopyOutMrgeResult` writes its `count<=32` rows as one contiguous
`count*1152`-byte transfer from the existing packed ping/pong UB. AIV halves
still own disjoint row intervals, and the final valid byte is bounded by the
end of that plane. The original MTE2-to-MTE3 and ping/pong reuse events remain.

`MergeKv` still computes the same physical-pair order, causal compaction and
valid counts. Its invalid tail now writes one all-zero 576-element row per
slot. The original zero-fill events precede these writes. Initializing only
512 elements would leave invalid RoPE values in the new row layout. The
valid-size publication and softmax masks are unchanged.

The CUBE GM-to-L1 `Nd2NzParams.srcDValue` becomes 576 elements. Its row count,
column count, NZ destination strides and L1 destination addresses are
unchanged. All source starts remain 32-byte aligned. The A3 ND-to-NZ contract
allows `srcDValue` in `[1,65535]`; it describes the source row pitch, independently
of the selected column count. See
`asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_GMToL1_ND2NZ.md:85-95`.

| CUBE transfer | Packed source within row | Width | Original L1 destination |
| --- | ---: | ---: | --- |
| QK first K-L1 block, CKV | 0 | 256 | 0 |
| QK first K-L1 block, RoPE | 512 | 32 | `256*aligned_rows` |
| QK second K-L1 block, CKV | 256 | 256 | `32*aligned_rows` |
| QK second K-L1 block, RoPE | 544 | 32 | 0 |
| PV N tile `n`, row tile `k` | `128*n` | 128 | original `(k-k_offset)*128*128` |

In QK, the first L1 block remains `[CKV0:256,KR0:32]` and the second remains
`[KR32:64,CKV256:512]`. Each 288-column block is consumed in the original
three 96-column K-L0 steps. Reading two natural contiguous 288-column slices
would change the operands and their reduction order. In PV, the row is still
`k*128`; only its source pitch changes from 512 to 576. L1/L0 allocations,
LoadData, MMAD, Fixpipe, scale, softmax and accumulation order are unchanged.

The two v2 output headers are:

- Vector: `5c65d6676b856a1e0ef26598b9a9cecfe1ab27db7139d922765ecca5fd6a8847`.
- CUBE: `a99ad8dab2ac1156c5635a4072de58d7338018d94d1aa67d3f98fb6d3acbfc38`.

`prepare_merge.py` additionally checks unchanged post-merge vector arithmetic,
unchanged synchronization calls, unchanged `MergeKv` outside zero-fill, and
unchanged QK/PV functions outside their V-template source-load branches.
The no-RoPE path remains excluded by the packed overlay's type assertions.

## Resource and validation boundary

UB remains 178 KiB per AIV, including the original unused RoPE merge buffer.
GM remains four 512-by-576 BF16 planes per AIC group; L1 and L0 allocations do
not change. No event ID, cross-core flag, global barrier or new dependency is
introduced. Source-load strides can affect transfer performance even when
the operands are identical, so byte equivalence does not predict a speedup.

`test_merge_contract.py` checks every flush count 0..32 in both ping/pong
buffers and both AIV halves, all four planes, QK/PV L1 bytes for partial row
tiles and independent invalid tails, and the six QK reduction-step column
orders. These are CPU address/operand checks. The complete mixed SFA path
must also compile with the recorded CANN toolkit, then pass the unchanged
full-cache and production-byte device gates before performance comparison.
