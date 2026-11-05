# M12 WK dispatch and K traversal

The captured full Decoder failure first differs at two BF16 WK outputs. The
original `F.linear` replay reproduced all 1,920 vendor output elements on
a3-194 with TorchNPU 2.12 and CANN 9.2. The operands are BF16 `[12,6144]`
and NZ BF16 `[160,6144]`; the latter retains logical stride `[6144,1]`.
The exact input artifact is identified in [cpu-witnesses.json](cpu-witnesses.json).

[dispatch.json](dispatch.json) decodes all 38 UINT32 fields of the actual
152-byte tiling. The following eight bytes are DFX metadata, excluded from
[tiling-152.bin](tiling-152.bin). The installed launch is MatMulV2 key
`229585`, 10 AICs, `m_dim=1`, `n_dim=10`, `k_dim=1`, `split_k_flag=0`,
`datatype_bf16=1`, and `close_k_shift=0`.

The K tiling is 12 outer segments of 512 elements, each containing four
K128 Mmad operations. For N16 tile `j=0..9`, the installed kernel computes
the outer segment as

`q(i,j) = ((j % 2 ? 11-i : i) + 2*floor(j/2)) % 12`.

The witness columns 95 and 115 belong to tiles 5 and 7, whose first segments
are 3 and 5. This differs from the previously tested M32 segmentation.
The kernel's runtime entry minus its binary base is `0xc2bc8`, matching the
key-229585 disassembly entry. [source-anchors.json](source-anchors.json)
records the actual a3-194 fatbin SHA, matching installed schedule/schema
source SHAs, and bounded excerpts. The branch emits `MMAD.bf162f32`; its
`FP16_FP16` symbol spelling does not identify the actual operand dtype.

| Coordinate | Native FP32 accumulator | Native BF16 | Vendor BF16 | BF16 midpoint |
| --- | ---: | ---: | ---: | ---: |
| `(5,95)` | `0xbd657fe8` | `0xbd65` | `0xbd66` | −0.0560302734375 |
| `(6,115)` | `0xb9937200` | `0xb993` | `0xb994` | −0.00028133392333984375 |

The retained FP32 accumulators lie respectively `8.940696716308594e-8`
and `1.043081283569336e-7` above these midpoints. CPU FP64 dots are retained
as auxiliary rounding witnesses; matching them does not establish the CANN
BF16 contract.

The independent native probe keeps one L0C tile through all 48 K128
updates. Its acceptance requires the natural order to reproduce every
captured native FP32 accumulator before interpreting the shifted arm.
The shifted arm must reproduce all vendor BF16 outputs, with unchanged
four-row output guards and repeatability. The real CANN build and CPU
contracts are recorded in [the build evidence](../captured-wk-m12-kshift-build-v1/manifest.json).
That build alone does not establish the device result or a full Decoder fix.

[The subsequent device probe](../captured-wk-m12-kshift-v1-a3-194/summary.json)
passed both whole-tensor gates in three interleaved repetitions, with unchanged
four-row guards. The shifted witness accumulators are `0xbd658032` and
`0xb9938540`. The M12-only integration in indexed library `5f8868bf` then passed
[all 816 original Decoder boundaries](../vendor-hook-decoder212-wk-m12-int8-wide-v1-a3-194/summary.json)
for M1/2/12/32 and eight changing inputs per width. This is a rank-local eager
comparison with the official FFN; TP16 reduction and whole-Decoder graph replay
are separate validation scopes.
