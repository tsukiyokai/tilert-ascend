# Captured WK projection discriminator

[The seven-arm device probe](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-v2-a3-194/wk-projection.json)
reproduces production for every full-N160 arm and native for every split-N32
arm, across three repetitions. NZ/ND and transpose storage do not alter these
groups. The original seven BF16 arms complete in v1, followed by an unsupported
`aten.mm.dtype` request reaching the PrivateUse1 CPU fallback. V2 removes only
the two FP32-output arms and completes all 21 cases.

[The native geometry probe](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-native-v1-a3-194/summary.json)
does not reproduce production with full N160. Its N32 control and N160 candidate
both match all 5,120 captured native FP32 accumulators, in three repetitions.
The candidate changes baseK from 512 to 96, stepKa/stepKb from 1/12 to 64/2
and iteration order from 0 to 1. None changes an accumulator byte. All four
BF16 differences remain. Stock's geometry dependence therefore does not imply
that the native full-width Matmul API has the same accumulation contract.

[The TorchNPU 2.12 FP32 probe](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-fp32-v1-a3-194/summary.json)
completes 27 cases. Its seven BF16 arms retain the original 2.10 grouping.
Both stock FP32 arms, full-N160 NZ and split-N32 contiguous KN, reproduce all
5,120 captured native accumulators in each of three repetitions. Full-N160
FP32 followed by BF16 rounding therefore differs from direct BF16 output at
the same four coordinates; split-N32 does not.

[The installed dispatch and schema](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-dispatch-source-audit/dispatch-schema.json)
identify different implementations for the two output dtypes. BF16 uses
MatMulV2 key 230097 with 152 bytes of tiling and no workspace, on 20 cores for
N160 and four cores for N32. FP32 uses MatMulV3 key 65536 with 280 bytes of
tiling, 24 cores and a workspace. Both BF16 tilings have `k_dim=1` and
`datatype_bf16=1`; the compiled kernel contains `MMAD.bf162f32`. Its filename's
`FP16_FP16` token does not describe the active arithmetic branch.

The BF16 tilings use M16/N16 tiles, K256 L1 segments and K128 Mmad operations.
For full N160, `core = 2 * n_tile + m_tile`. With 24 K segments, the recorded
kernel traverses segment
`q(i) = ((n_tile % 2 ? 23-i : i) + 4 * floor(n_tile/2)) % 24`.
This formula is scoped to these recorded tilings: no overlapping tiles,
`close_k_shift=0`, 20 cores and a larger N core dimension of ten.
N32 has a largest core dimension of two, which disables this shift and gives
`q(i)=i`. All four differing coordinates are in odd N tiles:

| Coordinate | Core | M tile | N tile | First K256 segment |
|---|---:|---:|---:|---:|
| (0,29) | 2 | 0 | 1 | 23 |
| (6,153) | 18 | 0 | 9 | 15 |
| (19,51) | 7 | 1 | 3 | 3 |
| (31,25) | 3 | 1 | 1 | 23 |

The source excerpts retain the 38-field schema and BF16 instruction selection;
the disassembly excerpts retain the axis selection, reversal and rotation.
Raw logs remain outside the repository with hashes. Word 31 is `hf32_flag`,
not a hash: its recorded values differ, but the BF16 schedule supplies
`hf32=0` and selects arithmetic with `datatype_bf16`.

The probe replays the actual BF16 `[32,6144]` input and `[160,6144]` weight
captured at the first differing production boundary. It compares all 5,120
outputs with both retained implementations. The source capture comes from
the BF16 v13 outer-forward diagnostic; C8 reaches the same four WK differences.
It neither runs a replacement attention layer nor measures latency.

The first arm must reproduce `F.linear(hidden, NZ_weight, bias=None)` exactly.
The remaining arms compare full N=160 with five N=32 calls, NZ with ND weights,
and the original `[N,K]` transpose view with contiguous `[K,N]` ND storage.
Every arm runs three times in alternating order.
The report records physical formats, both weight and right-operand strides,
storage offsets, source hashes, output hashes, all byte differences and the
four finite rounding witnesses. Outputs are retained as CPU tensors.

`kshift_kernel.cpp` tests the decoded K order using 20 independent M16/N16
tiles. Each tile retains one FP32 L0C buffer across all 48 BF16 K128 Mmad
operations. Both arms use the same buffers, instructions and fences; only the
K256 segment order differs. Final FP32 and BF16 outputs come from that same
L0C. There is no GM partial-product reduction. The natural-order arm must
first match all 5,120 retained native FP32 accumulators; the shifted arm must
match all production BF16 outputs. [The device result](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-kshift-v1-a3-194/summary.json)
passes all six calls on a3-194 with CANN 9.2 and TorchNPU 2.12. Across three
repetitions, natural order matches all retained native FP32 accumulators and
retains the four BF16 differences; shifted order matches all 5,120 production
BF16 outputs. Each direct BF16 output also equals the RNE cast of that arm's
FP32 output. This establishes the K-order mechanism for the captured M32
projection; complete attention acceptance is a separate gate.

`completed` means the experiment ran with exact production reproduction and
stable repeated results. Experimental arms are allowed to differ; this is not
a production numerical acceptance flag.

CPU contract check:

```bash
python check.py --cpu-only --capture /path/to/outer-index-diagnostic.pt \
  --output /path/to/new-cpu-report
```

The launcher uses the fixed TorchNPU 2.10 image, CANN 9.2 and one exposed device.
Its private 12 GiB memory limit accommodates first-use CANN compilation.
The controlling runner must admit the die and release any previous namespace
before invoking it:

```bash
bash vendor-hook-runtime/tests/glm52/whole_layer/indexer_wk_probe/run.sh \
  4 captured-wk-v2
```

The separate native probe uses the accepted `Project<BF16,FP32>` template
without changes. Its N32 control links the original frozen tiling object
(`fe1c6036…`) and must reproduce all 5,120 retained FP32 accumulators before
interpreting the N160 candidate. The candidate uses explicit contiguous
`[1,6144,160]` weights and a fresh Matmul API tiling. Both dump their complete
512-byte tiling buffers and named tiling fields. Output conversion to BF16
is performed on CPU after synchronization; this isolates the projection
accumulator and leaves LayerNorm, RoPE and production kernels unchanged.

`check_fp32.py` preserves the seven BF16 controls and uses the separately
validated TorchNPU 2.12 entry `torch.mm(..., out_dtype=torch.float32)` for full
NZ and split contiguous-KN inputs. Every repetition checks the full BF16
partition before either FP32 arm executes. The report compares the complete
FP32 output with retained native accumulators, its BF16 cast with the matching
stock BF16 arm, and the four finite witnesses. `run_fp32.sh` uses the fixed
2.12 image and a private `/root/.tvm_test_data` tmpfs. This is an output-contract
probe; an output-dtype change may also change dispatch.
