# Captured M12 WK dispatch

The M12-only indexed-attention build `5f8868bf` passed the original Decoder
layers 6/7 at M1/2/12/32 with eight changing inputs per width: all 816 raw-byte
boundaries matched. Both arms retained the official FFN and explicit local
TP cuts. [The device summary](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-wk-m12-int8-wide-v1-a3-194/summary.json)
records the exact libraries and preserved weight banks; graph replay is a
separate gate.

The full Decoder diagnostic first differs at the BF16 WK projection, at
`(5,95)` and `(6,115)`. Its input and weight bytes, query path, head weights
and key scale match. LayerNorm, RoPE, Hadamard and one INT8 key code then
differ. The accepted M32 K-order branch does not run for M12.

`dispatch.py` executes one original `F.linear(hidden, NZ_weight, bias=None)`
on the retained BF16 `[12,6144]` input and `[160,6144]` weight. All 1,920
outputs must reproduce the captured vendor projection. CANN DEBUG logs retain
the actual kernel and tiling; the report records physical formats, transpose
strides, storage offsets, library hashes and finite rounding witnesses.
CPU FP64 dots are auxiliary observations, not the acceptance oracle.

The launcher uses the recorded TorchNPU 2.12 image and CANN 9.2, one exposed
die, private temporary caches and a persistent driver-log directory. The
controlling runner must admit the die and release its previous namespace.

```bash
bash /home/shanshan/tilert-megakernel-20261021/wk-m12-dispatch-v1/run_dispatch.sh \
  4 captured-wk-m12-dispatch-v1
```

The frozen input SHA is checked before device initialization. This probe
does not modify or load the native attention library.

The observed CANN 9.2 dispatch is MatMulV2 key `229585`: ten N16 tasks,
twelve K512 segments and four K128 Mmad updates per segment. The decoded
152-byte payload, actual fatbin identity and finite witnesses are in the
[source audit](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-m12-source-audit/README.md).

`kshift_kernel.cpp` preserves one L0C accumulator through all 48 updates.
The shifted arm uses `((j%2 ? 11-i : i) + 2*floor(j/2)) % 12` for the
outer K segment. `kshift_check.py` first reproduces the original `F.linear`,
then requires natural K order to reproduce all 1,920 captured FP32
accumulators. It compares every shifted BF16 output, checks the four guard
rows by raw bytes, and repeats both orders three times. No timing is taken.

```bash
bash /home/shanshan/tilert-megakernel-20261021/wk-m12-kshift-v1/run_kshift.sh \
  4 captured-wk-m12-kshift-v1
```

This isolated probe leaves all previously built attention libraries and the
accepted M32 branch unchanged.

The isolated device run passed three interleaved repetitions per arm: natural
order reproduced all 1,920 retained FP32 accumulators; shifted order reproduced
all 1,920 vendor BF16 outputs. Four guard rows remained unchanged.
[Its summary](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-m12-kshift-v1-a3-194/summary.json)
retains both finite witnesses. `native.h` applies that traversal to the existing
KN32-packed weights only when `rows == 12`; `build_indexed.py` preserves the
M32 branch and every other front-end operation.
