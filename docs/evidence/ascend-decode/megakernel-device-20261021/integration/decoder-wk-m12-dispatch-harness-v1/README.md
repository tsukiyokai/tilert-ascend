# Captured M12 WK dispatch

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
