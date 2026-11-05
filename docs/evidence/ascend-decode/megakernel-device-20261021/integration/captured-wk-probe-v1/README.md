# Captured WK projection discriminator

This probe replays the actual BF16 `[32,6144]` input and `[160,6144]` weight
captured at the first differing production boundary. It compares all 5,120
outputs with both retained implementations. The source capture comes from
the BF16 v13 outer-forward diagnostic; C8 reaches the same four WK differences.
It neither runs a replacement attention layer nor measures latency.

The first arm must reproduce `F.linear(hidden, NZ_weight, bias=None)` exactly.
The remaining arms compare full N=160 with five N=32 calls, NZ with ND weights,
and the original `[N,K]` transpose view with contiguous `[K,N]` ND storage.
Two `aten.mm.dtype` calls request FP32 output from the same BF16 operands.
The full-N NZ BF16/FP32 pair and the split-N contiguous ND BF16/FP32 pair reuse
their exact input addresses. Every arm runs three times in alternating order.
The report records physical formats, both weight and right-operand strides,
storage offsets, source hashes, output hashes, all byte differences and the
four finite rounding witnesses. Outputs are retained as CPU tensors.

These comparisons distinguish three mechanisms in this order:

1. Geometry or layout changes the installed matmul route. A full-N arm that
   reproduces production while a split-N arm reproduces native would justify
   a matching native geometry experiment. ND and NZ comparisons separate the
   format effect; contiguous `[K,N]` also separates transpose handling.
2. K accumulation differs within the selected route. The native Matmul API
   uses one full K=6144 product per N=32 tile. Installed CANN MatMulV3 also has
   single-core and multiple-core split-K implementations. A captured tiling
   key or matching native experiment is still required to identify the active
   implementation; the presence of those source branches alone does not do so.
3. Direct BF16 output differs from FP32 output followed by BF16 rounding.
   All native outputs already equal the RNE cast of their saved accumulators.
   Requested output dtype can itself change CANN tiling, so the stock FP32
   pair does not by itself isolate Fixpipe rounding.

Unsupported BF16-input/FP32-output execution fails with the original exception
after preserving completed cases; the probe does not substitute FP32 inputs.
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
  4 captured-wk-v1
```
