# A8 operation order

Run: 2026-10-16, `a3-187:trt-asc`, CANN 9.2.0 master20260917.
The installed dynamic quantizer computes the code multiplier separately
from its returned scale:

```python
maximum = x.float().abs().amax(-1)
denominator = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
factor = torch.div(torch.full_like(denominator, 127), denominator)
codes = (x.float() * factor[..., None]).round().clamp(-127, 127).to(torch.int8)
scale = maximum * (1 / 127)
```

Zero rows produce zero codes and zero scales. Tensor/tensor division is
required in the CPU formula: Torch scalar `127 / maximum` may lower to a
reciprocal followed by multiplication. `reciprocal(scale)` introduces a
different rounding boundary.

For ordinary BF16 DynamicQuant, the [A3 Div calibration](dynamic-quant-div.md)
adds the measured native-factor correction to this CPU formula. It covers
all 128 mantissas over exponents −120 through 127 and closes the first 8K
indexer difference. The C8 CPU formula remains separate.

The [real layer-1 counterexample](a8-operation-order-20261016.json) has
`maximum=0.0032196044921875` and `x=±0.00160980224609375`. CANN returns
`±64`; the old reciprocal-scale expression produces `±63.4999961853` and
rounds to `±63`. Three output-projection activation codes differ on rank 6.
Freezing the actual INT8 codes/scales makes all sixteen W8 output
projections bitwise equal to the integer-dot formula.

The source-derived formula matches [184 device cases](a8-operation-order-fixed-20261016.json)
bitwise, covering real M1/M4 inputs plus zero/tie rows at K128/512/768/1024/
2048/6144. [Frozen CPU replay](a8-reference-parity-20261016.json) verifies
both CPU quantizers and the HF leading dimensions against the retained
device tensors. The [expanded operator check](check-weights-msd-a8-20261016.json)
passes 189/190: one random M4/K6144 case still differs by one INT8 code
(NRMSE 0.000407); all frozen MSD projections pass.

## Source identity

Installed source:
`/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/dynamic_quant/dynamic_quant_multi_row.h`.
Lines 215/222/224 implement `Div(127,amax)`, multiplication and RINT;
line 236 computes the independent returned scale.
SHA256: `fd0fb30d924b48ed4455cb505d2d530e822fa48d78b526db663d0bee69ffaefc`.
The dispatch file `dynamic_quant.cpp:103–105` maps key 10 to this kernel;
its SHA256 is `32fbfcb6d072572adb1ddf5810d76cd6be44a56307ec7beb4c10608f3e13cd75`.
The current device tiling key was not captured.

## Full-model result

The corrected [HF short reference](hf-full-short-msd-a8-20261016.json)
passes 1,273 reset/control checks and generates `[198,17351]`. Recomparison
of the saved native run has [438/2,934 failed checks](recompare-short-msd-a8-20261016.json),
including the second greedy token. Step 0's first two decoder outputs now
match exactly. Full numerical acceptance remains open.

The next diagnostic evaluates each HF layer at the same native hidden,
historical KV and shared indices using
[check_saved_layers.py](../../../tests/glm52/check_saved_layers.py).
This measures local arithmetic error separately from independent-rollout
error; the 8K numerical gate still precedes profiling.
