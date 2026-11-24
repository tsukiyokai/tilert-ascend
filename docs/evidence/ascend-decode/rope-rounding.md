# CANN rotary rounding

`npu_interleave_rope` accepts BF16 values and trigonometric coefficients,
converts both to FP32, performs separate products and a sum, then casts once
to BF16. Main MLA stores real and imaginary halves; the indexer restores
alternating coordinates before its BF16 Hadamard transform.

The installed CANN 9.2.0 source is
`ops_transformer/ascendc/interleave_rope/op_kernel/interleave_rope_b11d.h`:
lines 99–105 convert coefficients, 127–134 extract and convert coordinates,
139–149 perform FP32 products and addition, and 151–152 cast the output.
Its SHA256 is
`96028ea9c3f2b5426b60c00e03951fafbcd4903ceff5880b4ad31482d37e0a8a`.

HF 5.14.1's `apply_rotary_pos_emb_interleave` rounds BF16 products before
addition. For position one's first coefficient pair, BF16 `c=0.5390625`,
`s=0.83984375` and `even=odd=1.0078125` produce `-0.3046875` with stepwise
BF16 arithmetic and `-0.302734375` with FP32 intermediates. Position zero's
`c=1,s=0` does not expose this difference.

## Frozen checkpoint inputs

[The differential](saved-rope-rounding-20261016.json) uses the same native
layer inputs, CPU projection, LayerNorm, Hadamard and A8 encoder. Only rotary
intermediate precision changes. The BF16 variant exactly reproduces all
saved HF keys and scales, binding the experiment to the original mismatch.
The 42 native rows come from two positions and 21 full indexer layers.

| Position and storage | Rows | Stepwise BF16 unequal values | FP32 rotary unequal values |
| --- | --- | --- | --- |
| 0, BF16 | 5 | 0 | 0 |
| 0, C8 | 16 | 0 codes, 0 scales | 0 codes, 0 scales |
| 1, BF16 | 5 | 20 | 0 |
| 1, C8 | 16 | 45 codes, 1 scale | 0 codes, 0 scales |

Run [the CPU differential](../../../tests/glm52/check_saved_rope.py) on the
server with `--checkpoint`, `--native-run`, `--local-run` pointing to the
saved same-input HF tensors, and `--output`. The report binds both traces
and the installed CANN source by SHA256. It does not execute an NPU kernel
or measure performance.

## Reference adapter

[The HF reference](../../../tilert/models/glm_5_2_ascend/reference.py)
retains HF frequency generation and its BF16 coefficient cast. The indexer
uses FP32 rotary intermediates with alternating output coordinates. Each
reference attention instance uses the same HF forward bytecode with only
its rotary function lookup replaced; Transformers files, module globals,
other instances, projections, cache and attention logic remain unchanged.
The report's `weight_arithmetic.rotary` records this numerical convention.
The MTP reference installs the same helper on its decoder instance.
[Independent CPU checks](check-rope-adapter-cpu-20261016-mtp.json) pass
61/61: 14,464 outputs match scalar FP64 arithmetic with independent BF16
rounding across positions 0/1/8192 and M1/M4. Real-checkpoint MTP construction
checks instance binding, a separate metadata copy, BF16 index storage and
reset. This check does not execute the MTP forward or an NPU operator.

[Same-input layer checks](same-input-layers-rope-20261016.json) have
38/1,572 failures, compared with
[52/1,572 with stepwise BF16 rotation](same-input-layers-20261016.json).
All cache fields pass; route and associated hidden failures decrease from
seven to five. The remaining 28 layer-logit failures include the final norm
and head applied to local outputs; they are not 28 independent decoder
failures. Each layer still receives the native input hidden state, KV and
shared indices, so this comparison excludes cross-layer propagation.

Complete decode acceptance still compares independent trajectories at 8K
for 32 steps, with the existing hidden/logit tolerances, greedy IDs, routes,
indices, effective KV and reset checks. Exact frozen indexer rows establish
the rotary/encoding result for these inputs, not whole-model acceptance.
