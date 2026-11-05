# Token-owned FFN outputs

This candidate changes the consumer of routed down-projection results. Each AIV owns one token and 1536 adjacent hidden columns, reconstructs that token's eight routes, and completes the weighted sum and FFN epilogue in UB. The source parent is uninstrumented composed build-v2, library SHA256 `3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655`. Active-expert enumeration and descriptor candidates are not included.

The public Torch schema, C ABI, 12 bridge buffers and 20 body buffers retain their parent contracts. The new namespace is `tilert_ffn_token_output`; the library is `ffn_token_output.so`. All original 30 checked boundaries remain materialized, including route BF16, routed BF16, scaled routed BF16, shared BF16 and combined BF16. The change removes reads of route, routed and scaled tensors by their next consumers. At M12 these logical reads total 1,474,560 bytes; this is not a measurement of HBM traffic or latency.

## Ownership and numeric boundaries

Each token has four column tiles. Task `4*token + tile` belongs to AIV `task % (2*blocks)`. M12 with 24 groups gives 48 tasks and 48 AIVs. M1 through M32 with 16 or 24 groups use the same mapping. Every original route and output column has one writer. Idle AIVs still reach every original mixed barrier.

An inverse maps `source_row*8 + source_slot` to the expert-major route index. For expert prefix `s`, count `c` and route index `q`, the original compact high and low rows remain `2*s + (q-s)` and `2*s + c + (q-s)`. The original top-k IDs select the expert's bias and weight scale. Expert-local order may change without changing original-slot accumulation order. Valid routing requires eight distinct, in-range experts per token, as in the parent contract.

The original `LoadScale`, `ScaledPartial` and `Recombine` statements are unchanged: truncated channel scale, independent FP16 high/low partials, FP32 high-times-16 plus low plus correction, then route activation scale. Each route rounds to BF16 before probability multiplication. Probabilities retain the parent's BF16-to-FP32 boundary. Slot zero directly seeds the FP32 accumulator when its probability is nonzero; later slots add in order. A zero probability skips only its weighted contribution, including a NaN route. It still produces the route diagnostic. This preserves negative zero and zero-probability behavior.

The accumulator rounds to BF16 routed output. Multiplication by 2.5 rounds again to BF16 before adding the independently produced BF16 shared output and rounding the final output to BF16.

## Publication and storage

All six body barriers and both outer `EndPhase` calls remain. The body prefix through barrier 4 is unchanged. Before barrier 4, routed down has ended its matrix operation and published, and shared `Store` has completed its MTE3-to-MTE2 fence and published. Barrier 4 joins those producers before token owners read either output. Barrier 5 remains after token owners publish all outputs; the old combine and epilogue arithmetic are replaced by the UB consumers.

The inverse is scalar-written and scalar-read. `DataSyncBarrier<MemDsbT::UB>()` separates those accesses. The local expert IDs and activation scales use MTE2-to-S before scalar reads. Each route BF16 store uses V-to-MTE3 and MTE3-to-V before the same UB is consumed or overwritten. Loading shared BF16 into that buffer also waits for V-to-MTE2 and MTE3-to-MTE2; final output stores complete before the next task reuses it.

The original routed `held.other` buffer holds the accumulator; `Recombine` uses `value`, `work`, `scale`, `staging` and `half_value`. One additional 3072-byte UB buffer contains 256 inverse indices, 256 expert IDs and 256 FP32 activation scales. Peak resident UB is 115,424 bytes: routed 81,952, shared 6592, shared output dequantization 23,808 and metadata 3072. This includes the shared `Store` allocations after its local objects have left scope; no allocator reclamation is assumed. No GM allocation is added. The original system-workspace calls remain; CANN 9.2 selects the first non-null workspace and does not switch it on the later ordinary `SetSysWorkspace` call.

## Validation

`test_contract.py` checks all M1..32 ownership maps at 16/24 groups, compact high/low row coverage, arbitrary expert-local permutations, negative zero, zero-probability NaNs, required BF16 boundaries and frozen-prefix preservation. CPU arithmetic checks establish the proposed boundary order; device byte comparisons remain the acceptance gate.

Build into a new directory:

```bash
python3 build.py /workspace/tests/glm52/whole_layer/ffn_composed/build-v2 \
  /workspace/tests/glm52/whole_layer/ffn_token_output/build-v1
```

The generated checker changes only the candidate Torch namespace in the parent's frozen dual-oracle checker. The first device gate must compare all 30 boundaries under changing inputs at M1/2/12/32 and 16/24 groups, including zero input and zero probabilities. Performance requires a separate same-address paired experiment with at least 30 event samples per arm, preserving owning Torch validators and fresh-write checks.
