# The expert GMM re-measured outside L2, and what the fused combine permits

`a3-84` die 0 (shared with a co-tenant's idle job, AICore 0% throughout), one
process, W4A8 expert shapes at TP16: `gate_up` `[E, 6144, 256]`, `down`
`[E, 128, 6144]`. Per point six processes of twenty batches × twenty calls
after twenty warmups; the number is the median of process medians, because
sessions carry a ±4 µs additive jitter that twenty in-process batches never
show (in-process spread ~1%). Differences below ~5 µs are not resolvable.
Raw reports: `/workspace/experiments/gmm-fix-20261018/` (239 JSON),
benchmark revisions in [bench_gmm_shape_scaling.py](../../../../tests/glm52/bench_gmm_shape_scaling.py).

## Rotation sized by reuse distance, calibrated twice

One bank of `groups × group_bytes` is re-read after `banks − 1` other banks,
so the rotation defeats L2 only when that distance passes a multiple of the
192 MiB. A sweep at 27 groups over 1/2/5/10/20/30/40 banks shows the cache
dividend gone by ~0.4 × L2; at 256 groups the ceiling test binds: 2 × L2
still yields a marginal 1256.7 GB/s, above the die's measured 1190.77
maximum, while 4 × L2 falls under it. `rotation_banks` therefore sizes the
rotation to 4 × L2 (39/33/17/9/5 banks for `gate_up` at 27/32/64/128/256
groups; 77/65/33/17/9 for `down`, whose per-group bytes are half).

| groups | `gate_up` µs | GB/s | `down` µs | GB/s |
| --- | --- | --- | --- | --- |
| 27 | 80.250 | 264.6 | 79.614 | 133.4 |
| 32 | 78.294 | 321.4 | 80.937 | 155.5 |
| 64 | 107.263 | 469.2 | 98.157 | 256.4 |
| 128 | 151.983 | 662.3 | 135.424 | 371.7 |
| 256 | 238.340 | 844.7 | 215.568 | 467.0 |

Marginal rates (`gate_up` 32→64/64→128/128→256: 868.7/1125.5/1165.7 GB/s;
`down`: 730.7/675.3/628.0) all sit under the ceiling. Rows 1 against 4 differ
by less than the jitter at the model's shapes, so the activation side is
negligible. Against the L2-contaminated table the clean 128/256 points
reproduce within 1% and the suspect 32-group points were only 5–6%
optimistic. The `down` GB/s column omits scale and bias bytes (+18.7% there,
+0.4% for `gate_up`), kept for comparability with the old table.

The model's measured 27.121-expert union, now outside L2: **80.250 µs
(`gate_up`) and 79.614 µs (`down`) per layer-call, of which the streaming
part is ~18 µs** (asymptotic slope 0.68 µs/group). The fixed cost is 62–63 µs
per call — three quarters of the model-shape call is prologue, not bytes.

## The fused combine exists, is faster, and is barred from the side that needs it

`npu_grouped_matmul_finalize_routing`'s true contract, from
`aic-ascend910_93-ops-info-transformer.json`: output dtype float32 only
(the probe's bf16 request was the "dtype should be float" refusal), `scale`
int64 `[E, 1, n]` exactly, `bias` mandatory on A2/A3 and two-dimensional,
`row_index` int64. On `gate_up` at 27 groups it runs at 74.229 µs — faster
than the plain grouped matmul (80.250) by one dispatch and a `tokens`-row
output. But `down` is refused outright: `In W4A8 mode ... k should bigger
than 128`, and TP16 gives `k = LOCAL_INTERMEDIATE = 128` exactly. The side
whose output needs the combine cannot use the operator; the side that can
use it does not need the combine.

`npu_moe_token_unpermute`'s contract (the production combine): the inverse
permutation must be int32 — int64 is refused with 161002 — and `probs` is
`[tokens, 8]`, matching `expert_bank.py:219`. The pair costs +2.9 to +5.8 µs
over the matmul alone at the model shapes.

## What this changes upstream

The round's expert arithmetic at 75 MoE layers is ~160.5 µs × 75 ≈ 12.0 ms of
which ~9.4 ms is per-call fixed cost and only ~2.7 ms is weight streaming.
Together with the vendor matmul already at 93.8% of the ceiling, the cube
block's remainder is prologue and launch shape, not bandwidth — which is the
case for fusing calls, not for a faster matmul. A `k = 128` lower bound also
marks the W4A8 fused path as hostile to TP16 shapes; an EP layout
(`intermediate` stays 2048) clears it.
