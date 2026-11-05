# Which stream owns the round

A capture tells you what every kernel cost. It does not tell you what the
round would lose if a kernel vanished, because a kernel that overlaps another
stream costs the round nothing. This separates the two: for each stream, the
time it is the only one running.

Input: `profiles/width8-20261019/rank-0`, `kernel_details.csv` md5
`acde4cf368e464b7cb54580c7bff0857`, 9803 kernels. Tool
`tests/glm52/critical_path.py`. A round is one backbone pass — every 78th
attention call — and the first is dropped for its compilation and first-touch
costs, leaving two complete rounds. Earlier working figures divided the
capture into four equal parts and read 22% low; 78 layers and 75 MoE layers
are the normalisation, from `plan.md`.

Profiling inflates the round from the unprofiled 53.05 ms to 62.4 ms, so the
shares below are the result and the absolutes are not. That inflation is also
what the 11.48 ms of "no kernel anywhere" mostly is: 62.4 − 53.05 = 9.4 ms of
it, leaving around 2 ms of real gap. It is not an 11 ms target, and the
pinned-staging work already took 3.4–3.8 ms of whatever gap there was.

| per round, ms | round 2 | round 3 |
| --- | --- | --- |
| span | 63.003 | 61.871 |
| stream 81, backbone compute, alone | 35.632 | 35.588 |
| stream 80, TP reduction, alone | 5.451 | 5.248 |
| stream 83, draft chain, alone | 3.539 | 3.549 |
| stream 79 alone | 0.980 | 0.978 |
| stream 19, indexer, alone | 0.648 | 0.643 |
| streams 13, 78, 82 and N/A, every HCCL collective, alone | 0.000 | 0.000 |
| no kernel on any stream | 11.608 | 11.347 |
| FP32 cast bracket around the 146 reductions | 1.991 | 1.988 |
| the reduction between those casts | 5.644 | 5.418 |
| AI_VECTOR_CORE, against the bytes it moves | 8.733 / 0.345 | 8.693 / 0.347 |

## The decomposition's own floor eats the budget

No call costs less than the structure its core kind carries, and
[kernel-floor-20261019](../kernel-floor-20261019/README.md) measured those:
0.70 µs on AI_VECTOR_CORE, 2.30 on AI_CORE, 12.04 on MIX_AIV, 13.58 on
MIX_AIC. Apply them to this round's call counts and the decomposition has a
floor of its own, before any kernel computes anything.

| core | calls | floor ms | spent ms | floor is |
| --- | ---: | ---: | ---: | ---: |
| AI_VECTOR_CORE | 1893 | 1.325 | 9.071 | 15% |
| MIX_AIC | 505 | 6.858 | 26.190 | 26% |
| AI_CORE | 306 | 0.704 | 4.197 | 17% |
| MIX_AIV | 218 | 2.625 | 6.819 | 38% |
| COMMUNICATION | 86 | — | 10.290 | — |
| all | 3008 | 11.512 | 56.567 | 20% |

2 ms a token at A=6.959 allows a 13.92 ms round. This decomposition's launch
floor is 11.512 ms, 83% of that budget, and the second round reads 11.503. So
the target is unreachable by making kernels faster: it requires launching
fewer of them. 82% of the floor is on the MIX cores, 505 MIX_AIC calls a
round, 6.5 a layer — `GroupedMatmul`, `GroupedMatmulSwigluQuantV2`,
`MlaPrologV3`, `SparseFlashAttention` and about 3.3 `QuantBatchMatmulV3`.

The same capture carries the control. `MlaPrologV3` is one fused operator
covering several projections, the RoPE and the cache write, and it runs at
2.950 ms against 2.745 ms of declared bytes — 1.1x, essentially at the
bandwidth bound. Every unfused neighbour runs at 3x to 91x. Fusion is not a
launch-count argument here; it is the difference between 1.1x and 5x.

| family | ms | byte floor ms | x |
| --- | ---: | ---: | ---: |
| SparseFlashAttention | 6.973 | 1.382 | 5.0 |
| GroupedMatmulSwigluQuantV2 | 6.793 | 2.272 | 3.0 |
| GroupedMatmul | 5.596 | 1.212 | 4.6 |
| QuantBatchMatmulV3 | 3.810 | 0.780 | 4.9 |
| MlaPrologV3 | 2.950 | 2.745 | 1.1 |
| Cast | 2.300 | 0.076 | 30.3 |
| ForeachAddList | 1.622 | 0.018 | 91.1 |
| MoeInitRoutingV3 | 1.368 | 0.030 | 45.8 |

The two grouped matmuls are 12.388 ms against 3.484 ms of W4 weights, 4.023
GB a round at the teacher's 44.35-expert union. A grouped matmul declares all
256 experts whatever it reads, so that floor is the declared bytes scaled by
the union; `DT_INT4` had to be added to the byte table to see it at all.

## The TP reduction is wholly serial

Every HCCL stream runs 0.000 ms alone: those collectives hide completely. The
TP reduction does not — stream 80 runs alone for 5.35 ms a round, 8.6% of it,
and not one microsecond overlaps compute. At batch 1 there is no independent
work for it to hide under, which is why.

`probe.md` records the complete one-shot push as "no decoder gain claimed".
That was discipline about an unmeasured end-to-end, and it is read easily as
evidence that the site is cheap. It is not: with the cast bracket the site is
7.34 ms a round, 11.8% of it, all of it on the critical path.

## The forced FP32 costs 1.99 ms a round in launches

`cann_model._tp_reduce` widens BF16 to FP32, reduces, and narrows the result.
146 sites a round, two `Cast` launches each, 13.6 µs a site to convert 98 KiB
— 26% of the bracketed region, against 0.25 µs of actual data movement.

The FP32 is a project invariant, not a vendor constraint: `_tp_reduce` sets
`tp_reduction = "float32"` by choice, and `onesided_reduce.install` says it
"inherits the bfloat16 round trip". Whether the vendor collective associates
identically given BF16 input is untested, so on that path the casts stay.

On the one-sided reducer they need not. It accumulates locally in rank order,
which is what makes it bit-reproducible; widening each term on read and
rounding once on write is that same arithmetic, term for term and in the same
order. So folding both conversions into the kernel is bit-exact by
construction, removes 292 launches a round, and halves the bytes on the wire.

## The one-sided headroom is smaller than recorded

`onesided_reduce`'s docstring prices the vendor collective at 63.5 µs a site
and the one-sided push at 22.3 µs, giving 9.97 ms against 3.50 ms over 157
reductions. The decode path's measured cost is 5.53 ms over 146 sites, 37.9 µs
a site. Headroom against 22.3 µs is 2.28 ms a round, not 6.47. The protocol
is still worth taking; the figure it was argued with is not the one the decode
path pays.

## Vector work runs at 25x the bytes it moves

8.71 ms a round of AI_VECTOR_CORE execution moves 0.35 ms of data, 14% of the
round. A further 0.34 ms sits in index-driven gathers, which read selected
rows rather than their whole source, so those are reported apart rather than
excused by name.

The residual combine is the worst of it. Per layer: `Muls` 8.5 µs,
`ForeachAddList` 21.4, and the cast pair 13.5 — 43.4 µs to compute
`residual + scale * out`, feeding an `AddRmsNormBias` that costs 4.1. Ten
times the normalisation it serves, for 98 KiB.

`ForeachAddList` is settled by a matched pair inside this one capture, not by
a model of it. At `1,6144;1,6144` on AI_VECTOR_CORE, plain `Add` is 1.820 µs
and `ForeachAddList` 21.282 — 11.7x for the same arithmetic. Its cost is also
flat in rows, 21.282 at one against 21.442 at eight, so all of it is
structure. `fused_moe.py:236` reaches it through
`torch._foreach_add([shared_output], [fused_output])[0]`: a multi-tensor
kernel whose launch count does not grow with the list, called with a list of
one. The override's own docstring retires it once a superkernel add exists,
and `enable_super_kernel` is on. `--plain-moe-combine` substitutes the plain
add; both are an elementwise BF16 add, so it is bit-exact. 75 MoE layers,
about 1.4 ms a round.

Confirmed on device, one shared die, no model loaded, seconds to run — the
capture is not the only witness. `bench_kernel_floor.py` samples both
spellings in every batch, so they are paired by construction
([record](combine-floor-die1.json)):

| median µs | 1 row | 8 rows | 32 rows |
| --- | --- | --- | --- |
| plain add | 9.176 | 9.520 | 9.681 |
| multi-tensor add | 38.704 | 38.490 | 38.318 |

Flat in rows on both sides, and `torch.equal` holds on the result. This
instrument is wall time a call with one synchronize a batch, so its absolutes
carry host issue — `empty_like` alone costs 4.680 µs — and its 28.97 µs
difference is an upper bound. The capture's device-side difference, 19.46 µs,
is the one to price: 1.46 ms a round over 75 layers.

The combine's other half splits the same way. A second run on the same die
([record](combine-floor-die1-axpy.json)) is slower throughout — `add_8_rows`
11.779 against the first run's 9.520, the die having grown busier — which is
why only pairs inside one run count. In that run's units: `Muls` then add,
spelled `(x * s) + y`, costs 27.505 µs, while the single `torch.add(y, x,
alpha=s)` costs 14.453 against a plain add's 11.779. So one kernel does the
whole scale-and-add for about what an add costs.

That gives two steps rather than one, and only the first is free of numerics:

| step | per layer, device side | a round | exact? |
| --- | --- | --- | --- |
| now: `Muls` 8.46 + `ForeachAddList` 21.403 | 29.86 µs | — | — |
| plain add, keeping `Muls` | 10.36 µs | 1.46 ms | bit-exact |
| one `add(..., alpha=scale)` | about 3.5 µs | 1.98 ms | more accurate |

`alpha` keeps the product in the accumulator instead of rounding it to BF16
first, so the second step changes values and needs the frozen teacher. Take
the bit-exact step first; the second is worth about 0.5 ms a round beyond it.

This is the megakernel's target stated in numbers, and it is not the boundary
the closed brick tested. That brick fused publication with its consumer and
measured 162.141 against 162.781 µs — a communication boundary, with the
vector launches left in place on both arms.

## The shared expert cannot join the routed bank

`QuantBatchMatmulV3` is the largest recoverable floor in the round: 254 calls,
3.810 ms, of which 2.603 is floor — 68% structure, and its down projection is
15.311 µs against a 13.58 µs MIX_AIC floor, 89%. Three of those calls a layer
are the shared expert's own dense path, `6144 → 256`, swiglu to 128, then
`128 → 6144`, followed by the scale-and-add. 64.0 µs a layer, 4.8 ms a round.

The shared expert is dimensionally one more expert: `n_shared_experts = 1` at
`moe_intermediate_size = 2048`, the same 128 columns a rank that each routed
expert carries. So appending it to the bank as group 256 looked free.

It is not. `check_shared_expert_layout.py` reads the tensor headers, which
settle it in seconds without a device:

| layer 4 | shared | routed expert 0 |
| --- | --- | --- |
| `gate_proj.weight` | (2048, 6144) I8 | (1024, 6144) I8 |
| `down_proj.weight` | (6144, 2048) I8 | (3072, 2048) I8 |
| `*.weight_scale` | (2048, 1) F32 | (2048, 1) F32 |
| `gate_proj.scale_bias` | absent | (2048, 1) F32 |
| `down_proj.scale_bias` | absent | (6144, 16) F32 |

The routed weight is half the rows because it is W4 packed two to a byte; the
shared weight is W8. And the shared expert has no `scale_bias`, which is the
W4A8 MSD correction. Joining the bank would mean requantising W8 to W4 — a
different model, not a different schedule. Route closed at the layout, before
any kernel work.

## The megakernel's target, in calls a layer

Acceptance is at its ceiling: the copy replay gives 147 rounds at width 32 and
144 at width 64 through 256, so A saturates near 7.1. With A fixed, 2 ms a
token allows a 13.92 ms round against an 11.512 ms launch floor.

That sets the engineering target in the only unit that matters here. A layer
issues 6.5 MIX_AIC calls; the floor budget for a 13.92 ms round with room left
for arithmetic is one to two. The per-layer sequence to collapse:

| layer stage | calls now | core |
| --- | --- | --- |
| MlaPrologV3 | 1 | MIX_AIC |
| SparseFlashAttention | 1 | MIX_AIC |
| O projection, `QuantBatchMatmulV3` | 1 | MIX_AIC |
| router `MatMulV3`, `MoeGatingTopK`, `MoeInitRoutingV3` | 3 | AI_CORE, AIV, MIX_AIV |
| `GroupedMatmulSwigluQuantV2` | 1 | MIX_AIC |
| `GroupedMatmul` | 1 | MIX_AIC |
| shared expert, two `QuantBatchMatmulV3` plus dequant | 3 | AI_CORE, MIX_AIC |
| combine and norm, `Muls`/`ForeachAddList`/`Cast`/`AddRmsNormBias` | 6 | AIV |

The two grouped matmuls are the first brick: the intermediate between them is
64 rows by 128 columns of INT8 plus its scales, 8 KiB, which never needs to
leave on-chip storage. Fusing them removes one MIX_AIC call a layer, 13.58 µs
of floor, 1.02 ms a round, and the HBM round trip with it.
