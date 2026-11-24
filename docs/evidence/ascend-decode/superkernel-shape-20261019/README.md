# SuperKernel already slices by engine homogeneity

Read from `profile-rope-20261018`'s `trace_view.json` and `op_statistic.csv`,
rank 0, with `--super-kernel` on.

## It is already the topology I was about to propose

55 distinct SuperKernel group shapes, each named `start_<operator>_end_<operator>`.
The largest class, 471 instances, is
`full_model_start_aiv_all_reduce_float_end_aiv_all_reduce_float` — start and
end both on AIV. The rest are runs of consecutive same-engine operators:
`start_static_kernel_AddRmsNormBias … end_static_kernel_Grouped…`,
`start_…MoeTokenUnpermute …`, `start_…ForeachAddList … end_…ForeachAddList`.

So "slice a layer into a few engine-homogeneous segments" is not a design to
adopt; it is what the configuration already does, and the skeleton document
proposed it without checking.

## But it merges dispatch, not the data path

| row | calls | total µs |
| --- | --- | --- |
| `SuperKernel` / MIX_AIC | 1053 | 68386.994 |
| `SuperKernel` / MIX_AIV | 858 | 35199.677 |

and **coexisting with them as standalone operator entries**: `Cast` 1669,
`Muls` 235, `Abs` 235, `MatMulV2` 189, `Slice` 132, `LayerNormV3` 73. If the
grouping absorbed those, they would not be listed separately.

The group names say the same thing from the other side: a group running from
`AddRmsNormBias` to a `GroupedMatmul` is a sequence of complete kernels packed
into one dispatch, not one kernel body. And a cross-engine handoff inside such
a group cannot stay on chip on this part, because `L0C → UB` does not exist in
this generation.
[the data path](../cube-vector-path-20261019/README.md)

Two measured figures agree with that reading: the in-graph dispatch floor is
1.2 µs while a single operator's prologue is 8.3 to 16.4. SuperKernel removes
the former times the group's length and leaves the latter times the group's
length.

## So the proposal has to be restated

Not "slice by engine into two kernels per layer" — that exists. The thing that
does not exist is **keeping a homogeneous run's intermediates in UB**, which
means compiling N kernels into one kernel body rather than into one dispatch.
SuperKernel cannot do that: it is a dispatch-level merge that does not change
operator implementations.

The AMD write-up's corresponding mechanisms are prefetch across the kernel
boundary and streaming synchronisation, and what lets them carry it to a whole
layer is `L0C2UB`. Without that path, what we can merge is the inside of one
engine's run: on AIV the norm, quantise, cast, epilogue and reduction, with
nothing landing in GM between them.

## What is not established

The 1.5–2.9 ms of elementwise intercept was measured per call in isolation on
one die and multiplied by the round's family counts. It does **not** show that
those intercepts survive inside a SuperKernel group — the census that produced
it said plainly that a per-operator method cannot reach inside those spans.

`Cast`'s 1669 standalone entries are direct evidence that this family is
outside the groups, so its intercept very likely stands. For the operators
that are inside — `AddRmsNormBias`, `DynamicQuant`, `MoeTokenUnpermute` — it is
unknown.

One comparison settles it: the same prefix twice, changing only
`--super-kernel`, comparing per-family device time and entry counts rather
than TPOT. The ablation already gives the TPOT difference, 1.807 ms, but not
the family decomposition — and the decomposition is what says how much is left
for a real in-kernel fusion to win.
