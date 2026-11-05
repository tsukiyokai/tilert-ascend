# The MoE epilogue, hand-written, against the vendor's three operators

The first megakernel brick, and it is vector-only on purpose: the three
operators it replaces are all `AI_VECTOR_CORE`, which reaches 0.72 µs on this
die where a MIX kernel cannot go below 13.56.

After the routed combine each MoE layer runs `Muls` to scale the routed
output, `ForeachAddList` to add the shared expert's, and `Cast` to widen the
sum to FP32 for the TP reduction. In the width-8 capture those cost 8.641,
21.403 and 6.860 µs — 36.90 µs a layer to move 393216 bytes, whose own
bandwidth cost at the die's measured 1.155 TB/s is 0.34 µs.

`tests/glm52/epilogue/` is a standalone AscendC program: both arms in one
process on the same inputs, alternating, built with `bisheng` for `dav-c220`,
run on one shared die with no model. Raw output in [die1.txt](die1.txt).

## It is bit-exact, and that took knowing the chip

`dav-c220`'s `Muls` rejects BF16 — half, float, int16 and int32 only. So the
fused kernel computes in FP32 and uses `Cast` to put the two BF16 roundings
back exactly where the three operators have them: once after the product,
because `Muls` emits BF16, and once after the sum, because the add does.
BF16 to FP32 is exact, so the final widening adds nothing. A kernel that kept
the product in FP32 would be more accurate and not comparable.

At 8 rows and at 32, against a host reference that rounds where the device
rounds, and against the three-kernel arm: 0 of 49152 and 0 of 196608 differ.

That the vendor's `Muls` has to widen too is most of why it costs 8.641 µs to
scale 98 KiB.

## Two instruments, because they answer different questions

One launch a call is what a stream of separate kernels pays from the host.
Many passes inside one launch amortize the issue away and leave the device
cost, which is what an in-graph replay pays.

| 8 rows, µs | fused | three kernels | ratio |
| --- | --- | --- | --- |
| issue-bound, one launch a call | 5.68 | 16.70 | 2.94 |
| device, 64 passes a launch | 0.517 | 1.222 | 2.37 |

| 32 rows, µs | fused | three kernels | ratio |
| --- | --- | --- | --- |
| issue-bound | 6.39 | 17.90 | 2.80 |
| device | 0.875 | 1.665 | 1.90 |

Six paired rounds at 8 rows, three at 32; the spread within a round is under
0.02 µs on the device arm.

The device figures are cache-resident and must not be read as bandwidth. 393
KiB re-read 64 times stays in L2, and the 32-row arm's apparent 1797 GB/s is
above the die's 1.155 TB/s, which proves it. The ratio between the arms
survives that, since both arms are equally hot.

## What the brick is worth, and what it is not

Fusing three hand-written kernels into one is worth 0.705 µs a layer on the
device and 11.0 µs from the host. Over 75 MoE layers that is 0.05 ms and
0.83 ms a round.

The larger term is not the fusion. The same arithmetic costs 36.90 µs as
three vendor operators and 1.222 µs as three hand-written ones — 30x — and
0.517 µs fused, 71x. Cache state does not explain it: a cold pass over 393216
bytes needs 0.34 µs, so 36.90 µs is 1% of the die either way. Over 75 layers
the hand-written epilogue is worth about 2.7 ms a round, of which the fusion
is 0.05 and the hand-writing is the rest.

So the brick's lesson for the megakernel is a priority, not an encouragement:
on this path the per-operator excess is a bigger multiplier than the operator
count. Both compound, and the count is what a megakernel removes that nothing
else can, but a fused kernel built out of vendor-quality stages would keep
most of the loss.

## Next

The epilogue's other half — `Cast`, `AddRmsNormBias`, `DynamicQuant` after the
reduction, 14.6 µs a layer — is the same shape of work and the same kernel can
take it. Beyond that the brick needs the MIX cores and therefore cube tiling,
a cross-core flag whose direction from AIV to AIC is not settled by the
headers, and a cross-block barrier for the SwiGLU's gate/up pairing and the
dynamic quantisation's row maximum.

One finding for the closed route-3 brick: `tests/glm52/shmem/fused_main.cpp`
allocates its 4096-byte tiling buffer and never writes it, so the matmul in
`fused_proj_reduce_kernel` ran on uninitialised `TCubeTiling`. The 162.141
against 162.781 µs comparison that closed the projection-reduction route
should not be relied on.
