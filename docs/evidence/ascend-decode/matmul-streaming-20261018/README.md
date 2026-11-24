# The vendor matmul is already at the ceiling, and my first kernel is not

`a3-84`, one idle die (device 5), CANN 9.2.0, `[4, 2048] × [2048, 6144]`,
twelve independent weight banks so a rotation reads memory rather than L2.
The die's measured streaming maximum is 1190.77 GB/s
([HBM](../../a3-hbm-20261017.json)).

| path | weight bytes | µs | GB/s | share of ceiling |
| --- | --- | --- | --- | --- |
| `torch.matmul`, bfloat16 | 25.17 MiB | 22.52 | 1117.3 | 93.8% |
| `npu_quant_matmul`, int8 ND weight | 12.58 MiB | 23.92 | 526.0 | 44.2% |
| this project's TileLang kernel, bfloat16 | 25.17 MiB | 127.18 | 197.9 | 16.6% |

## Two claims this retracts

The bfloat16 matmul reaches 93.8% of the ceiling, so "a batch-one matmul
reaches a quarter of peak" was wrong as a general statement. It was a property
of the quantised call, not of the shape.

And the quantised call's deficit was an artefact of the benchmark. Half the
bytes took the same time, which looked like a two-fold read amplification from
an ND weight — a precise prediction, and the wrong target. Production does not
present ND: `AscendW8A8DynamicLinearMethod.act_quant_type` defaults to
`torch.int8` and `process_weights_after_loading` then calls `maybe_trans_nz`
(`vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py`), so the model's
weights are already FRACTAL_NZ. There is no win here, and the earlier
attribution of the round's 22.80 ms of cube time to inefficient matmuls rested
on this same ND measurement and does not survive it.

## A chain of separate matmuls costs nothing extra

[The chain sweep](../../../../tests/glm52/bench_matmul_chain_bandwidth.py) ran
1, 2, 4, 8, 16 and 32 back-to-back quantised matmuls over rotating banks. Per
call the cost is flat: 25.25 µs at one, 25.78 at thirty-two for the
`[2048, 6144]` shape, and 21.86 against 21.77 for a second instance of it.
Dispatch is already hidden behind the previous kernel's execution, so the
mover does not idle between kernels.

That removes one argument for fusing kernels — the per-kernel overhead a
megakernel would eliminate is not being paid. It leaves the other arguments
untouched: a fused kernel can keep activations on chip across the round's ~637
cube-vector handoffs, can carry a collective in its epilogue when no vendor
operator will, and can choose its own tiling.

## What the hand-written kernel showed

[The kernel](../../../../tests/glm52/bench_tilelang_gemv.py) is bit-exact
against `torch.matmul` on its first run, maximum absolute difference 0, and
0.18× its speed. Twelve tilings — `block_outer` 64 and 128, `block_inner` 128,
256 and 512, two and three pipeline stages — all land between 190.1 and
197.9 GB/s, so the parameters the DSL exposes do not move it.

The cause is the layout, not the DSL. `T.copy(weight[core * block_outer,
step * block_inner], weight_l1)` takes a tile from a row-major
`[outer, inner]` tensor, so each contiguous run is `block_inner` elements —
256 bytes at 128, 1024 at 512 — while the vendor reads FRACTAL_NZ, whose runs
are long. FRACTAL_NZ exists for this reason. A hand-written kernel has to
choose that layout itself, in whichever language it is written.

`block_outer` above 128 does not compile: `nTile * 128U * sizeof(__bf16) <=
kL0Budget` fails as a static assertion reading `gemm_v0: the (kL0Size x nTile)
B tile does not fit its L0B ping-pong slot`, which is the abstraction carrying
a hardware budget to compile time.
