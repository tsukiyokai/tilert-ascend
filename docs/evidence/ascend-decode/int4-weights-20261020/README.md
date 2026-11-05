# The INT4 weights the chip can read and the library will not pass

The routed-expert kernel streams INT8 weights and is
[weight-streaming bound at 69% of peak](../routed-kernel-20261020/README.md).
The model is W4A8. So the dominant term is twice the size it has to be.

## The vendor does read INT4, and the profile says so

From this project's own capture, `kernel_details.csv` on rank zero:

| op | input data types | input formats |
| --- | --- | --- |
| `GroupedMatmulSwigluQuantV2` | `INT8;FLOAT;INT64;DT_INT4;UINT64;FLOAT` | `ND;ND;ND;FRACTAL_NZ;ND;ND` |
| `GroupedMatmul` | `INT8;DT_INT4;FLOAT;UINT64;FLOAT;FLOAT16;FLOAT16;INT64;FLOAT` | `ND;FRACTAL_NZ;ND;NCL;ND;ND;ND;ND;ND` |

The weight is `DT_INT4` in `FRACTAL_NZ`, and the op's own name says it:
`aclnnGroupedMatmulSwigluQuantWeightNzV2`. So A3 reads 4-bit weights.

That reframes the comparison. Per rank per layer the routed experts are
sixteen experts of 2,359,296 elements: 18.9 MB at INT4, 37.7 MB at INT8.
The vendor's 165.2 µs moves 18.9 MB, which is 114 GB/s -- 10% of the die's
1.155 TB/s. This project's kernel moves 37.7 MB in 49.3 µs, which is
765 GB/s, 66%. So it is 3.3 times faster while carrying twice the bytes, and
its bandwidth efficiency is 6.7 times better. The vendor op is not
bandwidth-bound by any reading.

At 66% efficiency on 18.9 MB the kernel would take 24.8 µs a layer. That is
the prize: half of the biggest lever this project has.

## Where the high-level library stops

`MatmulType<TPosition::GM, CubeFormat::*, int4b_t, *>` fails to compile in
every combination tried -- NZ and ND, transposed and not, under `CFG_NORM`,
`CFG_MDL` and `CFG_IBSHARE_NORM`. Seven variants, two failure sites:
`data_copy_wrapper_utils.h:312` for NZ and `data_copy_wrapper_nd.h:111` for
ND, both `no matching function for call to 'DataCopy'`.

The instantiation names the reason. The failing call is
`CopyNZ2NZImpl<integer_sub_type<4, true>, signed char, false>`, and every
frame carries `MatmulInputBType<MatmulType<...int4b_t...>, signed char>`: the
library's plan is that global memory holds INT4 while the L1 tile holds INT8,
so the copy would have to widen on the way in. A3's MTE2 does not widen, so
the overload does not exist. The surrounding code expects it to be possible
-- `if constexpr (IsSupportB4<TransT>()) { blockLen /= INT4_TWO; ... }` is
right there in the NZ path -- which makes this a gap in the abstraction
rather than a limit of the part.

## The chip's own path is open

Six compile probes, all accepted on `--npu-arch=dav-2201`:

| what | verdict |
| --- | --- |
| `DataCopy(LocalTensor<int4b_t> in UB, GlobalTensor<int4b_t>, count)` | accepted |
| the same with `DataCopyParams` -- the call the library failed on | accepted |
| the same bytes as INT8, half the count | accepted |
| `DataCopy` into `TBuf<TPosition::B1>` as INT4 | accepted |
| `DataCopy` into `TBuf<TPosition::A1>` as INT4 | accepted |
| `DataCopy` into `TBuf<TPosition::A2>` as INT4 | accepted |

So INT4 reaches L1 and L0 as INT4. The other half of the journey is already
written in the library this project cannot use: `load_to_l0_utils.h` returns
`LoadL0bInstrType::LOAD2DTRANSPOSE` for `int4b_t`, and
`load_to_l0b_loadInstr.h` has an `IsSameType<TransT, int4b_t>` branch.

What is blocked is the high-level `Matmul`, not the hardware. Reaching the
24.8 µs means managing L1 and L0 by hand -- `DataCopy` the INT4 tile to L1,
`LoadData` it to L0B, `Mmad` -- instead of `IterateAll`.

## How the part does W4A8 at all: two INT4 mads a weight tile

Driving L1 and L0 by hand got one error, and it was the useful kind.
`kernel_operator_mm_impl.h:391` wants `__cb__ signed char *` for the third
operand, and the assertion beside it lists what `Mmad` takes:

> dst: int32_t, fm: int8_t, filter: int8_t; dst: float, fm: half, filter:
> half; dst: float, fm: float, filter: float; dst: float, fm: bfloat16_t,
> filter: bfloat16_t; dst: int32_t, fm: int4b_t, filter: int4b_t

So INT4 is there, but only with *both* operands INT4, dispatching to
`mad_s4`. The intrinsics on this part are `mad`, `mad_s4`, `mad_sp` and
`mad_mx` -- nothing mixed. An INT8 activation against an INT4 weight is not
an instruction here.

And a vector expansion of the weights is not an answer either: 37.7 M weight
elements a rank-layer would have to pass through the vector units against a
49.3 µs budget, which is orders out.

The vendor's own source says what it does instead. In
`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul_swiglu_quant/op_kernel/`
there are three files named `grouped_matmul_swiglu_quant_a8w4_msd_pre.h`,
`_mid.h` and `_post.h`. The pre stage holds `xHighI4Tensor` and
`xLowI4Tensor` and masks with `Duplicate(xLowI16Tensor, 0x0F0F)`, commented
"get rid of high 4 bits in every int8"; its buffer accounting reads
`M * K * 7B (1B + 0.5B + 0.5B + 2B + 4B)`.

So the INT8 activation is split into two INT4 halves, each is multiplied by
the INT4 weight, and the products are recombined as 16 × high + low. Two
`mad_s4` calls a weight tile.

That is a good trade here and a bad one for the vendor. The split is over the
activation, which at batch-one decode is 12 KB an expert against 2.36 MB of
weights; the weight tile in L0B serves both mads, so L1 traffic is unchanged
and HBM traffic halves; and the cube is weight-streaming bound at 66% of
peak, not mad bound, so doubling the mads costs little. The vendor pays for
it by splitting pre, mid and post into three kernels with global memory
between them -- which is a plain reading of why its 165.2 µs is 10% of
bandwidth. One kernel keeps the split in UB.

## The primitive, driven by hand and measured

`DataCopy` global to L1, `LoadData` L1 to L0, `Mmad`, `Fixpipe` L0C to global.
M, K and N are 16, 64 and 16, one INT4 fractal of K a side, and both operands
are filled so that the answer does not depend on the fractal order.

| right-hand filling | expected | got |
| --- | --- | --- |
| `0x11`, every nibble a one | 64, which is K | 64 everywhere, nothing untouched |
| `0x21`, nibbles of one and two | 96 | 96 everywhere |

The second row is the one that settles it. If the part read one nibble a byte
the answer would be 64 or 128; 96 is 32 ones and 32 twos, so both halves of
every byte are separate INT4 elements. `mad_s4` consumes packed INT4
correctly.

One negative, now withdrawn as an explanation though the observation stands:
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` did fail here, with the queue
reporting `sqe_type=3(place holder), errType=0x20(sq sw status error),
sqSwStatus=0x23` and a 507015 return. But that is not the scheduler refusing
the type. The open-source runtime decodes `RT_DAVID_SQE_TYPE_PLACE_HOLDER = 3`
and `RT_STARS_CQE_ERR_TYPE_SW_STATUS = 1 << 5`, so the line is a placeholder
SQE carrying a software-status error -- and `KERNEL_TYPE_AIC_ONLY` is live on
A3 three independent ways, including being the global default in Huawei's own
`swiglu_gated_mlp` kernel, whose parameters are this very expert chain. The
documented requirement that fits is that a forced type needs a non-zero
workspace, and the cube-only type that comes with hard sync is
`KERNEL_TYPE_MIX_AIC_1_0` rather than `AIC_ONLY`. The `__mix__(1, 2)` shape
this project runs works, so nothing is blocked -- but "AIC-only is
unsubmittable" was the wrong conclusion and is retired.

So every link is proven: INT4 from global memory to L1, L1 to L0, the mad
itself, and the product out through the fixpipe. What remains is tiling
K = 6144 and N = 256 by hand, and the activation split the vendor's `a8w4_msd`
stages describe.

## The way through: a homogeneous INT4 pair, stacked along M

Hand-driving L1 and L0 got the primitive right and then stalled on the
micro-synchronisation of a reused L0 slot. With `PipeBarrier<PIPE_ALL>` the
same code answered a K of 128 exactly, 256 as 192, 512 as noise, 1024 as half
and 6144 exactly -- a hazard's signature, not arithmetic's. Replacing it with
`SetFlag`/`WaitFlag` on `MTE1_M` and `M_MTE1` made K of 512 and above fail at
launch instead. Along the way the layout did come out: the INT4 fractal is
`[k-fractal][16 columns][64 of K]`, a prefix of 64 nibbles lands entirely in
column zero and a prefix of 1024 spreads 64 across all sixteen; one mad covers
at most 128 of K; and eight fractals in one `LoadData` lose a mad's worth.

The way through is not to hand-roll at all. The library rejects INT4 because
it plans to widen it to INT8 on the way into L1 -- so give it a pair that
needs no widening. `MatmulType<..., int4b_t, ...>` for *both* operands
compiles in all four combinations tried: B as ND and as NZ, transposed and
not.

Two INT4 matmuls would read the weight twice and give back the saving, so the
two halves of the activation stack along M instead. With
`A' = [A_high ; A_low]` of 2 × ROWS by 6144 and one `Matmul`, the weight is
read once and

    C[r][n] = 16 × C'[r][n] + C'[ROWS + r][n]

M doubling from two rows to four costs almost nothing in a weight-bound
matmul, and the library's tiling -- the thing already reaching 66% of
bandwidth on INT8 -- is kept rather than reimplemented.

So the routed kernel's INT4 form is: the vector half splits the INT8
activation into two INT4 halves and stacks them, the cube runs one INT4
matmul an expert, and the vector half recombines with the shift before the
dequantisation it already does. The down projection takes the same treatment,
its activation being the intermediate the vector has just quantised.

## The INT4 kernel runs, and one defect is left

`routed_int4` builds and runs the whole chain -- vector split, INT4 matmul,
recombine, activate, requantise, split again, INT4 matmul, recombine, BF16 --
in one launch. Four experts on four blocks take 39.350 µs against the INT8
form's 47.108, which is the halved weight traffic showing through. It is not
correct yet: 35530 of 49152 values deviate.

The defect is isolated to one thing. An INT4 `DataCopy` to global memory lands
when its destination offset is zero and lands nowhere when it is not:

- The high halves, written at offset zero, are exact -- 98 → 6, 112 → 7,
  −73 → −5, 80 → 5, 13 → 0, −106 → −7, 24 → 1, which is `floor(x / 16)` every
  time.
- The low halves, written at nibble 12288, appear nowhere in the allocation. A
  search of the whole buffer for the expected sequence returns −1, and the
  bytes that should hold them read the same eight values across completely
  different input data -- they are a previous run's residue.
- Swapping which call goes first changes nothing, so it is not the second
  write that fails.
- Writing the known-good high half to the suspect offset also fails, so it is
  the offset and not the source.
- Three addressings fail alike: a `GlobalTensor<int4b_t>` subscript, a
  `__gm__ int4b_t*` offset, and that offset taken through a byte pointer.

## The write mechanism is exonerated, and the defect splits in two

The isolated probe was made to match the kernel exactly, one factor at a time,
and every combination lands all 2048 values:

| probe shape | result |
| --- | --- |
| plain vector kernel, offset 256 | 256 of 256 |
| with `ASCENDC_CUBE_ONLY` | 256 of 256 |
| as `__mix__(1, 2)`, vector half working | 256 of 256 |
| both, offsets 256 / 2048 / 6144 / 12288 / 24576 | 256 of 256 each |
| both, span 2048 at offsets 0 / 2048 / 12288 | 2048 of 2048 each |

So neither the macro, nor the MIX shape, nor the offset, nor the span is at
fault. The write mechanism is exonerated.

A dedicated witness buffer then separated the last two hypotheses. The low
half is emitted twice: once to `stacked` at nibble 12288, and once to a buffer
of its own at offset zero -- the shape proven above. For
`x = 98, 112, −73, 80, 13, 14, −106, 24` the low halves should be
`−6, −8, −1, −8, 5, 6, −2, 0`. The witness holds `4, 3, 1, −2, −3, −5, 3, 6`,
and the place in `stacked` still holds a previous run's residue.

Both are true at once, so there are two defects, not one:

1. The low half in `held.work` is already wrong when it is emitted, while the
   high half from the same `SplitHalves` call -- `6, 7, −5, 5, 0, 0, −7, 1`,
   exactly `floor(x / 16)` -- is right.
2. The copy to `stacked` at a non-zero offset does not land, although the
   identical copy lands in the probe and the witness copy at offset zero
   lands here.

The next cut is the one difference left between probe and kernel: the probe's
vector half runs unconditionally, while `Split` here runs inside
`if (sub == 0)` on one subblock of a MIX kernel. Both defects would follow
from a subblock-gated vector op inheriting the wrong mask or repeat state.

## What the sanitizer was for

Three defects came out of `mssanitizer --tool=memcheck` in one run, each as
"illegal write of size 32, 0 bytes after the nearest allocation", naming the
buffer by its size:

- 512 bytes: `middle`. 24576: `stacked`. 8192: `partial`. An expert's offset
  was being computed in nibbles and applied to a pointer, so every expert
  wrote two regions along.
- Then reads of 928 and writes of 992 bytes off the ends: the cube works in
  sixteen-row tiles and the `Matmul` library reads and writes the whole tile
  whatever M says, while M here is four. Every buffer along M now carries a
  tile.

Each fix halved the deviating count -- 44042, then 25458 -- which is how a
real defect behaves and how a guess does not.

## Two things the algorithm owes to being told not to be naive

The split was eight float operations computing `((x + 8) mod 16) - 8` and then
the remainder. It is now five, and it starts from `floor(x / 16)`, which is an
INT4 for every INT8 rather than only for those within 119. The eight that the
low half still owes comes back as `8 * sum(w)` a column -- a function of the
weight alone, so a loader computes it once and the epilogue adds it where it
is already scaling. No clamp, no correction term, no range lost.

The test data was an arithmetic progression, whose low nibbles are a sawtooth
of period sixteen. A search for where a half actually landed then matches at
every aligned phase and answers almost anywhere; worse, a wrong answer can
hide behind a right one. With a pseudo-random generator the same search
returns −1 and means it.

## The elimination list, and where it stops

Everything below is measured, not reasoned. The low half of the split is
correct and deterministic in a buffer of its own and never appears in the
buffer it is addressed into.

Eliminated as the cause:

- The split arithmetic. A dedicated witness buffer holds
  `-6, -8, -1, -8, 5, 6, -2, 0` for `x = 98, 112, -73, 80, 13, 14, -106, 24`,
  which is exact, on three consecutive runs. The earlier "wrong data" reading
  was a witness buffer shared by four blocks -- the third false signal this
  probe produced, after sawtooth test data and a high-half comparison that
  could not tell two nibble conventions apart.
- The write mechanism. An isolated probe lands all 2048 values at offsets 256
  through 24576, under `__mix__`, under `ASCENDC_CUBE_ONLY`, and with the
  vector half gated to one subblock -- every factor the real kernel has.
- The declared buffer length. `bufferSize_` has four readers in the whole
  devkit and none on the `DataCopy` path; over-declaring it only disables a
  CPU-debug abort.
- Cross-pipe synchronisation. Every `PipeBarrier<PIPE_ALL>` straddling a real
  hazard is now a named `MTE2_V`, `V_MTE3` or `MTE3_V` pair -- `PipeBarrier`
  is the single-pipe primitive and `PIPE_ALL` is absent from the headers' own
  `SUPPORTED_PIPE` and flagged `AKA3001`. No change.
- Where the base is computed. Deriving it on the device through a byte-pointer
  cast and passing it in already offset from the host fail identically.
- Alignment. An INT4 `DataCopy` moves 64 elements at a quantum; the spans
  here are 2048 and 128.

What is left is one difference. The write that lands targets a separate
allocation; the write that does not targets the same allocation as the high
half, 6144 bytes in. Two `GlobalTensor<int4b_t>` into one allocation, written
from one function, and only the one at offset zero takes.

That is now a question for the reference work rather than for another build of
this kernel. The INT8 form is unaffected, correct and 3.3 times the vendor, so
it is what integration should carry.


## The fix: `DataCopyPad` with an explicit byte count

Replacing every UB-to-global store with `DataCopyPad`, given a byte count
rather than an element count, makes the split land where it is addressed and
the gate-and-up projection exact:

```
low                               -6  -8  -1  -8   5   6  -2   0
low halves addressed at nibble 12288, found at 12288
product, first 4 columns: want 8883 562 -5267 -13984
                           got 8883 562 -5267 -13984
```

The effect is measured; the mechanism is not settled, and two candidates
remain. One does not fit this kernel and is recorded so as not to be
re-proposed: every `CastImpl` opens with `if ASCEND_IS_AIV` while plain
`DataCopy` to global memory carries no `ASCEND_IS_AIC` guard, so a store
after a cast would be issued on the cube half with a UB pointer -- two
writers on one address. But `ASCEND_IS_AIV` is `constexpr(g_coreType == AIV)`
and this kernel's stores sit inside such an `if`, so the cube half does not
reach them.

The candidate that does fit is a byte length computed at one byte an element
for a sub-byte type, which is a documented two-times trap in this API family:
a `DataCopy` of 2048 INT4 elements writing 2048 bytes instead of 1024 would
have each tile's store overrun the next, and the final tile's store spill
exactly 1024 bytes past the region -- which is precisely the duplicate
observed at 6144 and again at 12288. `DataCopyPad` took `span / 2` bytes
explicitly and the duplicate is gone.

## And a transpose that is dropped without a word

`load_cbuf_to_cb_s4` hardcodes transpose off and discards the `ifTranspose`
flag, which is the mechanism behind the official line that an INT4 A matrix
cannot be transposed. This kernel had asked for one on its weight. The weights
are now stored K-major on the host and no transpose is requested; the exact
product above is with that layout.

## Where it stands

| experts | blocks | wall, µs | INT8 form |
| --- | --- | --- | --- |
| 4 | 4 | 35.442 | 47.108 |
| 8 | 8 | 36.897 | 47.755 |
| 16 | 16 | 38.991 | 49.276 |
| 44 | 22 | 70.342 | 91.953 |

At the per-rank shape of sixteen experts on sixteen blocks that is 21% under
the INT8 kernel, which is the halved weight traffic arriving.

Not finished: 49105 of 49152 output values still deviate. The first
projection is exact and the output store is now guarded, so the fault is in
the stage between -- the recombination, the activation, the requantisation or
the down projection. The quantised intermediate reads
`-9, -9, -5, -13, -2, 5, -7, -9` with a row scale of 0.00165845, which is
plausible but unchecked against the reference. The method that found this
defect finds that one too: print the reference's own value for the stage and
compare.


## Correct, at every shape, and four defects deep

| experts | blocks | wall, µs | against the reference | INT8 form |
| --- | --- | --- | --- | --- |
| 4 | 4 | 36.995 | 0 of 49152 | 38.761 |
| 8 | 8 | 38.149 | 0 of 98304 | 40.000 |
| 16 | 16 | **40.299** | **0 of 196608** | 41.990 |
| 44 | 22 | 73.116 | 0 of 540672 | 74.555 |

Four defects, each silent, each found by a different instrument.

**The store's byte count.** `DataCopy` with an element count on a type whose
`sizeof` is one and whose data is four bits writes twice as far as asked, so
each tile overran the next and the last spilled exactly 1024 bytes past the
region -- the duplicate that appeared 6144 bytes in and again 6144 further on.
`DataCopyPad` with `span / 2` bytes. Found by dumping raw bytes at both
candidate offsets, which was the one link six rounds of elimination never
checked.

**The rounding.** `Cast(half, float, CAST_RINT)` then
`Cast(float, half, CAST_NONE)` does not round to an integer. A `RoundMode` is
a tie-break over the *destination type's* representable set: float to half
with `CAST_RINT` rounds to the nearest half, and only float to float rounds to
an integer. The pair computed `ToHalf(x)` -- the host harness's own helper,
defined and never called -- which leaves the low nibble in [−8, 8) as a real,
so whenever `x mod 16` reached 15.5 the stored nibble was +8 and the INT4 cast
saturated it to 7, one code unit short. The band that bites is x in [−0.5, 0),
and a SwiGLU output measured against its own row maximum crowds exactly there.
One line: `Cast(value, value, CAST_RINT)`, both operands float. Found by
simulating the kernel rather than the specification -- the device's print read
−9 at exactly the two columns whose reference code is 0, where saturation
predicts −9, wrapping −24 and a correct split −8.

**The reverse edge.** `Combine` loads two INT32 halves into one `staging`
tensor back to back. `MTE2_V` says the data has arrived; `V_MTE2` says the
consumer has finished and the buffer may be refilled. Nothing in this kernel
stalls MTE2, so it ran ahead and the second load clobbered what the first
cast had not read. Adding `V_MTE2` made the row scales match the reference
exactly -- 0.00145008 and 0.000781054 -- and took the deviation from 49066 to
24571.

**The scalar edge.** A `GetValue` is a scalar-unit read, so an MTE2 load
followed by one needs `MTE2_S`, not `MTE2_V`; the vector edge says nothing
about when the scalar unit may look. `Narrow` reads its per-row scale that way
immediately after the copy. Subblock zero got away with it because `Activate`
had just run on the same tensors; subblock one arrived with a cold buffer.
Localised by counting deviations per subblock rather than in total: columns
0 to 3071 read 0 of 24576 and columns 3072 to 6143 read 24571 of 24576.

## What it is worth, honestly

4.0 to 4.6% against the INT8 kernel at the same shapes -- 40.299 against
41.990 at sixteen experts -- not the 21% the incorrect version appeared to
give. Halving the weight bytes buys 4%, which is the scalar-bound diagnosis
confirmed once more: bytes are not what binds this kernel. The INT4 form still
matters for production, because the INT8 form needs 302 MB a layer of widened
weights and a full model does not fit, but it is not the 2x the byte count
promised.
