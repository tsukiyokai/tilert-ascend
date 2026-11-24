# Shared input-quantization prefix

The candidate shares one BF16 load, FP32 conversion, absolute-value pass and
row maximum between the two FFN input quantizers. It preserves two independent
quantization suffixes and their existing GM outputs. Its sole parent is the
validated N512 library `c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032`.
There is no candidate device result yet.

The [source audit](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-dual-quant-source-audit/summary.json)
records the parent manifest, copied source identities, installed CANN 9.2
reduction implementation and CPU contract checks. The source copies retain
their original line numbers.

## Mechanism choice

| Rank | Mechanism | Removed work | First falsifier |
| --- | --- | --- | --- |
| 1 | Share the two input-quantization prefixes | One H=6144 BF16 read, FP32 widening, Abs and ReduceMax per token | Either A8/scale pair differs, or full-FFN matched-address timing loses |
| 2 | Batch the high/low MSD output reconstruction | Vector command and event setup for two separately loaded partials; retain each FP16 boundary | Raw route/routed/scaled output differs, or batched copies cost more than the saved setup |
| 3 | A fixed-shape GU primitive executor | Generic per-expert Matmul state transitions while retaining all integer products | Any valid-M or NZ operand differs; otherwise paired full-kernel timing decides |

The first mechanism removes repeated vector work independently of expert
occupancy. The second changes command granularity without reducing the
integer products or their GM bytes. The third has a larger implementation
surface: INT4 L1/L0 layouts, dynamic `2*count` rows and per-expert reuse events
must be established before timing. The rejected microtile Down overlap,
GU-ready plus static-GU composition, and inner-norm-join removal do not supply
evidence of benefit for these mechanisms.

For E=65, the routed GU and Down logical weight payloads are respectively
48.75 MiB and 24.375 MiB per call; this prefix change removes neither. The
current approximately 200 us full-FFN measurement already exceeds the
historical 139.51 us average whole-layer allowance. This candidate addresses
one repeated operation, not that complete budget.

## Parent data and addresses

Both quantizers read the BF16 output of the same AddRMSNorm invocation.
`source/descriptor.h:41-49` binds `dispatch.bf16_rows` to `bridge[0]`;
`bridge/source/ffn_bridge/body_native.h:17-22` passes that same address to Gate
and shared Quantize. Between the two parent reads, Gate only reads normalized
data, TopK uses logits as its workspace, and StageInput reads shared A8. None
writes `bridge[0]`. The existing overlap validators exclude aliases between
these arrays.

| Address | Type and shape | Candidate producer | First consumer |
| --- | --- | --- | --- |
| bridge[0] | BF16[M,6144] | Original norm | Gate and dual quantizer |
| bridge[6] | INT8[M,6144] | Original shared suffix | Shared StageInput |
| bridge[7] | FP32[M,1] | Original shared suffix | Shared Activate |
| body[10] | INT8[M,6144] | Original routed suffix, moved earlier | Token-owned MSD SplitTokens |
| body[11] | FP32[M] | Original routed suffix, moved earlier | Routed Activate |
| bridge[4], bridge[5] | FP32[M,8], INT32[M,8] | Original TopK | Original front metadata |
| body[12:17], body[20] | Counts, starts, source rows, slots, row weights, active list | Original front metadata | Original routed/shared body |
| body[17] | Shared payload byte arena | Original StageInput and shared stages | Original shared stages |

The public interface remains 12 bridge plus 21 body tensors, a 408-byte
BodyArguments descriptor, and the existing C ABI argument order. Both new
destinations already have checked capacity, dtype, ND format, device and
nonoverlap. No new tensor, host route read, prepare-time initialization,
capture allocation or TensorImpl replacement is needed. Each DSO still owns
its prepare metadata and its tiling tensors. N512's entire 200-byte Down
tiling remains unchanged.

## Exact numerical boundary

Let `x` be the FP32 widening of normalized BF16 and `a` the result of the
existing basic-API `ReduceMax(Abs(x))`. The two suffixes remain distinct:

| Step | Routed input | Shared input |
| --- | --- | --- |
| Scale | Scalar FP32 `a / 127.0F` | Vector `Muls(a, 1.0F/127.0F)` |
| Normalization | Vector `Div(x, scale)` | Vector `Div(127, a == 0 ? 1 : a)`, then `Mul(x, factor)` |
| Clamp | None | `Mins(127)` then `Maxs(-127)` |
| Integer conversion | FP32 to INT32 RINT, INT32 to FP16 ROUND with deqscale 1, FP16 to INT8 TRUNC | Same three conversion boundaries |
| Zero row | Original `0/0` vector path | Original denominator-one guard |
| GM scale store | Exact 4-byte DataCopyPad | Original DataCopyCustom<float>(count=1) |

For the representable BF16 maximum `0x3c2f` (0.01068115234375), ideal IEEE FP32
scalar division gives scale bits `0x38b060c2`, while multiplication by the
FP32 reciprocal gives `0x38b060c1`. This CPU witness forbids sharing the scale;
it does not emulate the CANN vector reciprocal or prove device codes.

The parent shared call passes `false` explicitly and the routed call uses
its default value. Both resolve to the installed count overload with
`calIndex=false`. For FP32 H=6144 it generates 96 first-stage repeats, three
second-stage repeats selecting value lanes, and one final repeat selecting
three values. The installed implementation has no branch on source/work
aliasing. Its relative lane order is identical in the two calls. The local
basic-API documentation explicitly permits source/work storage reuse; its
repository commit and hash are recorded separately from the installed SDK.

For finite values, widening is exact and max selects an existing absolute
value. For NaNs, this design preserves the actual reduction tree rather than
substituting an associative mathematical max or a CPU reduction. NaN payload
selection and nonfinite float-to-integer conversion remain device assertions.
The first device boundary test must compare the two original quantizers with
the proposed dual quantizer directly, using raw bytes and fixed safe route
metadata; nonfinite inputs must not be sent through TopK to establish this
local property. No new finite-only public input contract is introduced.

## AIV buffer reuse

The original shared DynamicQuantRow's one 92,192-byte allocation suffices.

| Byte range | Existing view | Lifetime in the dual quantizer |
| --- | --- | --- |
| [0,12288) | BF16 input / FP16 narrow result | Input through widening, then routed FP16, then shared FP16 |
| [12288,36864) | FP32 value | Unmodified widened input through the routed suffix; original shared Mul then overwrites it |
| [36864,61440) | FP32 other | Abs source through ReduceMax; first 32 bytes hold routed scale; then original shared factor |
| [61440,86016) | FP32 work / INT32 | Reduction workspace; routed divisor, quotient and integers; then shared divisor and integers |
| [86016,92160) | INT8 codes | Routed codes through completed store, then shared codes |
| [92160,92192) | FP32 scalar[8] | Original amax in lane 0 and seven original zero lanes through routed work; original shared Muls |

Execute the routed suffix first. Its Divide writes `work`, not `value`, so
shared still receives the original FP32 input. Store routed scale from the
32-byte-aligned `other` view. This avoids overwriting or reconstructing the
shared scalar buffer and preserves its unused seven zero lanes.

Retain the original routed V_S and S_V ordering around scalar division,
all vector RAW barriers and both conversion boundaries. After routed A8 and
4-byte scale stores, MTE3_V completes the reads of codes and other before
either becomes a vector destination again; retain the routed MTE3_MTE2
back-edge as well. The shared suffix keeps its original barriers,
SetDeqScale(half(1)), padded scale store and final MTE3_V. There is no
BF16-to-FP32 widening overlap and no cross-row 32-byte GM scale write.

The original norm allocation remains the full-FFN peak, 195,584 bytes. The
dual-quant phase stays at 92,192 bytes. Body phase 0 retains its original
83,968-byte allocations, including StageInput and the active-list storage;
the later routed-body peak remains 116,480 bytes. TBuf scope exit alone is
not treated as allocator reclamation: each reduction in live allocation
occurs at an existing pipe.Reset after a complete EndPhase or barrier 0.

## Schedule and publication

1. Original norm and its caller EndPhase publish normalized BF16. Keep the
   inner norm SyncAll as well.
2. Original AIC Gate runs concurrently with the AIV dual quantizer. AIV row
   `r` owns all four outputs for token `r`. For every M=1..32 and groups=16
   or 24, `M <= 2*groups`, so each active AIV handles one row. Inactive AIVs
   still enter the caller EndPhase.
3. The unchanged Gate/quant EndPhase drains both new routed stores and the
   shared stores, performs DDR publication and a mixed join, then resets UB.
4. Original TopK and its EndPhase run unchanged. Only GM data survive.
5. Shared StageInput copies bridge[6] into its existing split-K payload.
   Original front metadata writes counts, starts, source/slot, rounded
   probabilities and the full active table. Only front's input-quantization
   loop is omitted. It does not clear or rewrite body[10]/body[11]. Keep its
   original allocations, PIPE_ALL, DDR, barrier 0 and Reset in this variant.
6. The complete body, five body joins, token-owned MSD split, N512 Down,
   shared activation arithmetic, token-owned output and terminal completion
   remain unchanged.

The candidate's private bridge helper receives body[10]/body[11] from the
existing descriptor; the private body entry explicitly selects a metadata-only
front. The public standalone quantizers and default front preparation remain
available unchanged. Metadata-only preparation is legal only after the dual
quantizer's existing EndPhase. It must not become an implicit default for an
independently launched body or a future callable entry.

The installed SetSysWorkspace behavior is unchanged: later calls do not
force a switch away from the first selected workspace. No timing or safety
claim relies on a newly selected body system arena.

## Counterexamples and acceptance

| Failure mechanism | Required check |
| --- | --- |
| Shared suffix receives routed quotient instead of original FP32 x | Keep value separate from routed work; compare both code tensors |
| `amax/127` is reused as shared scale | The BF16 `0x3c2f` witness must retain two different FP32 scales |
| NaN ordering changes under a different reduction tree | Same installed tree plus direct raw-byte quant tests with NaNs at repeat boundaries and distinct payloads |
| Zero-row guard leaks into routed quant | Compare routed zero-row codes and scales with its original Div path; retain shared's independent guard |
| A8 or scale store reads a UB view after reuse | Keep MTE3_V before replacing codes/other/narrow; poison outputs before each direct test |
| Earlier writes race with another phase | Assert all 33 public tensors do not overlap; verify body StageInput/front have no writer to body[10]/[11] |
| Quantization of a stale or different normalized version | Check the descriptor's exact bridge[0] pointer and preserve original norm/gate/TopK dataflow |
| New work lengthens the Gate barrier more than it shortens front preparation | Judge uninstrumented whole-FFN events with both owner arenas |

For a token worker, let P be load/widen/Abs/max time, S the shared suffix,
R the routed suffix and G the longest Gate AIC. The idealized change is
`max(G,P+S) + (StageInput+Metadata+P+R)` to
`max(G,P+R+S) + (StageInput+Metadata)`. Maxima must be taken across the actual
workers. This is a dependency model, not a latency prediction; register
allocation, GM placement and CUBE/vector contention can change all terms.

At M12, one removed load is 147,456 logical bytes and one removed
widen/Abs pass has 73,728 elements; at M32 these become 393,216 bytes and
196,608 elements. There are respectively 12 and 32 removed H6144 reductions.
These are operation counts, not measured HBM traffic or saved microseconds.

After CPU source/ABI checks, first validate both quantized arrays and both
FP32 scales against their frozen originals, including zero, signed zero,
largest/smallest finite BF16, ties and nonfinite payload cases. Then run the
actual original-Decoder FFN oracle, all M1/2/12/32 by 16/24 groups, changing
inputs and routes, all retained boundaries and immediate consumers. Timing
uses the same N512 parent, both owner allocations, all 33 common scratch
addresses, unchanged immutable tilings, 16 independent weight banks and at
least 30 paired event samples per arm. Preserve fresh-write poisoning and
post-event raw-byte checks. Device execution remains separate from this
source audit.
