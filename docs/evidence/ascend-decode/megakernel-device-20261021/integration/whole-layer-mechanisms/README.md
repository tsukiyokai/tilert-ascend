# Whole-layer device composition

The target is one device launch for a TP16 MoE layer, including both TP
dependencies. The existing rank-local attention and routed/shared bodies do
not constitute this layer. The three dense prefix layers need a separate FFN
body; the following design concerns the 75 MoE layers.

## Data and rounding

The vLLM decoder carries two BF16 tensors between layers: the FFN result and
the retained residual. Its input norm combines them. Attention produces a
BF16 rank partial; the first TP reduction precedes post-attention AddRMSNorm.
That norm returns normalized BF16 values and the next retained residual.
The gate is BF16 by BF16 with FP32 output, with shape
`[M,6144] @ [6144,256]`. The router applies sigmoid, adds the correction bias
for selection, selects eight experts, gathers the uncorrected scores, and
normalizes their sum with epsilon `1e-20`. The router factor is one; the
outer MoE runner applies factor 2.5 after expert combination.

The routed front converts the FP32 probabilities to BF16 before using them.
The shared expert retains its independent DynamicQuant input. InitRouting's
`Div(x, amax/127)` and standalone DynamicQuant's multiply by a vector
reciprocal are different numerical boundaries. The rank-local FFN epilogue
is `BF16(BF16(routed * 2.5) + shared)`, followed by the second TP reduction.
The next residual addition belongs to the next layer's input norm.

Sources: vLLM `deepseek_v2.py:1352-1423` and
`layers/fused_moe/runner/moe_runner.py:403-420`; vLLM-Ascend
`gate_linear.py:90-100`, `router/fused_topk_router.py:224-241`, and
`fused_moe.py:230-240`. The exact files and hashes are in
[sources.json](sources.json).

## Existing device interfaces and missing interfaces

| Component | Existing entry | Required composition work |
| --- | --- | --- |
| AddRMSNorm and independent A8 quantization | `whole_layer_norm::device::Forward`, `DynamicQuantRow` | Preserve the vendor BF16 rounding points; handle first-layer residual absence separately. |
| Shared-index attention | `whole_layer_attention::FullForward` | Supply live top-k and page metadata; retain the output until TP finishes. |
| Indexed attention | `whole_layer_indexed::Forward` | Retain cache and cross-layer top-k owners. Its real-framework precision gates remain separate from composition. |
| Router | Frozen bridge device gate and AIV top-k, with independent shared A8 quantization | Compose after post-attention AddRMSNorm; retain FP32 logits and original top-k slot order. |
| Routed/shared body | `routed_body::Forward(TPipe&,...)`, extracted from the accepted frozen source | The enclosing layer owns SoC initialization, sync base and system workspace. Add an explicit publication barrier after return before the next phase. |
| FFN epilogue | Bridge `device::Epilogue` with separate BF16 multiply and add stores | Consume the body outputs after publication; retain both stores before the second collective. |
| TP | `whole_layer::AllReduceBf16` | Integrate two calls with real rank-dependent values; validate the numerical reduction contract at TP16. |

`attention_native.h:121` drains all pipes, publishes DDR writes, executes a
mixed barrier and resets the pipe. These existing phase ends permit
sequential arena reuse. Routed/shared resident AIV storage must remain live
through its own six barriers. Communication allocates 20 KiB of temporary
AIV storage after the preceding body has drained. Matmul system workspace
can be shared only after all preceding objects have ended; a later body
must explicitly select the shared system-workspace pointer.

On arch2201, mixed SyncAll reserves flags 11-13; AIV-only SyncAll uses 14.
Matmul and vendor cross-core flags must be drained before phase reuse.
`TPipe::Reset` does not supply the missing publication or rank rendezvous.
Every resident AIC/AIV must take the same mixed barrier sequence, with
blocks limited by the actual device's AIC and AIV counts.

The [callable-body gate](../../a3-194/routed-body-v2-summary.json) passes
96 changing graph replays and 3,552 raw-byte comparisons. Its return is
not a barrier: `NoTrace::FinishFinal` is empty. The
[bridge gate](../ffn-bridge212-v1-summary.json) passes all eight production
boundaries at M1/2/12/32 and 16/24 groups, including exact FP32 gate logits
against the original ND NK weight with a transposed reference operand.
These component results do not establish the composed FFN or either TP
boundary. `ffn_composed` keeps independent component and official-bridge
oracles for that next gate.

## Ranked implementation routes

1. Close the post-attention norm/router bridge, then insert both real TP
   calls. This is the shortest path to measuring an actual layer instead of
   adding independent component timings. First implement AddRMSNorm, BF16
   gate, top-k and independent shared quantization in one MIX entry. The
   gate may run on AIC while AIV quantizes the same normalized BF16 tensor;
   top-k starts only after gate output is published. Then feed these exact
   outputs into the accepted routed body and retain the two FFN epilogue
   rounding points. For M=12 the gate has 37,748,736 FLOPs, 3 MiB of weights
   and 12,288 bytes of FP32 logits. These bytes do not establish a latency
   saving. The first experiment compares every bridge boundary with the
   actual vendor operators; a second experiment uses two real ranks and
   two collectives around a nontrivial FFN dependency.

2. Replace replicated TP summation with owner-partitioned reduction followed
   by broadcast. The current SHMEM implementation sends a full BF16 tensor
   to every peer and repeats the rank-ordered FP32 sum on every rank. Let
   `B = M*6144*2` bytes. Its nonlocal payload is `(P-1)*B` per rank per
   collective. Assigning disjoint output chunks to owners, then broadcasting
   the final BF16 chunks, needs `2*(P-1)*B/P`. At TP16, M12 this changes the
   logical nonlocal payload from 2,211,840 to 276,480 bytes, an eightfold
   reduction. Each owner must still sum the BF16 inputs in ascending rank
   order in FP32 and round once. This preserves the current candidate's
   numerical order, not an unmeasured HCCL TP16 order. Two rounds, two payload
   planes and matching ready planes need a new reuse proof. The first
   two-rank experiment compares both reductions and intermediate nonlinear
   consumers with the current ordered implementation; TP16 is required to
   establish the transport benefit. The existing two-rank probe proves
   graph reuse and dependency ordering, not TP16 latency.

3. Pipeline expert CUBE and Vector tasks across experts. The accepted body
   finishes all GU tasks before any routed activation, then all activation
   before down, then all down before route stores. A task pipeline would
   publish per-expert GU readiness, let AIV produce the exact existing
   activation and scale, and admit ready down tasks while unrelated GU work
   remains. Route output addresses and original-slot combination remain
   fixed, so there is no atomic floating-point combine. AIC work is still
   GU plus down; the proposed gain is overlap of otherwise serialized AIV
   work and reduced waiting, not fewer weight bytes. A correct first probe
   retains separate GM partials and tests changing 8/81/256-expert routes,
   hot experts, zero probabilities and signed zero. It requires an explicit
   deadlock-free ready protocol and a rule preventing a task from waiting
   for work assigned to a blocked resident owner. The current global
   barriers cannot simply be deleted.

Route 1 is the next implementation because it closes a required interface
and permits an end-to-end layer measurement. Route 2 has a concrete TP16
traffic reduction but needs a distributed protocol change. Route 3 can
alter the local compute path but has the largest scheduling and numerical
verification burden.

## Performance evidence and limits

The accepted front plus shared pipeline takes 161.111 us at M12 with 81
active experts and 285.147 us at M32 with 158 active experts in the
[four-bank comparison](../../a3-162/routed-pipeline-front-bank-summary.json).
Those runs contain 30 event samples per arm. The fixed shared-eight route
case has a much shorter M12 path; it is a reachable routing pattern, not a
measurement of production speculative-token correlation.

The [indexed attention profile](../../a3-162/attention-indexed-profile-summary.json)
places substantial time in several stages, rather than one removable
launch. Its intervals include existing barriers. The
[balanced routed profile](../../a3-162/routed-balanced-profile-summary.json)
is an earlier dispatch body, not the current front-plus-shared critical
path; its per-worker maxima cannot be summed into a present saving.

At the assumed 5.441 accepted tokens per round, a 2 ms TPOT permits
10.882 ms per round, or 139.5 us per layer before other work. Multiplying
the M12 four-bank routed/shared time by 75 already exceeds that allowance.
This is budget pressure, not a model latency prediction. Closing the
device layer does not by itself prove the target. Subsequent measurements
need actual routing, both TP reductions, same-address controls, and at
least 30 event samples per arm.

## Router bridge proof obligations

The standalone router gate uses M in 1-32, 16 or 24 mixed groups, BF16 input
and gate weights, FP32 correction bias and logits, INT32 ordered IDs and
FP32 normalized probabilities. Fixed-logit tests precede gate tests. They
include exact ties, eighth/ninth ties, negative bias, all-positive and
all-negative extreme finite logits, and changing values in a captured
graph. Ordered IDs and probability bytes must equal the actual vendor API;
an unordered set comparison or a CPU sigmoid reference is insufficient.

The pinned vendor `MoeGatingTopKWithoutGroup` is a useful arithmetic source
but requires a capacity audit: its ID copy handles 256 INT32 elements,
while both associated buffers are allocated from top-k=8. A private
prototype must make both capacities cover the actual copy. The MIX row
mapping must explicitly guard flat AIV IDs below M; the original standalone
operator's `GetBlockNum()` tail convention is not a launch contract for a
24-group mixed kernel. The original Sort/Sigmoid/Gather/ReduceSum/Div order
must remain unchanged.

The [ROCm GLM-5.2 extraction](../../../../../reference/tilert-rocm-glm52-structure.md)
provides a concrete whole-layer comparison: one per-layer kernel contains
both collectives and norm/router/expert work; its indexer remains separate.
It uses FP8 weights and AMD resource ownership, so neither its arithmetic
nor latency transfers to this W4A8 A3 implementation. The
[Ascend synchronization reference](../../../../../reference/megakernel-persistent-decode.md)
records the flag and workspace obligations used above.
