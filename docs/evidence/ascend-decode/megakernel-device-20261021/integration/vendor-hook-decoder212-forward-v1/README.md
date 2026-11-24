# Local decoder differential contract

The frozen fixture calls the original `DeepseekV2DecoderLayer.forward`
for actual GLM layers 6 and 7. Layer 6 owns a full indexer; layer 7 consumes
its shared top-k buffer. Both layers have the original TP16/EP1 W4A8 routed
experts and W8A8 shared experts. The original model loader configures packed
quantization names, and both INT8 shared weights plus `A8_INT_FUSED` are
required before execution.

The comparison changes only the installed attention body. Both arms retain
the official norms, router, shared/routed FFN and decoder forward method.
Rank 0 executes one local shard; fifteen actual Gloo participants supply TP16
construction geometry. The original W4A8 constructor obtains a real HCCL
name from a separate rank-0 singleton. Neither group arrangement validates
a sixteen-rank NPU reduction.

Every decoder call must reach exactly two explicit TP cuts, inside the
actual attention and FFN module calls. The first records pre-AR1 and returns
an independently generated BF16 `[M,6144]` post-AR1 input, identical in both
arms. The second records and returns local pre-AR2. Layer 7 therefore consumes
layer 6's local partial and separately returned residual. These inputs define
the local comparison; they do not approximate a numerical TP16 sum.

The gate compares both pre-reduction tensors, both norm outputs, the final
local output, retained residual, ordered top-k and every cache byte. It checks
that the returned residual is the actual post-attention norm's owner and that
the attention module returned the supplied post-AR1 input. A single NPU top-k
owner is supplied to both constructors before moving parameters, checked
after official postprocessing and cache binding, and retained through both
calls. Layer 7 may not change its contents. Each arm restores every cache
and top-k buffer before the two-layer chain.

Changing inputs use both absent and supplied initial residuals. Each case
records raw output hashes, nonzero counts, sequence lengths and native body
hits. The [five CPU checks](cpu-cut-contract.json) exercise cut routing,
missing/extra reductions, geometry, cleanup and signed-zero byte witnesses.
They do not establish device numerical correctness. Device acceptance is
recorded separately from this [frozen source manifest](manifest.json).
