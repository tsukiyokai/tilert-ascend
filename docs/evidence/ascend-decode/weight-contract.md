# GLM-5.2 W4A8C8 weight contract

Scope: `Eco-Tech/GLM-5.2-w4a8c8`, ModelScope revision `67e2b7157ac7d8ce1d15d7251853ce146819ceb9`, stored at `a3-gs:/data/shanshan/trt-asc/models/GLM-5.2-w4a8c8` (`/workspace/models/GLM-5.2-w4a8c8` in `trt-asc`). Source inspection on 2026-10-16 uses vLLM Ascend `f558b58c177916424ae9dcf9bb8e66c0b9031236` and vLLM `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`. These are loader and arithmetic contracts; no quantized model execution was performed for this audit.

## Model entry and rotation

The checkpoint architecture is `GlmMoeDsaForCausalLM`. The Ascend registry selects `AscendGlmMoeDsaForCausalLM` in `models/deepseek_mtp.py`; `glm5next/model.py` is a different model entry. [Registry](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/models/__init__.py#L72-L99)

| Tensor | Stored format | Runtime use |
| --- | --- | --- |
| `optional/quarot.safetensors:global_rotation` | FP32 `[6144,6144]` | Optional auxiliary-model basis conversion. The GLM-5.2 backbone and its DeepSeek MTP entry do not call `get_rotation_matrix`. |
| `rot.safetensors:rot.weight` | BF16 `[6144,6144]` | Backbone loader drops `rot.*`. MTP applies `h @ rot.weight.T` before its `hnorm` on every forward. |
| `model.layers.78.embed_tokens.weight` | BF16 `[154880,6144]` | Loaded into the MTP model embedding; the proposer may replace it with the target embedding. |
| `model.layers.78.shared_head.head.weight` | BF16 `[154880,6144]` | Checkpoint-provided MTP logits head; the proposer can alias an identical target head. |

Use the exported backbone weights directly. Adding an online global rotation to each residual or embedding would change the inspected execution path. `global_rotation` and `rot.weight` are separate artifacts; their algebraic relationship is not established by their names or shapes. [Backbone exclusion and MTP correction](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/models/deepseek_mtp.py#L16-L84), [global matrix loader](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/utils.py#L1869-L1893).

MTP then computes `eh_proj(cat(enorm(embedding), hnorm(corrected_hidden)))`; position-zero embeddings are masked first. The inspected DeepSeek implementation returns pre-final-norm states for logits and post-final-norm states for recycling. GLM5Next instead returns the normalized state twice, so copying its return contract would be incorrect here. [DeepSeek MTP](https://github.com/vllm-project/vllm/blob/c4df37dfcc7f72ca41cd43801a468526ecf99a2b/vllm/model_executor/models/deepseek_mtp.py#L113-L144), [GLM5Next comparison](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/models/glm5next/mtp.py#L89-L104).

For Ascend's V1 proposer, PP=1 MTP shares the target embedding when `use_compress` is false, without comparing the embedding values. Its logits head is shared only if absent from the checkpoint or exactly equal to the target head. File presence therefore does not determine the final runtime storage or embedding choice. This audit did not compare the full embedding/head tensors. [Embedding sharing](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/spec_decode/llm_base_proposer.py#L486-L565), [head sharing](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/spec_decode/llm_base_proposer.py#L611-L636).

## W8A8 and packed W4A8

The quantization description declares `group_size=0`. Attention Q/KV input projections, Q output projection, attention output projection, indexer `wq_b`, dense FFNs and shared experts use `W8A8_DYNAMIC`. Routed experts use `W4A8_DYNAMIC`. `kv_b_proj`, indexer `wk`/`weights_proj`, router and embeddings remain floating point. Dispatch by each tensor's recipe, rather than the model-wide `model_quant_type=W8A8_DYNAMIC`.

W8A8 performs dynamic per-token INT8 activation quantization and per-output-channel weight dequantization. Reconstructing floating weights and retaining BF16 activations exercises different arithmetic. The native W8A8 path does not consume `weight_offset`; a replacement should establish that its loaded offsets are zero before using the symmetric formula. [W8A8 implementation](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py#L60-L143)

For a logical W4 matrix `[N,K]`, the checkpoint stores INT8 bytes `[N/2,K]`. Each byte's low nibble is output row `2j`, and its high nibble is row `2j+1`; sign-extend each nibble into `[-8,7]`. Runtime transposition and INT32 packing are storage conversions, not additional quantization. The runtime INT64 scale carrier contains FP32 bit patterns in its low 32 bits; a numeric float-to-integer cast is incorrect. [Packing and scale conversion](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/quantization/methods/w4a8/w4a8.py#L131-L185), [conversion implementation](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/quantization/methods/w4a8/w4a8.py#L345-L371).

| Per-expert tensor | Checkpoint | TP16 rank `r` |
| --- | --- | --- |
| gate/up packed weight | `[1024,6144]` | Output packed rows `[64r:64(r+1)]`, logical `[128,6144]` |
| gate/up scale and scale-bias | `[2048,1]` | Output rows `[128r:128(r+1)]` |
| down packed weight | `[3072,2048]` | Input columns `[128r:128(r+1)]`, logical `[6144,128]` |
| down weight scale | `[6144,1]` | Replicated |
| down scale-bias | `[6144,16]` | Column `r`; sum only the columns owned by the rank |

The loader maps `scale_bias` into the group-scale sharding path before post-load summation. Summing all 16 down columns and then copying the result to every TP rank duplicates the correction. [Scale-bias mapping](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/quantization/configs/modelslim_config.py#L54-L105), [group-scale attribute](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/quantization/method_adapters.py#L304-L318), [post-load reduction](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/quantization/methods/w4a8/w4a8.py#L656-L665).

`scale_bias` compensates the A3 kernel's activation decomposition. For signed INT8 activation `a`, the kernel uses `hi=floor(a/16)` and `lo=(a & 15)-8`, so `16hi+lo=a-8`. It adds `b=8*sum_K(qW*sW)` before multiplying by the activation scale. A direct signed INT8 × signed INT4 reference already computes `a*qW`; adding this correction again would introduce an error. [Activation decomposition](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h#L164-L201), [correction order](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h#L131-L182).

This A3 MSD kernel reads FP16 partial products and combines them in FP32. Its mathematical equivalence to a direct integer dot product does not establish bitwise equivalence of the rounded outputs. [Partial-product conversion](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h#L112-L138).

A bounded checkpoint read independently confirmed this identity for output rows 0 and 1 of layer 3, expert 0, gate/up/down: all six offsets were zero and all scale-bias errors were exactly zero, including all 16 down columns. Tensor payload read: 14,528 bytes. For example, gate row 0 has `sW=1.3709068298339844e-6`, `sum(qW)=-127`, and `b=-0.0013928413391113281`. This sample establishes the local packing/correction interpretation; it does not validate every expert.

## Indexer precision, rotation and sharing

The following cache layout requires `attention_config.indexer_kv_dtype="int8"` and BF16 model dtype. Ascend also filters quantized layers using explicit per-layer recipes. With the indexer dtype switch disabled, the presence of `indexer_quant_type` alone does not activate this cache layout. [Enable condition](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/ascend_config.py#L914-L931), [layer filter](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/ascend_config.py#L1178-L1246).

| Layers | Indexer key cache per token |
| --- | --- |
| `6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66` | 128 INT8 values + one FP16 scale = 130 B |
| `0,1,2,70,74` | 128 BF16 values, no scale tensor = 256 B |
| MTP `78` | Full indexer with 128 BF16 values, no scale tensor = 256 B |
| Shared backbone layers | No separate indexer key cache; reuse the current full layer's top-k buffer |

The cache spec explicitly selects model dtype and `scale_dim=0` for unquantized layers. Thus a single logical scan of all backbone indexer keys at 8192 tokens is `8192*(16*130+5*256)=27,525,120 B` per replicated rank, before cache/block effects and repeated reads. MTP adds its own 2,097,152 B at the same length. This accounts for stored bytes, not measured HBM traffic. [Cache allocation](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/worker/v2/attn_utils.py#L290-L324).

Quantized indexer Q and K receive interleaved RoPE first, then right-multiply the normalized Sylvester Hadamard matrix `H128/sqrt(128)` in BF16, then dynamically quantize each 128-element row to INT8. Q scales are per token/head; K has one scale per token. Scales are converted to FP16, and the A3 quantized indexer also converts head weights to FP16. The inspected checkpoint index contains no `indexer.q_rot` or `indexer.k_rot` tensors; this backend constructs the Hadamard matrices itself. [Hadamard, quantization and cache write](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/attention/indexer.py#L192-L260), [K ordering](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/attention/indexer.py#L320-L353), [Q ordering](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/attention/indexer.py#L447-L492), [A3 selector](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/device/device_op.py#L482-L518).

Layers 3–5 reuse layer 2. For each full layer `f` in `6,10,...,74`, layers `f+1` through `f+3` reuse its indices. They still own and update their separate MLA latent/RoPE caches. MTP always constructs a full indexer; later draft steps may reuse its top-k indices, but still write new indexer keys. [Construction](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/patch/worker/patch_deepseek_v2.py#L47-L64), [runtime sharing](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/attention/sfa_v1.py#L2057-L2083).

## Integration and parity

1. Keep the existing BF16 engine as the HF reference. Add explicit per-layer quantization dispatch, packed-weight views and TP slices; retain both raw checkpoint metadata and the runtime representation.
2. Name the reference implementation. Ascend's patched Q/KV low-rank RMSNorm uses `config.rms_norm_eps=1e-5`; Transformers 5.14.1 `GlmMoeDsaAttention` constructs those norms with the default `1e-6`, matching the current BF16 engine. Exact parity with both formulas is not a valid requirement. [Ascend norm construction](https://github.com/vllm-project/vllm-ascend/blob/f558b58c177916424ae9dcf9bb8e66c0b9031236/vllm_ascend/patch/worker/patch_deepseek_v2.py#L134-L158).
3. Preserve indexer LayerNorm FP32 gamma/beta when matching vLLM: both the checkpoint and `Indexer.k_norm(dtype=torch.float32)` use FP32. The BF16 engine currently casts these parameters to BF16. Checkpoint FP32 storage alone does not impose FP32 runtime weights on ordinary RMSNorm, whose default dtype follows model construction. [Indexer constructor](https://github.com/vllm-project/vllm/blob/c4df37dfcc7f72ca41cd43801a468526ecf99a2b/vllm/model_executor/models/deepseek_v2.py#L701-L720).
4. Compare integer projections and scales before layer logits. Validate zero offsets for each loaded W8A8/W4A8 matrix before using symmetric quantization. Cover both nibble signs, TP down columns, a selected INT8 indexer layer and a BF16 indexer layer. Indexer validation must exceed top-k length; a short prefix where all positions are selected cannot validate sparse selection. Preserve the declared tie rule separately from quantization errors. MTP validation must include its input correction, effective embedding/head, and recycled hidden state.

Checkpoint metadata SHA256:

| File | SHA256 |
| --- | --- |
| `config.json` | `817f5fb39ca5d4c4b5648de89ca00deaea7537d8c2f130172a459252a05c1073` |
| `quant_model_description.json` | `a8242dd24f993b914ce567d6850ac86f044651d098a0b463a83ce63623c51985` |
| `quant_model_weights.safetensors.index.json` | `0e69a90937d7b14d4c699bbfbe0d86f1c476f4ee90c38b2b91236f1a25cfed82` |
