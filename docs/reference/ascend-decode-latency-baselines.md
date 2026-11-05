# Published Ascend batch-1 decode latency, and the host-side lever catalogue — verbatim source material

Everything here is quoted as published, with URL and date. This file exists to answer two questions our other
files do not: **what batch-1 decode TPOT has anyone actually published on Ascend**, so our 10.8528 ms has a
reference frame; and **which of the levers behind those numbers are host-side or framework-level**, i.e. ones a
hand-written kernel engine can still be leaving on the table.

Our measured baseline for comparison: best TPOT **10.8528 ms**, GLM-5.2-W4A8C8, batch=1, seq 8K, single machine,
TP16, A3 (910_9392, `__NPU_ARCH__ == 2201`, CANN 9.2.0). Target 2.x ms.

## 1. ★★★ vllm-ascend PR #19033 — "Optimization of TPOT under low concurrency conditions", 910_93, measured

https://github.com/vllm-project/vllm-ascend/pull/19033 — `zhu1365377615`, opened **2026-10-16** (four days before
this note), state OPEN, 882 additions across 18 files, a backport stack of ten commits onto `releases/v0.23.0`.
Read 2026-10-20 via `gh`.

**Platform, verbatim:** 「All changes were developed and validated on Atlas 910C (Ascend 910_93, CANN 9.1.0) with
Qwen3.5-35B-A3B (TP=2), covering both plain decode and MTP speculative decoding.」 — the same chip family as our
910_9392, one CANN minor behind.

### 1.1 The measured table, verbatim

> Measured on Atlas 910C, Qwen3.5-35B-A3B, TP=2, FULL_DECODE_ONLY cudagraph, static kernel
> + AllReduce/RMSNorm fusion + fused MC2:
>
> | Scenario | Before | After |
> | --- | --- | --- |
> | c=1 TPOT (no MTP) | ~8.2 ms | ~7.6 ms |
> | c=1 TPOT (MTP=4, min_tokens=512) | ~7.0 ms | ~5.1 ms (-27%) |
> | c=4 TPOT (MTP=4, min_tokens=512) | ~8.9 ms | ~9.0 ms (unchanged) |
> | c=8 TPOT (MTP=4, min_tokens=512) | ~12.7 ms | ~12.4 ms |

and from the test section, verbatim:

> 5. **Performance check**: c=1 end-to-end decode ≈ 8.6 ms/token for this configuration (in line with the ~8 ms
>    class measured for the non-MTP path); MTP + min_tokens scenarios were benchmarked separately with the
>    numbers in the table above (**MTP acceptance rate unchanged at 0.68**, TTFT ~195ms).

> Use case: latency-sensitive online serving (e.g. customer-service chat) where single-digit-concurrency TPOT
> dominates user experience. On this workload the stack cuts c=1 TPOT from ~8.2ms to ~7.6ms without MTP, and
> from ~7.0ms to ~5.1ms (-27%) with MTP (num_speculative_tokens=4, min_tokens=512).

### 1.2 ★★★ The ten commits, verbatim — the lever catalogue

> 1. **[Perf][Compilation] Add `enable_super_kernel` option for small-operator fusion**
>    New `ascend_compilation_config` key `enable_super_kernel`; when set, passes `super_kernel_optimize` to the
>    npugraph_ex compiler backend so small operators inside static kernels are fused automatically.
>
> 2. **[Perf][Kernel] Lower MoE weight prefetch token threshold to 1**
>    `MOE_PREFETCH_TOKEN_THRESHOLD` 96 → 1 so MoE weight prefetch also stays active for small decode batches
>    (**previously skipped below 96 tokens**), improving TPOT at concurrency 1/4/8 when `weight_prefetch_config`
>    is enabled.
>
> 3. **[Perf][Compilation] Always enable SuperKernel optimization with static kernel**
>    Enables `super_kernel_optimize` unconditionally whenever static kernel compilation is on (CANN 9.1+),
>    letting the compiler apply SuperKernel stream fusion for compute-communication overlap; also adds it to the
>    compiler cache hash factors.
>
> 4. **[Perf][Kernel] Overlap shared experts with routed experts and reduce decode overhead**
>    - Runs shared experts on a side stream concurrently with routed experts when
>      `multistream_overlap_shared_expert` is enabled; **the shared-expert AllReduce stays on the side stream so
>      it overlaps with routed expert computation.**
>    - Allows `VLLM_ASCEND_ENABLE_FUSED_MC2` and `multistream_overlap_shared_expert` to coexist (removes the
>      previous mutual-exclusion check).
>    - Skips dispatch/combine/gmm2 event recording in the overlap path via a new `skip_dispatch_combine_events`
>      forward-context attribute.
>    - Caches `(1.0 + weight)` for Gemma-style RMSNorm and the Qwen3.5 q/k norm patch instead of recomputing it
>      every step.
>    - **Prefetches the next-layer shared-expert gate_up weight together with the routed gate_up weight so it
>      lands in cache during the current layer's GroupedMatmul.**
>
> 5. **[Perf][ModelRunner] Reduce per-step kernel launches and host overhead for decode**
>    - GDN: caches the transposed contiguous conv1d weight (saves one `aclnnContiguous_Transpose` launch per
>      layer per step); cat-free `rearrange_mixed_qkv` returning reshape views for single-token decode; fused
>      q/k L2Norm into a single call for single-token decode.
>    - Sampler: greedy fast path that skips the bf16→fp32 cast and returns int32 token ids directly (argmax
>      ordering is preserved since every bf16 value is exactly representable in fp32).
>    - Model runner: overlaps the mamba align-mode state copy with `update_cos_sin` on the global stream; skips
>      redundant `commit_block_table` / `copy_to_gpu` / `fill_` calls when batch shape and block-table contents
>      are unchanged; copies positions from the pinned CPU buffer instead of recomputing on device.
>    - Block table: **CPU fast path for slot-mapping on small decode batches (avoids the fixed triton kernel
>      launch overhead)**; pinned staging buffer reuse in `commit_block_table` and `copy_snapshot_to_gpu`.
>    - Rotary embedding: single `index_select` shared by cos and sin.
>
> 6. **[Perf][ModelRunner] Skip full-vocab all-gather for greedy decode via vocab-parallel argmax**
>    Greedy fast path in `execute_model`: when the whole batch is greedy with no
>    logprobs/penalties/masks/structured-output/spec-decode, a vocab-parallel argmax is computed instead of
>    materializing full-vocab logits — each TP rank computes its local argmax and only `(value, index)` pairs are
>    all-gathered, **dropping sampling communication from O(batch × vocab_size) to O(batch × 2 × tp_size) (e.g.
>    248KB → 8B per rank for a 248K vocab at TP=2)**. Pre-sampled ids are carried in `ExecuteModelState` and
>    turned directly into a `SamplerOutput`.
>
> 7. **[Perf][ModelRunner] Harden greedy vocab-parallel argmax fast path**
>    Resolves the logits processor through `model.language_model` for wrapper/multimodal models; extracts the
>    MinTokens EOS mask from the active logits processor; skips the redundant `.to(torch.int32)` cast; guards
>    accepted-token event recording with `speculative_config is not None`; **reuses a pre-allocated NPU event for
>    the mamba align-mode stream sync (removes per-step event create/destroy, ~10µs host each step)**; AllReduce+RMSNorm
>    fusion pass enabled for all compile ranges including decode.
>
> 8. **[BugFix][Kernel] Make multistream shared-expert overlap capture-safe for MTP**
>    The side-stream overlap path was not capture-safe: **during NPU graph capture the shared-experts side stream
>    is not registered with the graph pool, so any side-stream op — or even a `wait_stream()` recording an event
>    on it — fails with `aclrtAllocatorGetByStream` / `rtStreamWaitEvent` errors, crashing startup at graph
>    capture** when MTP is combined with the overlap. All side-stream usage is now gated behind `not capturing`:
>    capture records the serial default-stream path, while the overlap runs in eager warmup and at replay time.
>
> 9. **[Perf][Sample] Cache min_tokens EOS mask for spec-decode rejection sampling**
>    `MinTokensLogitsProcessor.apply_with_spec_decode` rebuilds the `(row, token)` EOS mask on the host and
>    copies it H2D from *pageable* memory every step, **serializing the async scheduling pipeline (measured ~30%
>    of step time at c=1 with MTP)**. The new `_apply_min_tokens_spec_cached` caches the device tensors during
>    the steady phase … so each step issues a single `index_put_`; rebuilds go through pinned memory on
>    batch-structure change …
>    Reworks the AllReduce+RMSNorm fusion pattern to match the actual vLLM 0.23 decode graph (2-arg
>    `unquantized_gemm`, copy/detach of the o_proj output, inline Gemma-style `weight + 1.0`, 3-arg
>    `npu_add_rms_norm_bias`).
>
> 10. **[Perf][ModelRunner] Vocab-parallel argmax for greedy spec-decode verification**
>     Extends the vocab-parallel argmax to MTP verification: `_spec_greedy_shard_argmax` computes the lm_head
>     shard locally, applies the steady-state MinTokens EOS mask on the shard, and all-gathers only
>     `(value, index)` pairs for all selected rows (target draft rows + bonus rows);
>     `AscendRejectionSampler.forward_greedy_presampled` runs the greedy rejection kernels directly on the
>     pre-sampled ids, skipping the full-vocab logits all-gather, fp32 materialization and clone. Mask indices
>     are pre-filtered to the local vocab shard on the host at cache-build time, so per-step masking is pure
>     integer indexing — **boolean-mask indexing (`masked_select` → NonZero on Ascend) forces a host-device sync
>     worth several ms per step in the async pipeline.** Can be disabled with
>     `VLLM_ASCEND_DISABLE_SPEC_GREEDY_FAST=1`.

### 1.3 The serve line, verbatim — the full env and config that produced 7.6 / 5.1 ms

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_OP_EXPANSION_MODE=AIV
export LOCAL_WORLD_SIZE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_DETERMINISTIC=false
export HCCL_BUFFSIZE=128
vllm serve /path/to/Qwen3.5-35B-A3B \
  --served-model-name dewu-chat \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 --port 5001 \
  --trust-remote-code \
  --gpu-memory-utilization 0.82 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes": [1, 2, 4, 5, 8, 20, 40, 80]}' \
  --additional-config '{"ascend_compilation_config": {"enable_static_kernel": true, "fuse_allreduce_rms": true, "enable_super_kernel": true}}' \
  --max-num-batched-tokens 8192 \
  --seed 42
```

Environment, verbatim: 「Atlas 910C (Ascend 910_93, CANN 9.1.0, torch 2.10.0, torch_npu 2.10.0, vLLM 0.23.0)
… all csrc custom ops compile cleanly for `ascend910_93`」. Determinism was checked: 「two identical requests at
`temperature=0` produce byte-identical outputs (md5 match)」.

## Relevance

The reference frame first. **5.1 ms at c=1 is the best published Ascend decode TPOT we have found**, and it is
on the same chip family as ours (Ascend 910_93 against our 910_9392), one CANN minor behind. It is also on a much
smaller model at a much smaller TP — Qwen3.5-35B-A3B at TP=2 against GLM-5.2 at TP=16 — so it is not a like-for-like
target, but it bounds expectations: the non-MTP number is 7.6 ms and the MTP number 5.1 ms, and our 10.8528 ms on
a far larger model is in the same order. Nothing published is near 2.x ms, which says the target needs the
megakernel rather than tuning. Note also that their MTP acceptance is **0.68** against our prompt-lookup ~7.1
tokens/round, so our drafting is not the weak link.

What makes this PR worth reading in full is item 2 and item 10. **`MOE_PREFETCH_TOKEN_THRESHOLD` was 96, so MoE
weight prefetch was skipped entirely below 96 tokens** — i.e. switched off at exactly batch=1, by default, and
re-enabling it is a one-constant change. If our engine has an equivalent threshold anywhere, or inherits this
one through any shared config, it is off. And item 10's aside is a general Ascend trap with a large number
attached: **boolean-mask indexing lowers to `masked_select` → `NonZero`, which forces a host-device sync worth
several milliseconds per step in an async pipeline.** Any per-step boolean mask on the host side costs that.
Item 9 is the same shape with a measured fraction: an H2D copy from *pageable* rather than pinned memory every
step 「serializing the async scheduling pipeline (measured ~30% of step time at c=1 with MTP)」.

Three items bear on problems we have open. Item 4 puts **the shared-expert AllReduce on a side stream so it
overlaps routed-expert computation** — which is a different answer to our 5.35 ms, 100%-serial TP reduce than
either MC2 (ruled out at batch=1, `low-latency-tp-reduce.md` Part 8) or an in-kernel one-sided reduce
(`megakernel-persistent-decode.md`): overlap it with the *other* expert branch rather than fusing it into the
producer. Item 4 also **prefetches the next layer's shared-expert gate_up weight during the current layer's
GroupedMatmul**, which is software pipelining across the layer boundary and is exactly what a persistent kernel
would do for free. Items 1 and 3 say `super_kernel_optimize` is now unconditional with static kernel
compilation on CANN 9.1+, which is consistent with the independently measured 9.18% weighted gain from
SuperKernel `early_start` recorded in `megakernel-persistent-decode.md` §53 — and means a vendor-path baseline
we compare against may already include it.

Item 8 is a hazard to inherit rather than a lever: **a side stream is not registered with the graph pool during
NPU graph capture, so any side-stream op — or even a `wait_stream()` recording an event on it — fails with
`aclrtAllocatorGetByStream` / `rtStreamWaitEvent`**. If we ever combine a side stream with ACL graph capture
(`ascend-fusion-mechanics.md` §13), that is the error to expect, and the fix is to record the serial path at
capture and run the overlap only at replay.

## Gaps

- **No published single-card (TP1) GLM-4.5/GLM-5 W4A8 TPOT was found.** The nearest W4A8 points are
  GLM-5.3-Flash W4A8 TP16 + EAGLE at ~27 ms/token (36.5 tok/s) and DeepSeek-V4.1-Flash W4A8 on A3 TP8+EP8 at
  30.2 ms/step for 128K single-stream (improved from 39.10 → 31.39 ms/step with full patches). Both are far
  above our 10.8528 ms, but both are much longer context.
- **No published figure at or near 2.x ms for any GLM-class MoE on Ascend.** Searched 2026-10-20 in English and
  Chinese. The closest is PR #19033's 5.1 ms on a 35B-A3B at TP=2 with MTP.
- The GLM-5.3-Flash W8A8 A3 numbers in `vllm-project/vllm-ascend` issue #17443 (TPOT flat at 15.0-16.0 ms across
  102K/205K/460K inputs, 13.98 ms avg on a 128-request mixed load) and the GLM-4.1V-9B W8A8S figures in
  `sgl-project/sglang` PR #44442 (bs=1 TPOT 13.6-15.4 ms, with 0.3-1.4 ms gained from fusing activation quant,
  `AscendQuantV2` launches per decode step **121 → 41**) were surfaced by search but not fetched in full; the
  121 → 41 launch-count reduction is the closest published analogue to our `Cast` family's 2.300 ms and is worth
  retrieving.
- `angliubc/sglang-ascend-glm53-flash-serving` (GLM-5.3-Flash on 910B with SGLang, TP32 no-EP, 36.4 tok/s decode,
  4327 tok/s 80K prefill, msmodelslim fused-MoE W8A8/W4A8, "NPU serving bring-up failure catalog") was surfaced
  but not read; its failure catalogue may overlap our silent-failure list.
