# Raising speculative acceptance beyond prompt-lookup — verbatim source material

Collected 2026-10-20 for the GLM-5.2-W4A8C8 / Ascend A3 / TP16 / batch=1 ultra-low-latency effort.
Context this material is judged against: prompt-lookup drafting saturates acceptance near **7.1 tokens/round**; TPOT = round / ~7, so a 2 ms TPOT needs a ~14 ms round; a decode round is 53.05 ms unprofiled and current best TPOT is 11.6198 ms. GLM-5.2 is an agentic coding model that copies its input heavily.

Everything below the heading is quoted verbatim from the named source. Chinese text is left in Chinese.

---

### SuffixDecoding — 7.8 mean accepted tokens/step on SWE-Bench against PLD's 3.2 (the single most relevant measurement)

https://arxiv.org/abs/2411.04975 · https://arxiv.org/html/2411.04975v3
Gabriele Oliaro, Zhihao Jia, Daniel Campos, Aurick Qiao (Snowflake AI Research; Carnegie Mellon University). Code: https://github.com/snowflakedb/ArcticInference

Abstract, verbatim:

> Speculative decoding is widely adopted to reduce latency in large language model (LLM) inference by leveraging smaller draft models capable of handling diverse user tasks. However, emerging AI applications, such as LLM-based agents, present unique workload characteristics: instead of diverse independent requests, agentic frameworks typically submit repetitive inference requests, such as multi-agent pipelines performing similar subtasks or self-refinement loops iteratively enhancing outputs. These workloads result in long and highly predictable sequences, which current speculative decoding methods do not effectively exploit. To address this gap, we introduce SuffixDecoding, a novel method that utilizes efficient suffix trees to cache long token sequences from prompts and previous outputs. By adaptively speculating more tokens when acceptance likelihood is high and fewer when it is low, SuffixDecoding effectively exploits opportunities for longer speculations while conserving computation when those opportunities are limited. Evaluations on agentic benchmarks, including SWE-Bench and Text-to-SQL, demonstrate that SuffixDecoding achieves speedups of up to 5.3×, outperforming state-of-the-art methods—2.8× faster than model-based approaches like EAGLE-2/3 and 1.9× faster than model-free approaches such as Token Recycling. SuffixDecoding is open-sourced at https://github.com/snowflakedb/ArcticInference.

The explicit indictment of prompt-lookup:

> However, existing speculative decoding approaches fall short in meeting these dual requirements. Model-based methods can use significant GPU time when speculating long sequences, and can incur memory contention and kernel-level transitions (Chen et al., 2024b; Li et al., 2024) that must be managed carefully. Conversely, existing model-free approaches, such as prompt-lookup decoding (PLD) (Saxena, 2023), achieve low overhead and rapid token generation, but typically lack adaptivity. These methods speculate a fixed number of tokens irrespective of acceptance likelihood, leading to wasted computational resources on verifying long and improbable draft sequences.

The headline numbers, verbatim:

> We compare SuffixDecoding with EAGLE-2, EAGLE-3, PLD, and Token Recycling on SWE-Bench and AgenticSQL. We also run the Spec-Bench standard dataset, which is a more traditional non-agentic workload. Fig. 4 shows the results. First, on the agentic workloads, SuffixDecoding outperforms all baselines. In AgenticSQL, SuffixDecoding obtains a mean speedup of 5.3× over vanilla decoding, a 2.8× improvement over EAGLE-2/3, and 1.9× higher than Token Recycling. In SWE-Bench, EAGLE-2/3 fail due to their maximum sequence length limitations. SuffixDecoding obtains a mean speedup of 2.5× over vanilla decoding, a 1.7× improvement over PLD, the next best baseline. SuffixDecoding's superior performance in agentic workloads can be attributed to its consistently higher mean accepted tokens per decoding step. In AgenticSQL, SuffixDecoding reaches 6.3 mean accepted tokens per step—substantially higher than EAGLE-3 (3.6 tokens) and Token Recycling (3.2 tokens). In SWE-Bench, SuffixDecoding achieves 7.8 mean accepted tokens per step, while PLD only accepts 3.2 tokens per step on average.

> The hybrid approach also performs well in AgenticSQL, achieving a 4.1× speedup in the tree variant, significantly better than the 1.9× speedup from standalone EAGLE-2/3 and the 2.7× speedup from Token Recycling. These speedups are achieved thanks to the hybrid approach's impressive 7.5 mean accepted tokens per step, more than 2× higher than EAGLE-2/3 and Token Recycling. SuffixDecoding has a slightly lower mean acceptance length of 6.3, but its much lower speculation cost and higher acceptance rate make it the winning solution in agentic tasks (5.3× average speedup compared to the 4.1× speedup of the hybrid approach).

Measurement setup (batch size 1 — the same regime as ours):

> Figure 4: Speculative speedups (top) and mean accepted tokens per step (bottom) compared to vanilla decoding for SuffixDecoding and baseline methods on three benchmarks: Spec-Bench, AgenticSQL, and SWE-Bench. Experiments use Llama-3.1-8B-Instruct on a single H100 GPU with batch size 1. Speedup is measured as the ratio of wall-clock time-per-output-token relative to vanilla decoding. Suffix (tree) and Hybrid (tree) use SuffixDecoding's tree speculation algorithm, which constructs a speculation tree from the suffix tree for parallel verification. Suffix (linear) and Hybrid (linear) use a simpler linear speculation approach that only allows sequential token chains. The hybrid variants combine SuffixDecoding with EAGLE-3, dynamically selecting between suffix-based and model-based speculation based on pattern match confidence. Note that EAGLE-2/3 and Token Recycling failed to run on several SWE-Bench tasks due to long context lengths (>8192 tokens), indicated by missing bars. Spec-Bench represents a non-agentic workload and is included for comparison.

The algorithm — two suffix trees, greedy tree expansion, the scoring function:

> For each request, we consider speculating token sequences from (1) the prompt and output of that request, and (2) the outputs of prior requests. Doing so captures the source of output token repetition from many agentic algorithms, including self-consistency, self-refinement, and multi-agent pipelines.

> SuffixDecoding leverages suffix trees to perform fast pattern matching and find possible continuations of token sequences. Suppose the prompt and output tokens of an ongoing inference is x_{1:t}. Consider a suffix x_{t-p+1:t} of length p, which we will refer to as the pattern sequence. We walk the suffix tree starting from the root node, and at each step taking the child that corresponds to token x_{t-p+i}. If no such child exists, then the pattern is not found and SuffixDecoding reverts to standard non-speculative decoding. Otherwise, after p steps, we arrive at a node whose descending paths are the possible continuations of the pattern sequence.

> In reality, we found it convenient to maintain two different suffix trees: a global tree for the previously generated outputs, and a separate per-request tree for the current ongoing inference request. This circumvents the complexities and overheads due to synchronizing the suffix tree updates from multiple concurrent requests. The global tree can be constructed offline in O(n) time, while the per-request tree can be efficiently constructed and updated online (Ukkonen, 1995).

> Although suffix trees are memory-efficient at O(n) space, the global tree can still become large when there are many previous outputs. However, they only require CPU memory, which is typically plentiful and under-utilized in LLM serving scenarios.

The scoring math (this is the expected-accepted-length calculation):

```text
C(N) = COUNT(N) / sum_{M in CHILDREN(PARENT(N))} COUNT(M)

D(N) = D(PARENT(N)) × C(N),  if N != N_p
     = 1,                    otherwise
```

> where COUNT(N) is the number of occurrences of node N in the reference corpus, which can be computed when constructing the suffix tree. Starting with the single node N_p in our speculation sub-tree, we consider all children of all of its leaf nodes, and add the node N with the highest D(N). This process is repeated until the sub-tree reaches a predetermined size limit, MAX_SPEC.

> Intuitively, C(N) estimates the probability that TOKEN(N) would be the next observed token in a subsequence TOKEN(N_p),…,TOKEN(PARENT(N)), and D(N) estimates the probability that TOKEN(N) would be ultimately accepted by the speculative tree verification, assuming the output tokens follow historical patterns. Thus, SuffixDecoding builds the speculation tree by greedily adding leaf nodes that it believes to be the most likely to be accepted during verification.

```text
SCORE(T_spec) = sum_{N in T_spec} D(N).
```

> Intuitively, if D(N) estimates the probability that node N in a speculation tree T_spec would be accepted, then SCORE(T_spec) estimates the expected number of accepted tokens. SuffixDecoding then selects the T_spec with the highest SCORE as the final speculation tree to be verified.

Adaptive speculation length — the mechanism that breaks the PLD plateau:

> Figure 2: (a) the mean number of accepted tokens increases with the length of the pattern match, which motivates MAX_SPEC = αp. (b) shows that this choice achieves a better trade-off between acceptance rate and speculative speedup.

> To guide how to adaptively set MAX_SPEC, we observed that the number of accepted tokens in practice typically increases with longer pattern sequence lengths p (Fig. 2(a)). Thus, we define MAX_SPEC adaptively as
> MAX_SPEC(p) = αp,
> where α is a user-defined max speculation factor. Fig. 2(b) shows that setting MAX_SPEC adaptively according to the pattern length results in a better trade-off between acceptance rate and speculative speedup. In practice, we found that α ∈ [1,4] works well for agentic applications.

Hybrid fallback and the threshold rule:

> Lastly, we find that SCORE(T_spec) can be used to dynamically decide between using SuffixDecoding or falling back to a model-based speculation method, which is useful for practical scenarios when the workload can be mixed between agentic and more diverse applications. Specifically, for each decoding iteration, we always speculate using SuffixDecoding first. If SCORE(T_spec) > τ, where τ is a configurable threshold, then SuffixDecoding's draft tokens are used. Otherwise, we use a fall-back speculation method, such as EAGLE-3 (Li et al., 2025a).

> As a practical guideline, we find that setting τ close to the mean accepted tokens of the fallback speculator works well for mixed workloads, while using SuffixDecoding alone (τ=0) is optimal for highly repetitive agentic tasks.

> Table 3 shows the wall-clock speedup results on Spec-Bench for various threshold values. For open-ended generation tasks, we find that setting τ close to or slightly exceeding the mean accepted tokens (MAT) of the model-based speculator yields the best performance. For instance, EAGLE-3 achieves a MAT of approximately 4.65 tokens/step on Spec-Bench. The results show that τ ∈ [5,7] produces the highest overall speedups (2.5×), with performance being relatively stable across this range.

> In summary, practitioners should set τ ≈ MAT_fallback for mixed or open-ended workloads, where MAT_fallback is the mean accepted tokens of the fallback model-based speculator. For highly repetitive agentic workloads, using SuffixDecoding alone (τ=0 or very low values) yields the best performance.

Observed tree shapes on structured output:

> Fig. 5 shows an example of a speculation tree constructed by SuffixDecoding. We observed many instances of large speculation trees that branch at each boolean true/false value of several consecutive features. These speculation tree always contains a branch with high acceptance, advancing output generation by dozens of tokens or more in one step.

Corpus-size sensitivity — acceptance rate is nearly flat, speedup is not:

> In Fig. 8, we show the speedup and acceptance rate of SuffixDecoding on WildChat and Magicoder across a range of suffix tree sizes between 256 and 10,000 output examples. First, we note a promising pattern: the speedup consistently improves as the size of the suffix tree grows. This indicates that SuffixDecoding can learn useful patterns even in workloads with lower token repetition, and may be a substitute for model-based methods when a draft model is not available.
>
> Second, perhaps surprisingly, the acceptance rate does not change much even when the suffix tree size varies across almost two orders of magnitude. We believe this is primarily due to the effect of the adaptive speculation length MAX_SPEC = αp. Although less data may mean less certainty in the speculated tokens, the pattern matches are also shorter, which results in fewer speculated tokens.

How to predict in advance whether suffix drafting will pay off:

> SuffixDecoding tends to perform better on more structured tasks compared to very open-ended ones (e.g., AgenticSQL vs WildChat). We can measure this "structuredness" using empirical entropy. The steps are as follows: (1) create a suffix tree from example model outputs (100 examples is typically enough), (2) calculate the entropy of each node's output distribution by determining how often each child node is accessed, and (3) compute a weighted average of this entropy across all nodes. A low average entropy indicates that output tokens are more predictable based on their prefixes, which generally suggests that Suffix Decoding will perform better.

The real agentic-coding deployment setup:

> We conducted the end-to-end SWE-Bench experiment on a 8xH100 80GB GPU cluster, with 1TB RAM. We served the all-hands/openhands-lm-32b-v0.1-ep3 model locally using vLLM, with a tensor parallelism degree of 4 and with prefix caching enabled. We used the flashinfer kernels for sampling. We made some minor modifications to vLLM to record the per-request and per-step statistics of interest (time-per-token latency, throughput, acceptance length, acceptance rate). We used the same settings for all baselines. We ran the OpenHands daemon on the same machine, and used the OpenAI API to interact with the vLLM server. We ran OpenHands with the CodeActAgent (Wang et al., 2024b) with ITERATIVE_EVAL_MODE=true, and a maximum of 100 iterations, as recommended by the OpenHands authors. We used a maximum of 16 concurrent workers to run the SWE-Bench tasks.

Cross-domain transfer without adaptation:

> Fig. 9 shows the results. SuffixDecoding starts from having 4,000 output examples from WildChat, and begins to receive SpiderSQL inference requests. Without any adaptation, SuffixDecoding still achieves 1.5× speedup and 8% acceptance rate, but is far from the 2.6× speedup and 20% acceptance rate it would achieve if it were trained on 500 examples from SpiderSQL instead.

## Relevance
This is the one published measurement directly comparable to our 7.1 tokens/round: **7.8 mean accepted tokens/step on SWE-Bench** for suffix-tree drafting against **3.2 for PLD**, at batch size 1 on an agentic coding workload. Two mechanisms explain the 2.4x gap, and both are available to us: matching against a *global* tree of previous outputs rather than only the current prompt, and `MAX_SPEC = αp` — speculating proportionally to the observed match length instead of a fixed K. Our 7.1 already exceeds their 7.8-benchmark baseline's PLD figure by a wide margin, which suggests our prompt-copy rate is unusually high; the suffix-tree gain over us may therefore be smaller than 2.4x, but the adaptive-length mechanism is orthogonal to the match source and should still apply.

---

### vllm-ascend speculative decoding — the Ascend hard ceiling of 16 tokens per decode round, and the `suffix` method already wired up

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/speculative_decoding.md
(vLLM Ascend plugin, `main` as of 2026-10-20)

The supported-method table, verbatim:

```text
| Method | Description |
| ------ | ----------- |
| `ngram` | Match n-grams from the prompt |
| `suffix` | Suffix-based pattern matching (requires Arctic Inference) |
| `medusa` | Medusa heads embedded in the target model |
| `eagle` | EAGLE-based draft model |
| `eagle3` | EAGLE-3 based draft model |
| `mtp` | Multi-Token Prediction with shared embedding head |
| `dflash` | Block diffusion-based parallel draft model |
| `dspark` | Semi-autoregressive block drafting with a sequential Markov logit-bias head |
| `draft_model` | Generic external draft LLM |
| `extract_hidden_states` | Extract hidden states for EAGLE training |
```

**The hard ceiling**, verbatim:

```text
!!! note

    On Ascend NPUs, the `npu_fused_infer_attention_score` operator supports a maximum of 16 tokens per decode round. Therefore, `(num_speculative_tokens + 1)` must be ≤ 16.
```

The architecture:

```text
vLLM Ascend implements speculative decoding through a **proposer-verifier** architecture:

1. **Proposer** (`vllm_ascend/spec_decode/`): Generates draft (speculative) tokens using various methods — from simple n-gram matching to neural-network-based draft models.
2. **Rejection Sampler** (`vllm_ascend/sample/`): Verifies draft tokens against the target model's output, accepting matches and rejecting mismatches, with optional optimizations including Block Verify and Entropy Verify.
```

```text
- **`num_speculative_tokens`** (int, required): Number of speculative tokens to generate per forward pass. Auto-filled from the draft model's `n_predict` config (e.g., MTP) or `suffix_decoding_max_tree_depth` (suffix method) when available.
- **`draft_tensor_parallel_size`** (int, optional): Tensor parallelism size for the draft model. Can only be `1` or the same as the target model's tensor parallel size.
- **`disable_padded_drafter_batch`** (bool, default: `False`): Disable input padding for speculative decoding. If set to `True`, speculative input batches can contain sequences of different lengths, which may only be supported by certain attention backends. **Note:** Only effective with `eagle`, `eagle3`, `mtp`, `dflash`, `draft_model`, and `extract_hidden_states` methods.
```

n-gram / prompt-lookup configuration:

```python
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
        },
    )
```

Suffix Decoding section, verbatim:

```text
## Speculating using Suffix Decoding

The following code configures vLLM to use speculative decoding where proposals are generated using Suffix Decoding [(SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications)](https://arxiv.org/abs/2411.04975).

Like n-gram, Suffix Decoding can generate draft tokens by pattern-matching using the last `n` generated tokens. Unlike n-gram, Suffix Decoding (1) can pattern-match against both the prompt and previous generations, (2) uses frequency counts to propose the most likely continuations, and (3) speculates an adaptive number of tokens for each request at each iteration to get better acceptance rates.

Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts.

!!! note

    Suffix Decoding requires Arctic Inference. You can install it with `pip install arctic-inference`.
```

```python
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        enforce_eager=True,
        speculative_config={
            "method": "suffix",
            "num_speculative_tokens": 15,
        },
    )
```

```shell
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
      --tensor-parallel-size 1 \
      --enforce-eager \
      --speculative-config '{"method": "suffix", "num_speculative_tokens": 15}'
```

MTP, including the TP16 reference configuration:

```text
## Speculating using MTP

MTP (Multi-Token Prediction) boosts inference performance by parallelizing the prediction of multiple tokens, shifting from single-token to multi-token generation. This approach significantly increases generation throughput and achieves multiplicative acceleration in inference speed — all without compromising output quality.
```

```python
    llm = LLM(
        model="deepseek-ai/DeepSeek-V3.2-Exp-W8A8",
        tensor_parallel_size=16,
        enable_expert_parallel=True,
        max_model_len=36768,
        max_num_seqs=10,
        quantization="ascend",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 2,
            "disable_padded_drafter_batch": False,
        },
    )
```

Dynamic speculation — adapting K at runtime:

```text
## Dynamic Speculative Decoding

Dynamic Speculative Decoding adapts the number of draft tokens (K) at runtime, instead of always using a fixed `num_speculative_tokens`. This helps keep speculative decoding beneficial as concurrency and draft confidence change. vLLM Ascend currently provides two approaches:

- **Confidence-based verify length (DSpark / DFlash)**: adjust the per-request verify length from draft confidence. DSpark uses a dedicated confidence head; DFlash is **head-free** and uses `max(softmax(logits))` of the drafted token as a confidence proxy.
- **Batch-size based (Autoregressive)**: select a shared K from concurrency ranges via `num_speculative_tokens_per_batch_size`.
```

```text
When `dynamic_spec_config.method` is set to `"dspark"` or `"dflash"`, the corresponding proposer:

1. Runs the draft model as usual to produce up to `num_speculative_tokens` draft tokens.
2. Estimates per-position acceptance likelihood for each request:
   - **DSpark**: sigmoid output of the DSpark confidence head.
   - **DFlash (head-free)**: `max(softmax(logits))` of the argmax draft token (no extra neural head).
3. Periodically recomputes a shared per-request verify budget from those confidence scores.
4. Allocates that budget across requests (each request keeps at least `min_verify_tokens`), so each request verifies only a prefix of draft tokens.
```

```text
!!! note

    This is an **exploratory** feature and currently targets **model runner v1** only. Supported dynamic methods are `"dspark"` (confidence head) and `"dflash"` (head-free). Among DSpark draft models, only the **Qwen** series is supported today; support for models such as GLM and DeepSeek is being added incrementally.
```

Acceptance-criterion relaxations — Block Verify and Entropy Verify:

```text
vLLM Ascend provides two optional optimizations for the rejection sampler in speculative decoding: **Block Verify** and **Entropy Verify**. These features trade a small amount of output precision for improved inference throughput.

!!! warning

    Both Block Verify and Entropy Verify modify the token acceptance criteria and may cause minor precision degradation (e.g., slightly different output tokens compared to the standard rejection sampler). Evaluate the quality impact on your specific workload before enabling them in production.

### Block Verify

Block Verify evaluates all draft tokens as a block using cumulative probability products, rather than checking each token independently. This can improve the acceptance rate and reduce the overhead of rejection sampling, especially when `num_speculative_tokens >= 3`.

### Entropy Verify

Entropy Verify adjusts the acceptance threshold based on the entropy of the target distribution:

- **High entropy** (uncertain distribution) → lower effective threshold → more tokens accepted
- **Low entropy** (confident distribution) → higher effective threshold → stricter rejection

This entropy-aware threshold is controlled by two parameters:

- **`posterior_threshold`** (default: `0.95`, range: `(0, 1]`): The upper bound of the modified threshold. Even when entropy is very low, the effective threshold will not exceed this value.
- **`posterior_alpha`** (default: `0.4`, range: `>= 0`): Controls how strongly entropy influences the threshold. A higher alpha makes the threshold more sensitive to entropy changes, resulting in a higher acceptance rate for speculative tokens but also greater precision loss. You need to tune this value based on your specific model and dataset. When alpha is `0`, entropy has no effect and the threshold equals `posterior_threshold`.
```

```shell
    vllm serve <model> --additional-config \
        '{"rejection_sampler_config": {"enable_block_verify": true, \
        "enable_entropy_verify": true, "posterior_threshold": 0.95, \
        "posterior_alpha": 0.4}}'
```

Synthetic rejection sampling:

```text
In addition to the default `standard` rejection sampling (which accepts a draft token when `target_prob / draft_prob >= uniform`), vLLM Ascend supports **synthetic** rejection sampling. In this mode each draft token at position `i` is accepted with a configured probability `rates[i]`, compared against a uniform random draw — **independent of the target/draft probability match**. The first rejected position stops the request and emits a recovered token; if every draft token is accepted, the bonus token is appended.
```

## Relevance
The hard constraint that governs everything: **`num_speculative_tokens + 1 ≤ 16`** because `npu_fused_infer_attention_score` takes at most 16 tokens per decode round. Our measured 7.1 accepted tokens/round sits below but not far below that ceiling; pushing acceptance past ~15/round is not possible with this attention operator at all, so the achievable TPOT floor from speculation alone is round/15. `method: "suffix"` with `num_speculative_tokens: 15` is already a supported configuration on Ascend — the maximum the operator allows — and `Block Verify` plus `Entropy Verify` are two documented ways to raise the accepted count at a stated precision cost. Note `synthetic` rejection sampling exists purely as an acceptance-rate instrument, useful for isolating how much of a round-time win comes from acceptance versus from the kernel.

---

### SAM-Decoding — suffix automaton instead of suffix tree, O(1) per step

https://arxiv.org/abs/2411.10666 · https://arxiv.org/html/2411.10666v3 · code: https://github.com/hyx1999/SAM-Decoding
Hu et al.

As published:

> The method is retrieval-based and uses a suffix automaton to find the longest suffix match, rather than fixed n-gram matching. It achieves an average time complexity of O(1) per generation step.
> It builds a static automaton over a text corpus and a dynamic one over the current prompt and generated text. The suffix automaton matches the suffixes of the generating text and retrieves the draft from the text corpus and the generating text respectively according to the matching position.
> Assuming the matching length of the longest suffix implying the quality of retrieval draft tokens, the method can be integrated with other speculative decoding methods, deciding whether to adopt auxiliary decoding techniques.
> Drafts are then verified by the LLM, and accepted tokens are used to extend the automata.
> With Token Recycling, it reports a speedup of 2.27× over autoregressive decoding on Spec-Bench. With EAGLE-2, it reaches a speedup of 2.49×, surpassing all current approaches.
> It is 18%+ faster than other retrieval-based SD methods.

## Relevance
Same two-tree (static corpus + dynamic current-text) structure as SuffixDecoding, but with an O(1)-amortised suffix automaton rather than a suffix tree, and the *same* use of match length as a confidence signal — "assuming the matching length of the longest suffix implying the quality of retrieval draft tokens". That confirms match length as the right adaptive-K signal independently of SuffixDecoding. No mean-accepted-length figures are published, only speedups on Spec-Bench, which is a non-agentic workload — so it does not give a number comparable to our 7.1.

---

### FLUX / decode-phase speculation interaction — the serial cost of draft + verify

https://arxiv.org/abs/2411.04975 (SuffixDecoding, § motivation)

> To efficiently handle the long repetitions common in agent-driven applications, speculative decoding methods must satisfy two critical requirements. First, they need to generate draft tokens rapidly and with minimal overhead, enabling maximal exploitation of long speculation lengths. Second, they must do so adaptively—only generating more draft tokens when acceptance likelihood is high and fewer tokens when acceptance likelihood is low, to prevent verification from becoming a bottleneck.

> Model-based methods can use significant GPU time when speculating long sequences, and can incur memory contention and kernel-level transitions (Chen et al., 2024b; Li et al., 2024) that must be managed carefully.

From the Arctic Inference documentation (https://arcticinference.readthedocs.io/en/stable/suffix-decoding.html):

> a specialized, model-free speculative decoding method... enabling rapid draft token predictions using CPU resources without GPU overhead

> There is a global suffix tree holding outputs from completed requests and a per-request tree built from the current prompt and generated tokens.

## Relevance
The decisive structural property for us: suffix/n-gram drafting is **model-free and runs on CPU**, so the draft stage adds no device work and no extra kernel launches to the round. Every model-based drafter (EAGLE/EAGLE-3/MTP/Medusa) adds a device-side forward per round, which at our ~3008 kernels and 0.70–13.58 us launch floors is a direct addition to round time. That is the quantitative reason to raise acceptance by improving the match source and the adaptive length rather than by adding a neural drafter — and it is why vllm-ascend's own GLM-5.2 reference config sets `"enforce_eager": true` on the MTP drafter (see FILE 1) rather than capturing it in the graph.

---

### Snowflake's production write-up — SuffixDecoding in Arctic Inference and vLLM

https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/
(Snowflake engineering blog)

> SuffixDecoding at Production Scale with Arctic Inference and vLLM

As described in the paper that this blog accompanies, and as surfaced in the Arctic Inference docs:

> Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts.

## Relevance
Code-editing is named as a first-class target. Our workload — an agentic coding model that copies its input heavily — is exactly the case the method was designed for, which is also why our prompt-lookup acceptance is already as high as 7.1.

---

### Single-chain n-gram / prompt-lookup match policies — the exact code and defaults in Saxena's original, HF transformers, vLLM, TRT-LLM and llama.cpp

Sources, all retrieved 2026-10-20 and pinned (a `git clone --filter=blob:none` of each repo; line numbers below refer to that commit):
- https://github.com/apoorvumang/prompt-lookup-decoding @ cada4fe6ec7368d41c386ec3fb94fa11268971eb (Apoorv Saxena, "Prompt Lookup Decoding", November 2023; last commit 2024-08-23)
- https://github.com/huggingface/transformers @ 9f5750c67ad4c2e077dd3883d20503f3d77b2fe5 (2026-10-20)
- https://github.com/vllm-project/vllm @ fce64bc5c2c939a4521bfb9b462c8c74c1c13893 (2026-10-20)
- https://github.com/NVIDIA/TensorRT-LLM @ 083447cd22b228a36cf30d21385c624be905cddd (2026-10-20)
- https://github.com/ggml-org/llama.cpp @ 84663953535d4323e92b4bc90971aa022c305d06 (2026-10-19)

#### 1. Saxena's original (the algorithm every other implementation descends from)

> **TLDR**: We modify speculative decoding where we replace the draft model with simple string matching in the prompt to generate candidate token sequences. This results in significant speedups (2x-4x) in input-grounded tasks, with no effect on output quality. This method can be used with any decoder model without model changes or external datastore, and with both greedy and sampling techniques.

> **Intuition**: In several LLM use cases where you're doing _input grounded generation_ (summarization, document QA, multi-turn chat, code editing), there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs.

`README.md` lines 28-56:

````python
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    input_length = input_ids.size(1)

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)
````

> Input to this function is the same as to the draft model - all the tokens till the current generation step (`input_ids`). It then tries to match last few tokens to somewhere earlier in the prompt. If found, it returns the next-k token continuation as `candidate_input_ids` or candidate sequence. The 2 parameters are `max_ngram_size`, which is the maximum ngram to use when looking for matches in the prompt. `num_pred_tokens` is the candidate sequence length to return after match is found.

> - **GPU**: Single A100 40GB
> - **Model**: Mistral-7B-Instruct-v0.1
> - **Decoding Type**: Greedy decoding
> - **Hyperparams**: Matching max n-gram size = 3, Continuation length = 10

The only coding datum in the README, and the author's own list of open policy questions:

> - roleplay has the worst gain. This is probably because there isn't many ngrams to copy, since each generation is sort of unique.
> - coding has very high gain in 2nd turn, because there is lots of code copying
> - in first turn, extraction has highest gain. This agrees with our hypothesis - in extraction there is definitely n-gram copying, and PLD should help

> - There's probably better ways to do string matching than the current one, and there are several obvious things to improve eg. what to do when there are multiple matches? Whats the ideal length of continuation?
> - We haven't yet tried sampling, although there's no reason it shouldn't work.
>     - Here, one additional thing to test would be whether prompt lookup while sampling can affect hallucination rates, since this artifically increases probability of sampling exact sequences from input (this was suggest by my colleague Shwetha S)
> - Testing actual FLOPs impact and tradeoffs is needed
> - Also need to figure out best hyperparams - 3 and 10 were chosen on very little testing

#### 2. HF transformers `PromptLookupCandidateGenerator` (the reference many engines copy)

`src/transformers/generation/candidate_generator.py` lines 1017-1060:

````python
class PromptLookupCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information: https://github.com/apoorvumang/prompt-lookup-decoding

    Args:
        eos_token_id (`torch.Tensor`, *optional*):
            The token id of the end of sequence token.
        num_output_tokens (`int`, *optional*, defaults to 10):
            The number of tokens to be output as candidate tokens.
        max_matching_ngram_size (`int`, *optional*, defaults to 2):
            The maximum ngram size to be considered for matching in the prompt
        max_length (`int`, *optional*, defaults to 20):
            The number of total maximum tokens that can be generated. For decoder-only models that includes the
            prompt length. Defaults to 20, which is the max length used as default in generation config.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step. In
            prompt lookup assisted generation, they are not used to manipulate probabilities, but rather to find
            forbidden tokens (p = -inf) and block them from being valid candidates.
        vocab_size (`int`, *optional*):
            The size of the vocabulary. Required if `logits_processor` is provided.
    """

    def __init__(
        self,
        eos_token_id: torch.Tensor | None = None,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = 2,
        max_length: int = 20,
        logits_processor: Optional["LogitsProcessorList"] = None,
        vocab_size: int | None = None,
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.logits_processor = logits_processor
        self.vocab_size = vocab_size

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

````

`src/transformers/generation/candidate_generator.py` lines 1076-1112:

````python
            return input_ids, None

        chosen_ids = None
        match_found = False
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            # TODO (joao): this finds the first valid candidates (left to right), but perhaps we should find the
            # longest valid candidates?
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length, self.max_length)

                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]

                    # Check if the each new candidate token is forbidden according to the logits processor. If all
                    # tokens are allowed, we keep `chosen_ids` as is.
                    # 1. create random logits.
                    # 2. apply the logits processor to get output logits for the next token, using the arbitrary
                    #    logits as input.
                    # 3. compare the output logits with the next candidate token. If they are -inf, then the next
                    #    candidate token is forbidden and we don't want to generate it.
                    if self.logits_processor is not None:
                        sequence_with_candidate = input_ids
````

`src/transformers/generation/candidate_generator.py` lines 1130-1154:

````python

                    match_found = True

                    # remove remaining candidate ids if an "eos" token is found, otherwise the target model may
                    # accept eos and the rest as valid, thus not stopping generation after "eos"
                    # NOTE: below code is written based on the fact that assisted decoding supports only bs=1
                    if self.eos_token_id is not None:
                        mask = torch.isin(chosen_ids, self.eos_token_id)
                        match_indices_eos = torch.nonzero(mask)
                        if match_indices_eos.numel() > 0:
                            first_eos_index = match_indices_eos[0].item()
                            chosen_ids = chosen_ids[:first_eos_index]
                    break
            if match_found:
                break

        # In case we didn't find a match return the input sequence unchanged, reverts back to autoregressive decoding
        if not match_found or chosen_ids is None or len(chosen_ids) == 0:
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids, None
````

Config-level documentation of the same two knobs:

`src/transformers/generation/configuration_utils.py` lines 339-342:

````text
        prompt_lookup_num_tokens (`int`, *optional*):
            The number of tokens to be output as candidate tokens.
        max_matching_ngram_size (`int`, *optional*):
            The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.
````

#### 3. vLLM `ngram` — config defaults and the match routine

`vllm/config/speculative.py` lines 476-481:

````python
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""
````

`vllm/config/speculative.py` lines 1235-1263:

````python
        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"

        if self.method in ("ngram", "ngram_gpu"):
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
````

`vllm/v1/spec_decode/ngram_proposer.py` lines 12-36:

````python
class NgramProposer:
    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Pre-allocate buffers for numba batch propose.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)

        # Threshold of total number of tokens in the batch to enable
        # multi-threading in numba batch propose.
        self.num_tokens_threshold = 8192
````

`vllm/v1/spec_decode/ngram_proposer.py` lines 209-295:

````python
@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
````

vLLM's own documentation example (`docs/features/speculative_decoding/n_gram.md`):

`docs/features/speculative_decoding/n_gram.md` lines 7-21:

````python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
outputs = llm.generate(prompts, sampling_params)
````

#### 4. TensorRT-LLM `NGram` — the pool keeps several matches per pattern, then emits exactly one

`tensorrt_llm/_torch/speculative/ngram.py` lines 14-49:

````python
class NGramPoolManager(BaseResourceManager):
    """
    Drafter for NGram. This class maintains the pattern-matches pairs for NGram drafter.

    For example, one of the existed pairs could be: ["I","love"] -> [["apple", "because", "it", "is"], ["banana", "and"]].

    Here we call ["I","love"] as `pattern`, and [["apple", "because", "it", "is"], ["banana", "and"]] as `matches`.

    `pattern` is a list of token ids. The pool emits corresponding draft tokens from the matches if the pattern appears at the tail of the generated sentence.

    `matches` is a list of candidate draft token ids attaching to a pattern.

    Arguments:
        max_total_draft_tokens: int
            The length maximum of draft tokens (can be understood as length maximum of output draft tokens).

        max_matching_ngram_size: int
            The length maximum of searching tokens (can be understood as length maximum of input tokens to search).

        is_keep_all: bool = True
            Whether to keep all candidate pattern-matches pairs, only one match is kept for each pattern if False.

        is_use_oldest: bool = True
            Whether to provide the oldest match when pattern is hit, the newest one is provided if False.

        is_public_pool: bool = True
            Whether to use a common pool for all requests, or the pool is private for each request if False.

    Members:
        pool: dict[tuple[int], OrderedSet[int]] or dict[int, dict[tuple[int], OrderedSet[int]]]
            If is_public_pool == True, it maps from patterns to matches
            If is_public_pool == False, it maps from request ID to the request-specific pool

        start_index: dict[int, int]
            It maps from request ID to the index of the prompt to update the pool in the next step.
    """
````

`tensorrt_llm/_torch/speculative/ngram.py` lines 84-141:

````python
    def get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        max_sequence_length: int,
    ):
        prefix_len = len(prefix)
        max_draft_token_length_this_step = max_sequence_length - 1 - prefix_len
        if max_draft_token_length_this_step <= 0:  # No draft token is need if the prefix is long enough
            return []

        if request_id not in self.start_index:  # Extend start_index and pool for a new request
            self.start_index[request_id] = 0
            if not self.is_public_pool:
                self.pool[request_id] = {}

        pool = (self.pool if self.is_public_pool else self.pool[request_id])

        # Update pool
        sequence = prefix[self.start_index[request_id]:]
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            # Find each possible pattern-match combination, and use tuple for hash
            for l in range(len(sequence) - size):
                r = min(l + size + self.max_total_draft_tokens, len(sequence))
                pattern = tuple(sequence[l:l + size])
                new_match = tuple(sequence[l + size:r])
                if pattern not in pool or \
                    (not self.is_keep_all and len(new_match) > len(pool[pattern][0])):
                    # Replace the match if
                    # 1. the pattern does not exist in the pool
                    # 2. only one match is kept, and the new match is longer (MRU)
                    pool[pattern] = OrderedSet((new_match, ))
                elif new_match not in pool[pattern]:
                    # Update the matches if the pattern is already existed:
                    # TODO: need a strategy to maintain the short candidates, now we just remove them
                    # Drop all existed matches with small length
                    for match in pool[pattern]:
                        if len(match) < len(new_match):
                            pool[pattern].remove(match)
                    pool[pattern].add(new_match)

        draft_tokens = []
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            pattern = tuple(prefix[-size:])
            if pattern not in pool:
                continue
            draft_tokens = pool[pattern][0 if self.is_use_oldest else -1]
            draft_tokens = list(draft_tokens)[:max_draft_token_length_this_step]
            break

        # Update start_index
        self.start_index[request_id] = max(
            0, prefix_len -
            (self.max_total_draft_tokens + self.max_matching_ngram_size - 1))

        return draft_tokens
````

`tensorrt_llm/_torch/speculative/ngram.py` lines 161-181:

````python
class NGramDrafter(Drafter):

    _needs_padding_kv_extension = True

    def __init__(
        self,
        spec_config: NGramDecodingConfig,
        ngram_pool_manager: NGramPoolManager = None,
    ):
        super().__init__(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            max_concurrency=spec_config.max_concurrency,
            draft_len_schedule=spec_config.draft_len_schedule)
        assert ngram_pool_manager is not None, "NGram needs a resource manager to maintain the pool."
        self.spec_resource_manager = ngram_pool_manager
        self.spec_config = spec_config
        self.max_draft_len = spec_config.max_draft_len
        self.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
        assert self.max_draft_len == self.max_total_draft_tokens, "NGram only supports linear tree."

````

`tensorrt_llm/llmapi/llm_args.py` lines 3021-3048:

````python
class NGramDecodingConfig(DecodingBaseConfig):
    """Configuration for NGram drafter speculative decoding."""
    decoding_type: Literal["NGram"] = Field(default="NGram")
    max_matching_ngram_size: PositiveInt = Field(
        default=2,
        description=
        "The length maximum of searching tokens (can be understood as length maximum of input tokens "
        "to search).")
    is_keep_all: bool = Field(
        default=True,
        description=
        "Whether to keep all candidate pattern-matches pairs, only one "
        "match is kept for each pattern if False.")
    is_use_oldest: bool = Field(
        default=True,
        description="Whether to provide the oldest match when pattern is hit, "
        "the newest one is provided if False.")
    is_public_pool: bool = Field(
        default=True,
        description="Whether to use a common pool for all requests, or the pool "
        "is private for each request if False.")

    @model_validator(mode="after")
    def validate_ngram_config(self):
        if self.max_draft_len is None or self.max_draft_len <= 0:
            raise ValueError("max_draft_len must be > 0 for NGram")
        self.max_total_draft_tokens = self.max_draft_len  # Current NGram only supports linear tree
        return self
````

`docs/source/features/speculative-decoding.md` lines 83-100:

````markdown
### NGram

The NGram method is an implementation of [this Prompt Lookup Decoding algorithm](https://github.com/apoorvumang/prompt-lookup-decoding).

When the NGram algorithm is used, TRT-LLM will maintain a map from token prefixes to candidate draft sequences. For example, the 3-gram ["The ", " future ", " is"] could map to the draft sequence [" bright", " because"]. The prefixes are token sequences that are extracted from the prompt and the tokens generated by the target model. The NGram pool and matching procedure can be tuned with the following options:

* `max_draft_len`: Maximum draft candidate length.
* `max_matching_ngram_size`: Maximum prompt suffix length to match with keys in the pool.
* `is_public_pool`: If true, a single ngram pool is shared for all requests. Otherwise, each request has its own ngram pool.
* `is_keep_all`: If true, draft candidates will be retained in the pool forever. Otherwise, only the largest draft candidate is retained.
* `is_use_oldest`: If true, the oldest draft candidate is always proposed for a given match. Otherwise, the newest draft candidate is used. Only applicable if `is_keep_all == True` because `is_keep_all == False` means we'll only ever have a single value for each key.

```python
from tensorrt_llm.llmapi import NGramDecodingConfig

speculative_config = NGramDecodingConfig(
    max_draft_len=3, max_matching_ngram_size=4, is_public_pool=True)

````

#### 5. llama.cpp `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` / `ngram-mod`

`docs/speculative.md` lines 126-203:

````markdown
### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple --spec-draft-n-max 64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (argument `--spec-ngram-map-k-min-hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k --spec-draft-n-max 64
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2 --spec-draft-n-max 64
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `--spec-ngram-mod-n-min` and `--spec-ngram-mod-n-max`

llama-server ... --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).
````

`docs/speculative.md` lines 317-358:

````text
### n-gram Mod Parameters

```
--spec-ngram-mod-n-match                N
                                        ngram-mod lookup length (default: 24)
--spec-ngram-mod-n-min                  N
                                        minimum number of ngram tokens to use for ngram-based speculative decoding (default: 48)
--spec-ngram-mod-n-max                  N
                                        maximum number of ngram tokens to use for ngram-based speculative decoding (default: 64)
```

### n-gram Simple Parameters

```
--spec-ngram-simple-size-n              N
                                        ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-simple-size-m              N
                                        ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-simple-min-hits            N
                                        minimum hits for ngram-simple speculative decoding (default: 1)
```

### n-gram Map Key Parameters

```
--spec-ngram-map-k-size-n               N
                                        ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k-size-m               N
                                        ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k-min-hits             N
                                        minimum hits for ngram-map-k speculative decoding (default: 1)
```

### n-gram Map Key-4-Values Parameters

```
--spec-ngram-map-k4v-size-n             N
                                        ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k4v-size-m             N
                                        ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k4v-min-hits           N
                                        minimum hits for ngram-map-k4v speculative decoding (default: 1)
````

`docs/speculative.md` lines 424-445:

````text
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
````

## Relevance
Four independent codebases agree on the single-chain policy: try the longest suffix first (Saxena `max_ngram_size=3`, HF `max_matching_ngram_size=2`, vLLM `prompt_lookup_min/max` which both default to 5 when neither is given although the field docstring still says "Defaults to 1", TRT-LLM `max_matching_ngram_size=2`), and then take exactly ONE occurrence. Which occurrence differs: HF takes the first hit scanning left to right (its own TODO asks "perhaps we should find the longest valid candidates?"), vLLM takes the longest matched n-gram and, among equal lengths, "the earliest position in the original tokens", TRT-LLM stores every distinct continuation for a pattern in an `OrderedSet` and then emits `pool[pattern][0 if self.is_use_oldest else -1]` (`is_use_oldest=True` by default), llama.cpp `ngram-simple` uses "the last n-gram in history that matches". Nobody in this list verifies two occurrences at once: TRT-LLM states it outright in code (`assert self.max_draft_len == self.max_total_draft_tokens, "NGram only supports linear tree."`) and in `NGramDecodingConfig` ("Current NGram only supports linear tree"). That is the unexploited headroom for our shape: when the same suffix occurs k times earlier in the context with different continuations, a linear drafter pays the whole-round loss every time its single guess is the wrong occurrence, while the other k-1 continuations were already sitting in the pool. The draft-length defaults span a wide range (Saxena 10, HF 10, vLLM docs 5, TRT-LLM docs 3, llama.cpp `ngram-simple` m=48 and `ngram-mod` n_max=64) and llama.cpp's own sample comment says "MoEs require long drafts", which is consistent with our MoE target needing wide drafts to amortise an expensive verify.

---

### SGLang `NGRAM` — a production n-gram TREE drafter (trie of previously seen suffixes, node-budgeted BFS-by-recency or best-first-by-frequency expansion, ancestor-row tree mask, first-child/next-sibling greedy verify, accepted-node KV compaction)

https://github.com/sgl-project/sglang @ 1739d962a45d1c8ecd10cda5a7e387b5cc18c066 (committed 2026-10-20 00:35:29 -0700; retrieved 2026-10-20). Authors: SGLang team; the C++ trie design descends from Ant Group's `Lookahead` (see the Lookahead entry below). All excerpts are verbatim with file path and line numbers.

#### 1. User-facing contract and defaults

`docs/docs/advanced_features/speculative_decoding.mdx` lines 786-791:

````markdown
## Ngram Speculative Decoding

SGLang also supports **ngram-based speculative decoding** (no separate draft model). It retrieves draft tokens from an ngram cache built from previously generated tokens, and then verifies them with the target model.

Enable it with:
- `--speculative-algorithm NGRAM`
````

`docs/docs/advanced_features/speculative_decoding.mdx` lines 840-858:

````markdown
</table>

Notes:
- Ngram speculative decoding **only supports CUDA**.
- It currently **does not support** DP attention (`--attn-dp-size` > 1).
- It disables the overlap scheduler and mixed chunked prefill.
- If `--speculative-ngram-max-bfs-breadth > 1` (thus `speculative_eagle_topk > 1`) and `page_size > 1`, use `--attention-backend flashinfer`; otherwise the server will error.
- Optional: set `SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True` to force greedy verification.

```bash Command
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 16 \
    --speculative-ngram-max-bfs-breadth 10 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs-decode 8 \
    --log-level warning
```
````

`python/sglang/srt/arg_groups/fields/spec.py` lines 249-281:

````python
    # Speculative decoding (ngram)
    # -------------------------------------------------------------------------
    speculative_ngram_min_bfs_breadth: A[
        int,
        "The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
    ] = 1
    speculative_ngram_max_bfs_breadth: A[
        int,
        "The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
    ] = 10
    speculative_ngram_match_type: A[
        Literal["BFS", "PROB"],
        "The match type for cache tree.",
    ] = "BFS"
    speculative_ngram_max_trie_depth: A[
        int,
        "The max trie depth for ngram speculative decoding.",
    ] = 18
    speculative_ngram_capacity: A[
        int,
        "The cache capacity for ngram speculative decoding.",
    ] = 10 * 1000 * 1000
    speculative_ngram_external_corpus_path: A[
        Optional[str],
        "Path to an external JSONL corpus to pre-load into SAM at startup. Additional corpora can be added at runtime via POST /add_external_corpus.",
    ] = None
    speculative_ngram_external_sam_budget: A[
        int,
        "Number of draft nodes reserved for the external SAM subtree in ngram speculative decoding.",
    ] = 0
    speculative_ngram_external_corpus_max_tokens: A[
        int,
        "Fail startup if the tokenized external ngram corpus exceeds this many tokens. Tune this based on your CPU memory budget.",
````

`python/sglang/srt/arg_groups/speculative_hook.py` lines 1205-1250:

````python
def _handle_ngram(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.device not in ("cuda", "cpu", "xpu"):
        raise ValueError(
            "Ngram speculative decoding only supports CUDA, CPU, or XPU devices."
        )

    _disable_overlap_schedule_for_cpu(server_args)

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "_handle_ngram",
            max_running_requests=48,
        )
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    declare_resolution(
        server_args,
        "_handle_ngram",
        enable_mixed_chunk=False,
    )
    declare_resolution(
        server_args,
        "_handle_ngram",
        speculative_eagle_topk=cfg.speculative_ngram_max_bfs_breadth,
    )
    if cfg.speculative_num_draft_tokens is None:
        declare_resolution(
            server_args,
            "_handle_ngram",
            speculative_num_draft_tokens=12,
        )
        logger.warning(
            "speculative_num_draft_tokens is set to 12 by default for ngram speculative decoding. "
            "You can override this by explicitly setting --speculative-num-draft-tokens."
        )
    if cfg.speculative_num_steps is None:
        declare_resolution(
            server_args,
            "_handle_ngram",
            speculative_num_steps=cfg.speculative_num_draft_tokens
            // cfg.speculative_eagle_topk,
        )
````

`python/sglang/srt/arg_groups/speculative_hook.py` lines 1274-1292:

````python

    view = resolved_view(server_args)
    if (
        cfg.speculative_eagle_topk > 1
        and view.page_size > 1
        and view.attention_backend != "flashinfer"
    ):
        raise ValueError(
            f"speculative_eagle_topk({cfg.speculative_eagle_topk}) > 1 "
            f"with page_size({view.page_size}) > 1 is unstable "
            "and produces incorrect results for paged attention backends. "
            "This combination is only supported for the 'flashinfer' backend."
        )
    if attn_dp_enabled_of(view):
        # TODO: support dp attention for ngram speculative decoding
        raise ValueError(
            "Currently ngram speculative decoding does not support dp attention."
        )

````

`python/sglang/srt/speculative/cpp_ngram/ngram_corpus.py` lines 14-36:

````python
class NgramCorpus:
    def __init__(
        self,
        max_trie_depth=18,
        min_bfs_breadth=1,
        max_bfs_breadth=8,
        draft_token_num=8,
        match_type="BFS",
        capacity=1000000,
        external_sam_budget=0,
        external_corpus_max_tokens=10000000,
    ) -> None:
        cls = get_ngram_corpus_cls()
        self._obj = cls(
            capacity=capacity,
            max_trie_depth=max_trie_depth,
            min_bfs_breadth=min_bfs_breadth,
            max_bfs_breadth=max_bfs_breadth,
            draft_token_num=draft_token_num,
            match_type=match_type,
            external_sam_budget=external_sam_budget,
            external_corpus_max_tokens=external_corpus_max_tokens,
        )
````

`python/sglang/kernels/jit/csrc/ngram_corpus/param.h` lines 17-38:

````cpp
struct Param {
  bool enable;
  bool enable_router_mode;
  size_t min_bfs_breadth;
  size_t max_bfs_breadth;
  size_t max_trie_depth;
  size_t draft_token_num;
  size_t external_sam_budget = 0;
  size_t external_corpus_max_tokens = 10000000;
  std::string match_type;

  std::vector<size_t> batch_draft_token_num;

  size_t get_draft_token_num(size_t batch_size) const {
    if (batch_size < batch_draft_token_num.size()) {
      if (batch_draft_token_num[batch_size] !=
          std::numeric_limits<decltype(batch_draft_token_num)::value_type>::max()) {
        return batch_draft_token_num[batch_size];
      }
    }
    return draft_token_num - 1;
  }
````

#### 2. The trie: every suffix of every inserted token run is inserted, up to `max_trie_depth`; each node carries a frequency and an LRU position

`python/sglang/kernels/jit/csrc/ngram_corpus/trie.cpp` lines 23-67:

````cpp
void Trie::insert(const int32_t* tokens, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    auto start = tokens + i;
    auto end = start + std::min(len - i, param_.max_trie_depth);

    if (static_cast<size_t>(end - start) > free_node_count_) {
      squeeze(end - start - free_node_count_);
    }

    TrieNode* cursor = root_;
    path_.clear();
    while (start != end) {
      auto token = *start;
      auto iter = cursor->child.find(token);
      if (iter == cursor->child.end()) {
        iter = cursor->child.insert({token, getNode()}).first;
        auto node = iter->second;

        cursor->lru.emplace_front(node);
        global_lru_.emplace_back(node);

        node->token = token;
        node->parent = cursor;
        node->parent_lru_pos = cursor->lru.begin();
        node->global_lru_pos = --global_lru_.end();
        node->freq = 1;
        cursor->sorted_children.insert(node);
      } else {
        auto node = iter->second;
        cursor->sorted_children.erase(node);
        node->freq++;
        cursor->sorted_children.insert(node);
        cursor->lru.splice(cursor->lru.begin(), cursor->lru, node->parent_lru_pos);
      }
      cursor = iter->second;
      path_.emplace_back(cursor);
      ++start;
    }

    for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
      TrieNode* node = *it;
      global_lru_.splice(global_lru_.begin(), global_lru_, node->global_lru_pos);
    }
  }
}
````

#### 3. Matching: ALL suffix lengths that exist in the trie become "anchors" (deepest first) — this is where multiple occurrences and multiple match lengths become branches

`python/sglang/kernels/jit/csrc/ngram_corpus/trie.cpp` lines 192-218:

````cpp
std::vector<std::pair<const TrieNode*, int32_t>> Trie::getExpandableAnchors_(const MatchState& state) const {
  std::vector<std::pair<const TrieNode*, int32_t>> result;
  result.reserve(state.anchors.size());
  for (size_t depth = state.anchors.size(); depth > 0; --depth) {
    const auto node = resolve(state, state.anchors[depth - 1]);
    if (node != nullptr && !node->child.empty()) {
      result.emplace_back(node, static_cast<int32_t>(depth));
    }
  }
  return result;
}

std::vector<std::pair<const TrieNode*, int32_t>>
Trie::match(const int32_t* context, size_t len, MatchState& state, size_t total_len) const {
  const bool has_forward_progress = total_len >= state.processed_total_len;
  const auto appended_len = has_forward_progress ? total_len - state.processed_total_len : 0;
  const auto expected_prev_depth = std::min(state.processed_total_len, param_.max_trie_depth);
  const bool can_advance = state.trie_epoch == trie_epoch_ && has_forward_progress && appended_len <= len &&
                           state.anchors.size() == expected_prev_depth;

  if (can_advance && advanceMatchState_(state, context + len - appended_len, appended_len, total_len)) {
    return getExpandableAnchors_(state);
  }

  rebuildMatchState_(context, len, state, total_len);
  return getExpandableAnchors_(state);
}
````

#### 4. Tree expansion policy A (`--speculative-ngram-match-type BFS`, the default): recency-ordered children, breadth shrinking with match depth, shared node budget

The breadth allotted to an anchor is `(max_match_depth - depth) * bfs_breadth_scale + min_bfs_breadth`, so the longest (most trusted) match is expanded narrowly (breadth near `min_bfs_breadth`=1) and shorter, less trusted matches are expanded widely (up to `max_bfs_breadth`=10); the breadth then decays by `bfs_breadth_scale` per tree level. Anchors are visited deepest-first and all draw from the same `draft_token_num` budget (`cursor <= draft_token_num`).

`python/sglang/kernels/jit/csrc/ngram_corpus/trie.cpp` lines 220-264:

````cpp
Result Trie::buildRecency(
    const int32_t* context,
    size_t len,
    int32_t last_token,
    size_t draft_token_num,
    const Param& param,
    MatchState& state,
    size_t total_len) const {
  auto anchors = match(context, len, state, total_len);
  const auto max_match_depth = std::max<int32_t>(1, static_cast<int32_t>(param.max_trie_depth - 1));
  double bfs_breadth_scale = double(param.max_bfs_breadth - param.min_bfs_breadth) / max_match_depth;

  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;

  for (auto [node, depth] : anchors) {
    std::queue<std::tuple<int32_t, double, const TrieNode*>> queue;
    queue.push({root, (max_match_depth - depth) * bfs_breadth_scale + param.min_bfs_breadth, node});
    while (queue.size() && cursor <= static_cast<int>(draft_token_num)) {
      auto front = queue.front();
      queue.pop();

      auto parent = std::get<0>(front);
      auto cur_breadth = std::get<1>(front);
      auto iter = std::get<2>(front)->lru.begin();

      auto breadth = std::max(1, int32_t(cur_breadth));
      for (int i = 0;
           i < breadth && iter != std::get<2>(front)->lru.end() && cursor <= static_cast<int>(draft_token_num);
           ++i, ++iter) {
        auto token = (*iter)->token;
        auto pos = -1;
        if (auto tit = tree[parent].next.find(token); tit != tree[parent].next.end()) {
          pos = tit->second;
        } else {
          pos = tree[parent].next.insert(std::make_pair(token, cursor++)).first->second;
        }
        queue.emplace(pos, cur_breadth - bfs_breadth_scale, *iter);
      }
    }
  }

  return fillResult(last_token, draft_token_num + 1, tree, root);
}
````

#### 5. Tree expansion policy B (`--speculative-ngram-match-type PROB`): best-first by cumulative frequency-normalised probability, identical in spirit to SuffixDecoding's `D(N)`

`norm_freq = freq / sum_freq * prob` is the product of child-frequency fractions along the path; the heap pops the highest cumulative probability first; each parent contributes at most `top_k = max_bfs_breadth` children.

`python/sglang/kernels/jit/csrc/ngram_corpus/trie.cpp` lines 266-331:

````cpp
Result Trie::buildFrequency(
    const int32_t* context,
    size_t len,
    int32_t last_token,
    size_t draft_token_num,
    const Param& param,
    MatchState& state,
    size_t total_len) const {
  auto anchors = match(context, len, state, total_len);
  struct CompareByLastDouble {
    bool operator()(
        const std::tuple<double, const TrieNode*, double>& a,
        const std::tuple<double, const TrieNode*, double>& b) const {
      return std::get<2>(a) < std::get<2>(b);
    }
  };

  std::priority_queue<
      std::tuple<double, const TrieNode*, double>,
      std::vector<std::tuple<double, const TrieNode*, double>>,
      CompareByLastDouble>
      heap;

  std::vector<Node> tree(draft_token_num + 1);

  int root = 0;
  int cursor = 1;
  int top_k = param.max_bfs_breadth;

  auto addToHeap = [&heap, &top_k](int parent, const TrieNode* trie_node, double prob) -> void {
    double sum_freq = 0.0;
    int count = 0;
    std::list<std::pair<TrieNode*, int32_t>> topk_children;
    for (auto* child : trie_node->sorted_children) {
      sum_freq += static_cast<double>(child->freq);
      topk_children.emplace_back(child, child->freq);
      if (++count >= top_k) break;
    }
    if (sum_freq <= 0) sum_freq = 1.0;
    for (const auto& [child, freq] : topk_children) {
      double norm_freq = static_cast<double>(freq) / sum_freq * prob;
      heap.emplace(parent, child, norm_freq);
    }
  };

  for (auto [node, _] : anchors) {
    addToHeap(root, node, 1.0);

    while (!heap.empty() && cursor <= static_cast<int>(draft_token_num)) {
      auto [parent, trie_node, prob] = heap.top();
      heap.pop();
      auto token = trie_node->token;
      int pos = -1;
      auto tit = tree[parent].next.find(token);
      if (tit != tree[parent].next.end()) {
        pos = tit->second;
      } else {
        pos = cursor++;
        tree[parent].next[token] = pos;
      }
      addToHeap(pos, trie_node, prob);
    }
  }

  return fillResult(last_token, draft_token_num + 1, tree, root);
}
````

#### 6. Tree -> flat arrays: BFS order, parent pointers, and the mask built by copying the parent's row (so row i is the ancestor set of node i)

`python/sglang/kernels/jit/csrc/ngram_corpus/result.cpp` lines 12-51:

````cpp
Result fillResult(int last_token, int draft_token_num, std::vector<Node>& tree, int root) {
  Result info;
  std::vector<int32_t> prevs;
  info.token.reserve(draft_token_num);
  prevs.reserve(draft_token_num);
  std::queue<std::tuple<int32_t, int32_t, int32_t>> queue;
  info.token.emplace_back(last_token);
  prevs.emplace_back(-1);

  for (auto [token, next] : tree[root].next) {
    queue.emplace(token, next, 0);
  }
  while (queue.size()) {
    auto [token, next, prev] = queue.front();
    queue.pop();
    info.token.emplace_back(token);
    prevs.emplace_back(prev);
    for (auto [t, n] : tree[next].next) {
      queue.emplace(t, n, info.token.size() - 1);
    }
  }

  // zero padding to length
  while (info.token.size() < static_cast<size_t>(draft_token_num)) {
    info.token.emplace_back(0);
    prevs.emplace_back(0);
  }

  int n = info.token.size();
  info.mask.resize(n * n, 0);
  info.mask[0] = 1;
  for (int i = 0; i < n; ++i) {
    if (prevs[i] != -1) {
      memcpy(&info.mask[i * n], &info.mask[prevs[i] * n], prevs[i] + 1);
    }
    info.mask[i * n + i] = 1;
  }

  return info;
}
````

#### 7. Budget split between the trie (generated text so far) and an external suffix automaton (a pre-loaded corpus), and the SAM-based builders

`python/sglang/kernels/jit/csrc/ngram_corpus/ngram.cpp` lines 144-207:

````cpp
Result Ngram::batchMatch(
    const std::vector<int64_t>& state_ids,
    const std::vector<std::vector<int32_t>>& tokens,
    const std::vector<size_t>& total_lens) {
  if (state_ids.size() != tokens.size() || state_ids.size() != total_lens.size()) {
    throw std::runtime_error("batchMatch expects state_ids, tokens, and total_lens to match in size");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  using TrieResultBuildFn =
      Result (Trie::*)(const int32_t*, size_t, int32_t, size_t, const Param&, MatchState&, size_t) const;
  using SamResultBuildFn = Result (SuffixAutomaton::*)(const int32_t*, size_t, int32_t, size_t, const Param&) const;
  TrieResultBuildFn trie_result_build_fn;
  SamResultBuildFn sam_result_build_fn;
  if (param_.match_type == "BFS") {
    trie_result_build_fn = &Trie::buildRecency;
    sam_result_build_fn = &SuffixAutomaton::buildRecency;
  } else if (param_.match_type == "PROB") {
    trie_result_build_fn = &Trie::buildFrequency;
    sam_result_build_fn = &SuffixAutomaton::buildFrequency;
  } else {
    throw std::runtime_error("Unknown match_type: '" + param_.match_type + "'. Must be 'BFS' or 'PROB'.");
  }

  // All budget values are loop-invariant (mutex_ held, sams_ won't change).
  const size_t num_sams = sams_.size();
  const auto total_draft_token_num = param_.get_draft_token_num(tokens.size());
  const size_t total_sam_budget =
      num_sams > 0 ? std::min(param_.external_sam_budget, total_draft_token_num) : size_t{0};
  const size_t per_sam_budget = num_sams > 0 ? total_sam_budget / num_sams : size_t{0};
  const size_t trie_budget = total_draft_token_num - (per_sam_budget * num_sams);

  Result merged;
  for (size_t i = 0; i < state_ids.size(); ++i) {
    const auto& suffix = tokens[i];
    if (suffix.empty()) {
      throw std::runtime_error("batchMatch received an empty token tail");
    }

    auto& state = match_state_[state_ids[i]];

    if (total_sam_budget == 0 || per_sam_budget == 0) {
      auto res = (trie_.get()->*trie_result_build_fn)(
          suffix.data(), suffix.size(), suffix.back(), total_draft_token_num, param_, state, total_lens[i]);
      merged.token.insert(merged.token.end(), res.token.begin(), res.token.end());
      merged.mask.insert(merged.mask.end(), res.mask.begin(), res.mask.end());
      continue;
    }

    auto combined = (trie_.get()->*trie_result_build_fn)(
        suffix.data(), suffix.size(), suffix.back(), trie_budget, param_, state, total_lens[i]);

    for (const auto& [_, sam] : sams_) {
      auto sam_res =
          (sam.get()->*sam_result_build_fn)(suffix.data(), suffix.size(), suffix.back(), per_sam_budget, param_);
      combined = combineRootResults_(suffix.back(), static_cast<int>(total_draft_token_num + 1), combined, sam_res);
    }

    merged.token.insert(merged.token.end(), combined.token.begin(), combined.token.end());
    merged.mask.insert(merged.mask.end(), combined.mask.begin(), combined.mask.end());
  }
  return merged;
}
````

`python/sglang/kernels/jit/csrc/ngram_corpus/suffix_automaton.cpp` lines 181-287:

````cpp
Result SuffixAutomaton::buildRecency(
    const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const {
  auto anchors = match(context, len, param.max_trie_depth);
  const auto max_match_depth = std::max<int32_t>(1, static_cast<int32_t>(param.max_trie_depth - 1));
  const double bfs_breadth_scale = double(param.max_bfs_breadth - param.min_bfs_breadth) / max_match_depth;
  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;

  for (const auto& anchor : anchors) {
    std::queue<std::tuple<int, double, int>> queue;
    queue.push(
        {root, (max_match_depth - anchor.matched_len) * bfs_breadth_scale + param.min_bfs_breadth, anchor.state});
    while (!queue.empty() && cursor <= static_cast<int>(draft_token_num)) {
      auto [parent, cur_breadth, state] = queue.front();
      queue.pop();

      const auto& children = states_[state].children_by_recency;
      const auto breadth = std::max(1, static_cast<int32_t>(cur_breadth));
      for (int i = 0;
           i < breadth && i < static_cast<int>(children.size()) && cursor <= static_cast<int>(draft_token_num);
           ++i) {
        const auto [token, child_state] = children[i];
        int pos = -1;
        if (auto iter = tree[parent].next.find(token); iter != tree[parent].next.end()) {
          pos = iter->second;
        } else {
          pos = tree[parent].next.insert({token, cursor++}).first->second;
        }
        queue.emplace(pos, cur_breadth - bfs_breadth_scale, child_state);
      }
    }
  }
  return fillResult(last_token, draft_token_num + 1, tree, root);
}

Result SuffixAutomaton::buildFrequency(
    const int32_t* context, size_t len, int32_t last_token, size_t draft_token_num, const Param& param) const {
  auto anchors = match(context, len, param.max_trie_depth);
  struct CompareByProb {
    bool operator()(
        const std::tuple<int, int32_t, int, double>& lhs, const std::tuple<int, int32_t, int, double>& rhs) const {
      return std::get<3>(lhs) < std::get<3>(rhs);
    }
  };

  std::priority_queue<
      std::tuple<int, int32_t, int, double>,
      std::vector<std::tuple<int, int32_t, int, double>>,
      CompareByProb>
      heap;
  std::vector<Node> tree(draft_token_num + 1);
  int root = 0;
  int cursor = 1;
  const int top_k = static_cast<int>(param.max_bfs_breadth);

  auto addToHeap = [this, &heap, top_k](int parent, int state, double prob) {
    if (top_k <= 0) {
      return;
    }
    const auto& children = states_[state].children_by_freq;
    if (children.empty()) {
      return;
    }
    double sum_freq = 0.0;
    int count = 0;
    for (const auto& [_, child_state] : children) {
      sum_freq += static_cast<double>(states_[child_state].occ_count);
      if (++count >= top_k) {
        break;
      }
    }
    if (sum_freq <= 0) {
      sum_freq = 1.0;
    }
    count = 0;
    for (const auto& [token, child_state] : children) {
      const auto scaled_prob = static_cast<double>(states_[child_state].occ_count) / sum_freq * prob;
      heap.emplace(parent, token, child_state, scaled_prob);
      if (++count >= top_k) {
        break;
      }
    }
  };

  for (const auto& anchor : anchors) {
    addToHeap(root, anchor.state, 1.0);
    while (!heap.empty() && cursor <= static_cast<int>(draft_token_num)) {
      auto [parent, token, child_state, prob] = heap.top();
      heap.pop();

      int pos = -1;
      if (auto iter = tree[parent].next.find(token); iter != tree[parent].next.end()) {
        pos = iter->second;
      } else {
        pos = cursor++;
        tree[parent].next[token] = pos;
      }
      addToHeap(pos, child_state, prob);
    }
  }
  return fillResult(last_token, draft_token_num + 1, tree, root);
}

}  // namespace ngram

}  // namespace sglang
````

#### 8. What is inserted into the corpus each round, and the verify-input contract (note the comment about `max_tree_depth` and `tree_topk`)

`python/sglang/srt/speculative/ngram_worker.py` lines 238-311:

````python
    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        bs = len(batch.reqs)
        stride = self.draft_token_num

        # Overlap mode processes results one iteration behind, so the last
        # round's accepted tokens are not yet in req.output_ids and must be
        # spliced in from spec_info. Sync mode and grammar batches process
        # results before the next draft prep, so output_ids is already
        # complete and splicing would duplicate the tail.
        use_prev_tokens = self.enable_overlap and not batch.grammar_needs_sync()

        # Accept-independent prep, hoisted above the blocking .cpu() below.
        req_ids = [req.rid for req in batch.reqs]
        # Only the last max_trie_depth tokens can match; the ids are
        # array.array, so list() the tails for list concat below.
        # multimodal cache padding contains identities, not vocabulary token IDs
        input_tails = [
            list(req.origin_input_ids_unpadded[-self.max_trie_depth :])
            for req in batch.reqs
        ]
        output_tails = [
            list(req.output_ids[-self.max_trie_depth :]) for req in batch.reqs
        ]
        base_lens = [
            len(req.origin_input_ids_unpadded) + len(req.output_ids)
            for req in batch.reqs
        ]

        if use_prev_tokens:
            prev_token_ids, prev_accept_lens = (
                batch.spec_info.accept_tokens,
                batch.spec_info.accept_lens,
            )
            if not prev_token_ids.is_cpu:
                prev_token_ids = prev_token_ids.cpu()
                prev_accept_lens = prev_accept_lens.cpu()
            # Worker-level staging: written here at draft prep, consumed by
            # _update_ngram_corpus after verify within the same forward call.
            self.prev_token_ids = prev_token_ids.tolist()
            self.prev_accept_lens = prev_accept_lens.tolist()
            assert bs == len(self.prev_accept_lens)
        else:
            # _update_ngram_corpus still reads the staging; fill it empty.
            self.prev_token_ids = []
            self.prev_accept_lens = [0] * bs

        self.ngram_corpus.synchronize()
        batch_tokens = []
        total_lens = []
        for i in range(bs):
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            check_token = self._efficient_concat_last_n(
                input_tails[i],
                output_tails[i] + prev_tokens,
                self.max_trie_depth,
            )
            batch_tokens.append(check_token)
            total_lens.append(base_lens[i] + len(prev_tokens))
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert total_draft_token_num == bs * self.draft_token_num, (
            f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        )
        return req_drafts, mask
````

`python/sglang/srt/speculative/ngram_worker.py` lines 402-426:

````python
    def _update_ngram_corpus(self, batch: ScheduleBatch):
        batch_tokens = []
        i, stride = 0, self.draft_token_num
        # Same splice condition as _prepare_draft_tokens: only overlap mode
        # has accepted tokens missing from req.output_ids.
        use_prev_tokens = self.enable_overlap and not batch.grammar_needs_sync()
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            put_ids = self._efficient_concat_last_n(
                list(req.origin_input_ids_unpadded[-self.max_trie_depth :]),
                list(req.output_ids[-self.max_trie_depth :]) + prev_tokens,
                self.max_trie_depth,
            )
            batch_tokens.append(put_ids)
            i += 1
        self.ngram_corpus.batch_put(batch_tokens)
````

`python/sglang/srt/speculative/ngram_info.py` lines 48-58:

````python
    @property
    def max_tree_depth(self) -> int:
        # NGRAM trees are node-budgeted with no depth cap: the corpus BFS only
        # stops on the node budget, so a single long match can chain all
        # draft_token_num nodes (spec_steps is meaningless for this tree).
        return self.draft_token_num

    @property
    def tree_topk(self) -> int:
        # Irregular tree: per-level branching follows the corpus matches.
        return -1
````

#### 9. One tree-verify round on the target: write mask to flashinfer, rebuild positions / first-child / next-sibling links from the mask, verify, compact the accepted nodes' KV

`python/sglang/srt/speculative/ngram_worker.py` lines 313-400:

````python
    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        # Decode-only: extend goes through the plain target forward, and an
        # IDLE batch must keep its forward_mode instead of being rewritten to
        # TARGET_VERIFY below (relevant once DP attention support lands).
        if not batch.forward_mode.is_decode():
            return

        bs = len(batch.reqs)

        retrieve_index = self.retrieve_indexes_batch[bs]
        retrieve_next_token = self.retrieve_next_token_batch[bs]
        retrieve_next_sibling = self.retrieve_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        # Pre-sync: only needs seq_lens_cpu, so build before the blocking
        # accept sync inside _prepare_draft_tokens.
        ones_masks = None
        if USE_FULL_MASK and not _is_cpu:
            ones_masks = [
                torch.ones(
                    (self.draft_token_num, batch.seq_lens_cpu[i]), device=self.device
                )
                for i in range(bs)
            ]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        # Staged for the grammar bitmask, derived after the verify launch below.
        self.grammar_tree_host = (mask, req_drafts) if batch.has_grammar else None

        # generate positions and some indices using tree_mask
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrieve_index,  # mutable
            retrieve_next_token,  # mutable
            retrieve_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK and not _is_cpu:
            tree_mask = []
            mask = mask.reshape(bs, self.draft_token_num, self.draft_token_num)
            # TODO(siyuan): the for loop here leads to significant overhead in large batch size. Can be written into a kernel.
            for i in range(bs):
                req_mask = torch.cat(
                    (
                        ones_masks[i],
                        torch.from_numpy(mask[i]).to(
                            device=self.device, non_blocking=True
                        ),
                    ),
                    dim=1,
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = draft_tokens
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + self.draft_token_num,
            batch_size=bs,
            draft_token_num=self.draft_token_num,
            device=self.device,
        )

        prepare_mamba_track_for_verify(batch)

        batch.spec_info = NgramVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=self.draft_token_num,
        )
````

`python/sglang/kernels/jit/include/sgl_kernel/speculative/ngram.cuh` lines 26-91:

````cpp
__global__ void reconstructIndicesFromTreeMask(
    bool* tree_mask,
    int64_t* verified_seq_len,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int batch_size,
    int draft_token_num) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid >= batch_size || tid >= draft_token_num) {
    return;
  }
  int base_offset = draft_token_num * draft_token_num;
  // token_idx: [bid * draft_token_num, (bid + 1) * draft_token_num)
  int token_idx = bid * draft_token_num;
  // tree_mask_idx: [bid * base_offset, (bid + 1) * base_offset)
  int tree_mask_offset = bid * base_offset;

  int depth = 0;
  int parent_idx = -1;

  for (int i = tid - 1, start_idx = tree_mask_offset + tid * draft_token_num; i >= 0; i--) {
    if (tree_mask[start_idx + i]) {
      depth++;
      if (parent_idx == -1) {
        parent_idx = i;
      }
    }
  }
  retrive_index[token_idx + tid] = token_idx + tid;
  positions[token_idx + tid] = depth + verified_seq_len[bid];

  int next_token_idx = -1;
  for (int i = tid + 1; i < draft_token_num; i++) {
    if (tree_mask[tree_mask_offset + i * draft_token_num + tid]) {
      next_token_idx = i;
      break;
    }
  }
  retrive_next_token[token_idx + tid] = next_token_idx;

  int next_sibling_idx = -1;
  if (parent_idx != -1) {
    for (int i = tid + 1; i < draft_token_num; i++) {
      int start_idx = tree_mask_offset + i * draft_token_num + parent_idx;
      if (tree_mask[start_idx]) {
        bool is_sibling = true;
        int end_idx = tree_mask_offset + i * draft_token_num + i;
        for (int j = start_idx + 1; j < end_idx; ++j) {
          if (tree_mask[j]) {
            is_sibling = false;
            break;
          }
        }
        if (is_sibling) {
          next_sibling_idx = i;
          break;
        }
      }
    }
  }
  retrive_next_sibling[token_idx + tid] = next_sibling_idx;
}
````

`python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh` lines 214-256:

````cpp
template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    IdType2* target_predict,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens) {
  uint32_t bx = blockIdx.x;

  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];

      if (draft_token_id == target_token_id) {
        // accept token
        predicts[last_accepted_retrive_idx] = target_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
}
````

`python/sglang/srt/speculative/ngram_worker.py` lines 512-521:

````python

            # The KV mover expects drafts-only counts. NGRAM's
            # accept_lens includes the bonus token, matching scheduler output.
            num_correct_drafts_per_req = accept_lens - 1
            move_accept_tokens_to_target_kvcache(
                batch,
                accept_index,
                num_correct_drafts_per_req,
                self.token_to_kv_pool_allocator,
            )
````

`python/sglang/srt/speculative/spec_utils.py` lines 791-852:

````python
def move_accept_tokens_to_target_kvcache(
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
):
    """
    Move accepted tokens (drafts + bonus) to the target KV cache.

    Args:
        batch: The batch to run.
        accept_index: The index of the accepted tokens (incl. bonus).
        num_correct_drafts: Per-req count of correct drafts (excludes bonus);
            seq_lens is advanced by ``num_correct_drafts + 1`` to cover the bonus slot.
    """
    bs = len(batch.seq_lens)
    device = batch.seq_lens.device
    # accept_index element count, NOT bs * num_draft_tokens: for topk > 1 the
    # tree exceeds the accepted chain, over-reading accept_index (illegal memory).
    size = bs * accept_index.shape[1]

    # fill_accept_out_cache_loc reads out_cache_loc[accept_index]; -1 sentinel ok.
    maybe_detect_oob(
        accept_index,
        -1,
        batch.out_cache_loc.size(0),
        "spec v2 move_accept_tokens accept_index",
    )

    tgt_cache_loc = torch.zeros(
        size,
        dtype=torch.int64,
        device=device,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    if _is_cpu:
        assign_extend_cache_locs_cpu(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
        )
    else:
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
    fill_accept_out_cache_loc_func(
        accept_index,
        batch.out_cache_loc,
        accept_out_cache_loc,
        size,
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
        tgt_cache_loc, accept_out_cache_loc
````

## Relevance
This is the one mainstream implementation that already does what the multiple-occurrence idea asks for, with a model-free CPU drafter: all suffix lengths that exist in the trie are expanded into ONE tree under a hard node budget (`speculative_num_draft_tokens`, default 12; docs example 16), the tree is flattened in BFS order with parent-row-copied mask (`result.cpp` lines 40-48: `memcpy(&info.mask[i * n], &info.mask[prevs[i] * n], prevs[i] + 1)`), and the target verifies the whole tree in one forward with that mask. Four design facts are directly reusable: (a) the node budget, not a depth cap, bounds the tree ("a single long match can chain all draft_token_num nodes" — a pure chain is the degenerate tree, so one code path serves both regimes); (b) the BFS breadth schedule gives LONG matches narrow breadth and SHORT matches wide breadth, i.e. width is spent where the match is least trustworthy; (c) greedy verify is a 25-line walk over first-child / next-sibling links and needs only the target's argmax per node, so it is host-side work; (d) after verify the KV of the accepted nodes is moved to the contiguous slots after the committed prefix (`move_accept_tokens_to_target_kvcache` -> `move_kv_cache(tgt_cache_loc, accept_out_cache_loc)`) and the rest is simply abandoned, which is the whole "rollback". Two caveats: SGLang inserts only the last `max_trie_depth` (18) tokens of prompt+output per step (`_update_ngram_corpus`; the FIXME says inserting the whole prompt "after testing, there is not much difference") so it is a self-history drafter, not a prompt-scanner — for agentic coding where the copy source is the prompt you would seed the trie (or `external_sam_budget`) with the full context; and the mask is handed to flashinfer as `custom_mask`, which the Ascend `npu_fused_infer_attention_score` verify path does not accept in this form (see the vllm-ascend entry).

---

### REST — retrieval from a datastore builds the Trie directly from MULTIPLE occurrences of the matched suffix, each with its own continuation; mean generated length tops out at 2.65 even on code

https://arxiv.org/abs/2311.08252 · https://arxiv.org/html/2311.08252 (NAACL 2024 camera-ready, v2, revised 4 Apr 2024)
Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D. Lee, Di He (equal contribution; correspondence Di He <dihe@pku.edu.cn>). Code: https://github.com/FasterDecoding/REST

Abstract, verbatim:

> We introduce Retrieval-Based Speculative Decoding (REST), a novel algorithm designed to speed up language model generation. The key insight driving the development of REST is the observation that the process of text generation often includes certain common phases and patterns. Unlike previous methods that rely on a draft language model for speculative decoding, REST harnesses the power of retrieval to generate draft tokens. This method draws from the reservoir of existing knowledge, retrieving and employing relevant tokens based on the current context. Its plug-and-play nature allows for seamless integration and acceleration of any language model, all without necessitating additional training. When benchmarked on 7B and 13B language models in a single-batch setting, REST achieves a significant speedup of 1.62× to 2.36× on code or text generation. The source code of REST is available at https://github.com/FasterDecoding/REST.

The mechanism, verbatim — this is the exact "multiple occurrences, each with a different continuation, verified together" construction:

> To use REST, the first step is constructing the datastore. In this paper, we leverage either the pretraining data corpus or the instruction-tuning data corpus to build our datastore, which serves as the source for the draft token sequence. During each inference step, we first use previous tokens (pre-generated tokens or prompt tokens) as queries to identify exact matches in the datastore. The subsequent tokens from these exact matches are considered generation candidates. A Trie is constructed using these candidates. The nodes with the highest frequencies are selected as the draft tokens. This sequence then undergoes verification by the LLM through a single forward pass, aided by a meticulously designed attention mask known as tree attention Cai et al. (2023); Miao et al. (2023); Spector and Re (2023). As many subsequences during generation likely appear in the datastore, REST can frequently generate multiple correct tokens per step.

> REST operates based on a pre-built datastore D = {(c_(i), t_(i))}, where c_(i) represents a context and t_(i) represents the corresponding continuation of the context c_(i). Given a text/code corpus, we construct the datastore D using the prefix context and the corresponding continuation at each position.

> At inference, given a context s = (x₁, …, x_(t)), our objective is to construct the draft tokens which are likely the continuations of s. Different from vanilla speculative decoding that uses a small LM to construct the draft, we leverage the built datastore D and directly retrieve draft tokens from the datastore. We first use the context s to retrieve context-continuation pairs from the datastore D and construct a set of continuation candidates S:

The greedy suffix-length search and its exact cutoff, verbatim (note the explicit tuned constant):

> Our retrieval process is shown in Algorithm 1. We aim to find contexts in D that match the longest suffix of s. We employ a greedy strategy and start from a pre-defined match length upper limit n_(max)^(\par)^(\par) We set n_(max) as 16 in our experiments, as only few cases lead to a maximum match with more than 16 tokens.. For each suffix length n, we obtain the context s’s suffix with n tokens q (line 5), and obtain all the contexts c_(i) that match q as a suffix (line 6). If at least one context in D matches the current q (i.e., S ≠ ∅), we return the corresponding context-continuation pairs as the retrieval result; otherwise we decrease the matching length n by one and try to match a shorter suffix (line 7). We use a suffix array Manber and Myers (1993) to implement efficient exact match in datastore D for a given q. The retrieval process leads to negligible overhead ( < 6%) in our experiments (see details in Section 5).

Draft construction from the retrieved set — a Trie keyed on frequency, not on a single longest match:

> The retrieved result S includes possible continuations of the context s. For each t_(i) ∈ S, any prefix of t_(i) can serve as draft tokens of s in the speculative decoding and be further verified by the LLM. Note that the retrieved set of continuation candidates S can be large. It is not feasible to use all candidates as draft tokens and feed them into the LLM for verification. Here we present how we select high-quality draft tokens from the retrieved set S. A naive strategy is to sample a subset of sequences in S as the draft tokens. However, this is suboptimal as the sampling contains randomness when S is large.

> We select draft tokens from the retrieved result S using a Trie. In the Trie, the unique path from a node to the root node corresponds to a prefix of t_(i) ∈ S. For each node, we assign a weight reflecting the number (frequency) of the corresponding prefix that appears in the retrieved candidates. As shown in Algorithm 2, we first construct a Trie using all sequences in S, and the node weight is updated when a candidate t_(i) is inserted into the Trie (lines 2-7). The Trie data structure allows us to prioritize tokens using the weights and select high-frequency prefixes (lines 8-15). In the practical implementation, we choose a subtree that contains the top c nodes with the highest weights, which equals to selecting the top c high-frequency prefixes as the draft sequences.

Why a tree, not independent sequences — the exact reason, verbatim:

> In REST, multiple draft sequences may be retrieved from the datastore. While one might initially approach the drafts independently and feed them into the LLM as distinct sequences in a batch, practical observations reveal that many drafts share common prefixes. This leads to redundant computation of Transformer layers on these shared prefixes across different sequences, resulting in a waste of computational power. To optimize the efficiency, we construct a pseudo sequence from the subtree using breadth-first search. By definition, it can be immediately obtained that each draft constitutes a sub-sequence of this pseudo sequence, and any shared prefix appears only once. To correctly execute LLM on this pseudo sequence, we implement a carefully designed attention mask in each attention layer, ensuring that the computation of each token precisely reflects its dependencies in the original draft sequence. This attention strategy is also known as tree attention (Cai et al., 2023; Miao et al., 2023; Spector and Re, 2023).

The exact hyperparameters used in their main experiments, verbatim (note the explicit cross-reference to Medusa's budget):

> When performing exact match in the datastore, the starting context suffix length, n_(max), is set to 16, and is progressively reduced by one until we find matching contexts in the datastore. The length of each retrieved continuation candidate denoted as m, is truncated to 10. Empirical results from Medusa (Cai et al., 2023) suggest 64 draft tokens to be an optimal computation configuration. Hence, we limit the maximum number of selected draft tokens in the constructed Trie to 64, designated as c.

> Note that the Mean Generated Length (M) acts as the upper limit of the speedup that REST can achieve, ignoring the overhead for retrieving and constructing draft tokens.

Datastore construction, verbatim:

> For CodeLlama, we construct a datastore using a portion of the Python pretraining code from The Stack (Kocetkov et al., 2022). This dataset comprises approximately 2.7M Python code samples and results in a datastore with a size of 27GB. On the other hand, for Vicuna, we construct a datastore using data derived from UltraChat Ding et al. (2023). This dataset consists of around 774K conversations from ChatGPT^(\par)^(\par) https://chat.openai.com/, yielding a datastore with a size of 12GB.

The HumanEval result table (CodeLlama 7B/13B, greedy and nucleus), reproduced verbatim from the rendered paper:

`arxiv.org/html/2311.08252 (rendered table, HumanEval)` lines 239-261:

````text
+---------------------+---------------+--------------------------+--------------------+------------+
| Benchmark           | Model         | Method                   | Mean Token Time(↓) | Speedup(↑) |
+---------------------+---------------+--------------------------+--------------------+------------+
| HumanEval (1 shot)  | CodeLlama 7B  | Autoregressive (Greedy)  | 27.89 ms/token     | 1×         |
|                     +---------------+--------------------------+--------------------+------------+
|                     | CodeLlama 7B  | Speculative (Greedy)     | 15.90 ms/token     | 1.75×      |
|                     +---------------+--------------------------+--------------------+------------+
|                     | CodeLlama 7B  | REST (Greedy)            | 11.82 ms/token     | 2.36×      |
|                     +---------------+--------------------------+--------------------+------------+
|                     | CodeLlama 13B | Autoregressive (Greedy)  | 44.32 ms/token     | 1×         |
|                     +---------------+--------------------------+--------------------+------------+
|                     | CodeLlama 13B | Speculative (Greedy)     | 19.39 ms/token     | 2.29×      |
|                     +---------------+--------------------------+--------------------+------------+
|                     | CodeLlama 13B | REST (Greedy)            | 19.53 ms/token     | 2.27×      |
+---------------------+---------------+--------------------------+--------------------+------------+
| HumanEval (10 shot) | CodeLlama 7B  | Autoregressive (Nucleus) | 27.99 ms/token     | 1×         |
|                     +---------------+--------------------------+--------------------+------------+
|                     | CodeLlama 7B  | Speculative (Nucleus)    | 18.83 ms/token     | 1.49×      |
|                     +---------------+--------------------------+--------------------+------------+
|                     | CodeLlama 7B  | REST (Nucleus)           | 13.18 ms/token     | 2.12×      |
|                     +---------------+--------------------------+--------------------+------------+
|                     | CodeLlama 13B | Autoregressive (Nucleus) | 44.46 ms/token     | 1×         |
|                     +---------------+--------------------------+--------------------+------------+
````

The headline comparison, verbatim:

> Regarding generation speed, REST demonstrates a significant speed enhancement compared to standard autoregressive decoding and speculative decoding, achieving 2.16× to 2.36× increase for CodeLlama in the HumanEval benchmark. The MT-Bench benchmark also reveals a speedup for Vicuna when using our method, with a factor ranging from 1.62× to 1.77×. These empirical results lend weight to the effectiveness of our method for speeding up the generation process of LLMs. Note that the speedup of nucleus sampling is not as good as that of greedy sampling. We speculate that this drop in performance is caused by the randomness introduced by nucleus sampling. Since speculative decoding may achieve better results with a more powerful draft LM that aligns with the LLM, we do not claim that REST can outperform speculative decoding under all circumstances. Yet, REST undoubtedly provides a potent and straightforward approach for faster inference of LLMs.

> Another intriguing observation that emerges from these results is the domain-dependent nature of the speed improvements. This characteristic has also been noted in other methods like speculative decoding Chen et al. (2023) and Medusa Cai et al. (2023). Specifically, the speedup achieved with REST is significantly greater in the HumanEval benchmark than in the MT-Bench benchmark, suggesting that the effectiveness of REST may vary depending on the specific domain.

Datastore-size ablation — Mean Generated Length (M, i.e. mean accepted length) as a function of corpus size, verbatim table and the accompanying text:

`arxiv.org/html/2311.08252 (rendered table, datastore-size ablation)` lines 332-340:

````text

  ------------------ ---------------- ---------------- ------ -------------------- ------------
  Method             Datastore Size   Retrieval Time   M      Mean Token Time(↓)   Speedup(↑)
  Baseline(Greedy)   -                -                1      27.89 ms/token       1×
  REST(Greedy)       0.9 GB           0.2 ms           1.96   15.28 ms/token       1.83×
  REST(Greedy)       4.4 GB           0.5 ms           2.18   13.98 ms/token       1.99×
  REST(Greedy)       8.7 GB           0.6 ms           2.35   13.24 ms/token       2.11×
  REST(Greedy)       14 GB            0.6 ms           2.45   12.99 ms/token       2.15×
  REST(Greedy)       27 GB            0.7 ms           2.65   11.82 ms/token       2.36×
````

> Increasing the size of the datastore is an effective strategy for enhancing the accuracy of retrieved draft tokens in the Trie, which in turn can significantly boost generation speed. In Table 2, we show that as the datastore size increases, both the Mean Generated Length and Mean Token Time correspondingly improve. However, it’s important to note that the speedup growth is not as pronounced as that of the Mean Generated Length. This discrepancy could be attributed to the overhead of getting draft tokens. We assume that in industry applications, there will be ample disk storage to build a large datastore and ample CPU cores for fast retrieval. We also visualize the trend of scaling the retrieval datastore size in Figure 2. From this, we can infer that there is still potential to achieve even faster speeds with a larger datastore.

Trie-selection vs random-selection ablation (same total draft-token budget, 64, in both arms):

`arxiv.org/html/2311.08252 (rendered table, Trie vs Random selection)` lines 349-355:

````text
  ------------------- ------ ---------------------
  Selecting Methods   M(↑)   Mean Token Time (↓)
  Random(Greedy)      2.51   12.80
  Trie(Greedy)        2.65   11.82
  Random(Nucleus)     2.44   14.19
  Trie(Nucleus)       2.57   13.18
  ------------------- ------ ---------------------
````

> We compare selecting draft tokens in the Trie with randomly sampling retrieved continuation candidates as draft tokens. For an equitable comparison, we employ a random sampling technique to sample at most eight sequences from all the retrieved candidates. Furthermore, each sequence is truncated to a maximum length of 8. This results in a maximum number of 64 draft tokens, corresponding to the maximum number of selected draft tokens from the Trie. The data presented in Table 3 indicates that selecting draft tokens from the Trie, as opposed to employing a random sampling approach, enhances the performance.

Retrieval/Trie-construction overhead, verbatim:

> Additionally, it is important to note that the average time (divided by the total number of tokens) required for retrieval (which includes the time taken to construct the Trie) is less than 1 ms. This time is very small and can, for all practical purposes, be considered negligible. This negligible retrieval time further underscores the efficiency of REST.

## Relevance
REST is the primary source the task asked for by name, and it is the cleanest existing example of "the same suffix occurs several times earlier in the context, each occurrence proposing a different continuation, verified together": the retrieved set S can contain many distinct continuations of the same matched suffix, a Trie is built over all of them with per-node frequency weights, and the top-c highest-weight nodes (c=64, chosen to match Medusa's own budget) are linearized by BFS into one pseudo-sequence verified in a single tree-attention forward pass. Three numbers matter for comparison with our 7.1: (1) the datastore-size ablation shows Mean Generated Length M saturating at only **2.65** even with a 27 GB / 2.7M-sample Python datastore and a 64-node budget — this is the clearest published curve of accepted-length-as-a-function-of-corpus-size for a multi-match trie, and it never gets remotely close to 7.1; (2) M=2.65 is on HumanEval, i.e. fresh code generation from a docstring, not code-editing — REST's own ablation (Trie 2.65 vs Random-sampling-from-the-same-candidates 2.51) shows the Trie-vs-chain gain from exploiting multiple occurrences is real but small, roughly +5.6%, which is a second, independent, code-domain data point for "how much does a multi-match tree buy over one chain at matched budget" (compare SGLang's own two policies in Section 2 and the suffix-automaton papers); (3) retrieval and Trie-construction cost is "less than 1 ms" and the mask is a device-side tree-attention mask, exactly the shape vllm-ascend and sgl-kernel-npu need. The huge gap between REST's 2.65 ceiling and our 7.1, or SuffixDecoding's 7.8 on SWE-Bench, is domain, not mechanism: REST is tested on HumanEval (write new code from a docstring, near-zero prompt/output overlap) while our workload and SuffixDecoding's SWE-Bench are agentic coding where the output IS mostly the input. See the ISSTA 2026 and vLLM-production-study entries below for figures that isolate this domain effect precisely.

---

### "An Empirical Study of Speculative Decoding on Software Engineering Tasks" (ISSTA 2026) — PLD and Suffix Decoding head-to-head on SWE-bench/Aider/CanItEdit/Defects4J, four model scales, and the direct warning that agentic acceptance can be inflated noise

https://arxiv.org/abs/2604.26469 · https://arxiv.org/html/2604.26469 (arXiv:2604.26469v4 [cs.SE], first submitted 29 April 2026, revised 16 August 2026)
Yijia Li (Zhejiang University), Junkai Chen (Singapore Management University), Xing Hu (Zhejiang University, corresponding author), Xin Xia (Zhejiang University / Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security). PACMSE Vol. 3, ISSTA, article 05410, DOI 10.1145/3832145.

Abstract, verbatim:

> Large Language Models (LLMs) have become widely used for Software Engineering (SE) tasks, spanning from function-level code generation to complex repository-level workflows. However, the high latency of autoregressive inference remains a significant bottleneck, hindering their deployment in interactive environments. While Speculative Decoding (SD) offers a promising technique for lossless acceleration, prior research on long-context repository-level tasks and complex agentic interactions remains limited. To bridge this gap, we present a systematic empirical study to evaluate the effectiveness of SD in SE tasks. We benchmark a representative spectrum of strategies, encompassing both model-based and model-free methods, across generation, editing, and repair scenarios. Our empirical results show that SD provides clear acceleration potential for SE tasks, but its realized benefits jointly vary with model architecture and task scenario. Specifically, model-based approaches are well-suited for code generation, whereas model-free methods are better adapted to repository-level repair and editing scenarios. We further observe that the repetitiveness of SE tasks improves the performance of model-free methods, while complex agentic workflows can introduce repetitive failure modes that skew acceleration measurements. In contrast to natural language tasks, the higher predictability of SE tasks allows for more aggressive hyperparameter settings. Our findings provide practical guidance for selecting, configuring, and evaluating SD methods in SE scenarios.

PLD's exact formulation as this paper writes it, and its named weakness, verbatim:

> Prompt Lookup Decoding (PLD) (Saxena, 2023).    As illustrated in the upper-left corner of Figure 1, PLD implements a deterministic N-gram retrieval strategy over the entire context. The method operates by matching the current context suffix with previous occurrences in the input sequence. As shown in the figure, the input sequence ends with the suffix "B C". The algorithm identifies a prior occurrence of this pattern and observes that it is followed by the token "D". Consequently, PLD retrieves "D" as the draft token. Formally, if a suffix x_(t − N : t) (where N can vary up to a maximum N_(max)) appeared previously at position k, the method assumes the sequence following that historical occurrence is a likely candidate. Upon a successful match (visualized as the dotted arrow in the figure), PLD copies a fixed number of subsequent tokens as the speculation:

> PLD is primarily constrained by its static speculation budget K. This fixed budget fails to adapt to the varying confidence levels of the retrieved context; it may either be too aggressive in uncertain scenarios, leading to frequent verification rollbacks, or too conservative when the historical pattern is highly repetitive and predictable.

Suffix Decoding's formulation as this paper writes it (independent restatement of Oliaro et al.'s mechanism, confirming `γ = ⌊α · L_match⌋` is the same adaptive rule reported in Section 1 above):

> Suffix Decoding (Oliaro et al., ).    To overcome the fixed budget limitation of PLD, as visualized in the lower-left of Figure 1, Suffix Decoding utilizes Suffix Trees for probabilistic structural retrieval. As depicted in the schematic, the method maintains two distinct data structures: a Global Tree that indexes historical outputs from prior requests, and a Local Tree that tracks the current ongoing generation. The core mechanism involves querying these trees to retrieve frequent patterns. Nodes in the trees encode the occurrence frequency Count(s) of sub-sequences. The “Prob.” block in the figure represents the scoring phase, where the conditional probability of a next token v given a suffix context s is estimated via:

`arxiv.org/html/2604.26469 (equation 3)` lines 190-192:

````text
  ----- -- ----------------------------------------------------------------------------- --
  (3)      $$P_{\text{suffix}}(v|s)=\frac{\text{Count}(s\oplus v)}{\text{Count}(s)},$$   
  ----- -- ----------------------------------------------------------------------------- --
````

> where ⊕ denotes the concatenation. Based on these probabilities, the method performs a greedy expansion to construct a Speculation Tree (shown as the tree-structured “Draft” in the figure). Moreover, it employs an Adaptive Speculation Strategy: the speculation budget γ (the total token count in the draft tree) is dynamically scaled proportional to the pattern match length L_(match) (γ = ⌊α ⋅ L_(match)⌋). This allows the method to perform “extreme speculation” by generating deep draft trees when historical patterns are strong, while falling back to conservative drafting for novel logic.

> This approach is specifically tailored for agentic applications, particularly automated coding assistants and multi-agent workflows, where workloads exhibit high repetitiveness (Oliaro et al., ).

The three metrics this paper standardizes on, verbatim — note `τ` ("Mean Acceptance Length") is exactly our "accepted tokens per round":

> Following previous research (Li et al., 2025b; Oliaro et al., ; Chen et al., 2023; Saxena, 2023), we employ three key metrics to quantify the acceleration performance of SD. For all metrics listed below, higher values indicate better performance.
>
> - •
>   Speedup is defined as the ratio of the generation throughput (i.e., the number of tokens produced per second) of the speculative method to that of the autoregressive baseline.
> - •
>   Mean Acceptance Length (τ) is the expected number of candidate tokens validated by the target model in a single verification step. This metric serves as a direct proxy for the drafting model’s quality and the semantic alignment between the drafter and the verifier.
> - •
>   Acceptance Rate (n − α) denotes the acceptance rate α at a specific position index n within the draft chain. It is calculated as the conditional probability that the draft token at position n is accepted, given that all preceding tokens (0…n − 1) have been verified as correct. For example, we calculate 3 − α by dividing the number of tokens accepted at position 3 by the number of tokens accepted at position 2.

Configuration used for every method, verbatim — PLD here is lookup window N=4, draft length K=5; Suffix Decoding is K=32 ("the authors' recommended default settings"); note this is a MUCH narrower PLD draft width than our 32:

> SD Approaches.    To ensure reproducibility and prioritize precision in code generation, we employ greedy decoding across all experiments. For detailed configurations of different methods, we conduct the experiments following their common practice: (i) For PLD, it is configured with a lookup window size N = 4 and a draft length K = 5, adhering to the default recommendations provided by the vLLM framework. (ii) For Suffix Decoding, we use a maximum draft length of K = 32, adhering to the authors’ recommended default settings (Oliaro et al., ). (iii) For Eagle-3, the speculative length is fixed at 3. We use official RedHatAI²² 2 https://huggingface.co/RedHatAI checkpoints for 8B/30B/32B models; for the 70B model, due to the unavailability of pre-trained drafter weights, we train a custom drafter via SpecForge (Li et al., 2025a) on BAAI/TACO (Li et al., 2023) and SWE-Gym/SWE-Gym (Pan et al., ) with a learning rate of 5e-5. (iv) For MLP Speculator, we employ the standard implementation with default hyperparameters, utilizing the pre-trained draft models provided by IBM AI Platform³³ 3 https://huggingface.co/ibm-ai-platform. Note that this baseline is evaluated on vLLM v0.9.2 due to specific version compatibility constraints, which restrict it to single-threaded execution.

Table 1 — Speedup and τ (Mean Acceptance Length) for PLD, Suffix Decoding, Eagle-3 and MLP Speculators, across four target models and six SE benchmarks, reproduced verbatim from the rendered paper:

`arxiv.org/html/2604.26469 (rendered Table 1)` lines 280-310:

````text
+------------+-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
| Model      | Method          | LiveCodeBench     | HumanEval       | SWE-bench       | Defects4J       | Aider Polyglot    | CanItEdit       | Mean           |
|            |                 +---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            |                 | Speedup | τ       | Speedup | τ     | Speedup | τ     | Speedup | τ     | Speedup | τ       | Speedup | τ     | Speedup | τ    |
+------------+-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
| Qwen3-32B  | PLD             | 1.13×   | 2.89    | 1.17×   | 2.89  | 1.05×   | 2.92  | 1.87×   | 3.40  | 1.08×   | 3.19    | 2.23×   | 4.07  | 1.42×   | 3.23 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | Suffix Decoding | 1.27×   | 1.21    | 1.56×   | 1.10  | 1.66×   | 3.48  | 2.61×   | 3.33  | 1.22×   | 1.35    | 3.05×   | 4.11  | 1.90×   | 2.43 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | Eagle-3         | 1.71×   | 1.87    | 1.94×   | 1.99  | 1.10×   | 1.87  | 1.21×   | 2.10  | 1.29×   | 1.81    | 2.14×   | 2.35  | 1.57×   | 2.00 |
+------------+-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
| Qwen3-30A3 | PLD             | 0.87×   | 2.21    | 0.99×   | 2.57  | 1.02×   | 3.25  | 1.36×   | 3.38  | 1.08×   | 3.29    | 1.55×   | 3.83  | 1.15×   | 3.09 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | Suffix Decoding | 0.95×   | 1.02    | 1.03×   | 0.91  | 1.73×   | 4.78  | 1.53×   | 3.14  | 0.99×   | 1.41    | 1.83×   | 2.93  | 1.34×   | 2.37 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | Eagle-3         | 0.77×   | 1.75    | 0.54×   | 1.93  | 0.41×   | 0.43  | 0.55×   | 1.57  | 1.15×   | 1.52    | 0.87×   | 2.22  | 0.72×   | 1.57 |
+------------+-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
| L31-8B     | PLD             | 1.07×   | 2.36    | 1.16×   | 2.81  | 1.71×   | 2.99  | 1.78×   | 3.23  | 1.30×   | 3.36    | 2.05×   | 4.01  | 1.51×   | 3.13 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | Suffix Decoding | 1.58×   | 1.30    | 1.63×   | 1.13  | 3.35×   | 5.48  | 2.74×   | 3.18  | 1.82×   | 1.99    | 2.90×   | 3.54  | 2.34×   | 2.77 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | Eagle-3         | 1.77×   | 1.82    | 2.19×   | 2.11  | 0.70×   | 0.64  | 1.55×   | 1.56  | 1.32×   | 1.49    | 2.12×   | 2.18  | 1.61×   | 1.63 |
+------------+-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
| L31-70B    | PLD             | 0.97×   | 2.39    | 0.99×   | 2.40  | 1.02×   | 2.69  | 1.67×   | 3.77  | 1.12×   | 3.40    | 1.71×   | 4.05  | 1.25×   | 3.12 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | Suffix Decoding | 1.24×   | 1.08    | 1.13×   | 1.00  | 1.79×   | 3.39  | 1.70×   | 3.55  | 1.50×   | 2.72    | 1.99×   | 3.93  | 1.56×   | 2.61 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | Eagle-3         | 1.66×   | 1.74    | 1.41×   | 1.27  | 0.87×   | 1.07  | 0.83×   | 0.35  | 0.88×   | 0.48    | 1.29×   | 1.09  | 1.16×   | 1.00 |
|            +-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|            | MLP Speculators | 0.81×   | —       | 1.22×   | —     | 0.91×   | —     | 0.88×   | —     | 1.23×   | —       | 1.03×   | —     | 1.01×   | —    |
+------------+-----------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
````

The domain split, verbatim — model-free methods win on repair/editing, model-based wins on fresh generation:

> For code generation, represented by LiveCodeBench and HumanEval, Eagle-3 demonstrates superior performance. As shown in Table 1, Eagle-3 achieves an average speedup of 1.50× across these two benchmarks, outperforming Suffix Decoding (1.30×) and PLD (1.04×). We attribute this advantage to the characteristic of the task, which involves translating natural language into code; since the ground-truth code does not appear in the context history, model-free methods fail to find sufficient repetitive patterns to copy. As shown in Table 1, Suffix Decoding achieves a mean τ of 1.09 on code generation benchmarks, compared to 3.79 on code repair benchmarks and 2.75 on code editing benchmarks, respectively. PLD exhibits a similar trend, recording a mean τ of 2.57 on code generation benchmarks, which is consistently lower than the 3.20 and 3.65 observed on code repair and code editing benchmarks.

> In contrast, for code repair and code editing tasks, Suffix Decoding demonstrates distinct superiority. Across SWE-bench and Defects4J, Suffix Decoding achieves an average speedup of 2.14×, compared with 1.44× for PLD and 0.90× for Eagle-3. Across Aider Polyglot and CanItEdit, Suffix Decoding reaches 1.91×, compared with 1.52× for PLD and 1.38× for Eagle-3. We attribute this dominance to the higher contextual redundancy in repair and editing scenarios, which is especially pronounced in repository-level agentic workflows such as SWE-bench and also appears when models modify code around existing context. As suggested by Oliaro et al. (Oliaro et al., ), such redundancy inherently favors Suffix Decoding, which retrieves explicit repetitive patterns rather than approximating distributions.

Diminishing returns in long context — τ keeps climbing but speedup does not, verbatim (this is a second, independent confirmation of the mechanism already noted in Section 3 above, now with exact numbers and a request-length measurement):

> Diminishing Returns in Long-Context Tasks.    In SWE-bench, we observe a phenomenon of diminishing returns. Specifically, the increase in speedup slows down even as the mean acceptance length (τ) continues to grow. We demonstrate this by comparing representative macro-average values of speedup and τ on LiveCodeBench and SWE-bench, which correspond to code generation and repository-level repair tasks, respectively. For Suffix Decoding, as shown in Table 1, although high redundancy in the repository causes τ to increase from 1.15 to 4.28, the speedup only increases from 1.26× to 2.13×. Similarly, for Eagle-3 and PLD, we observe that the speedup decreases even when the mean acceptance length (τ) remains stable. Taking Qwen3-32B as an example, Eagle-3 maintains an identical τ of 1.87 on both benchmarks. However, the speedup drops from 1.71× to 1.10×. In the same way, for PLD, while τ stays at a high level (2.89 on LiveCodeBench and 2.92 on SWE-bench), the speedup declines from 1.13× to 1.05×. This indicates that draft quality (implied by τ) is not the only factor for acceleration in complex tasks.

> We attribute this phenomenon to the extensive context windows of repository-level tasks. According to Yang et al. (Yang et al., 2025), in long-context scenarios, the inefficiencies in attention mechanisms and massive KV cache demands significantly impede verification efficiency, offsetting potential speedup offered by the high acceptance rates. Our analysis of L31-8B request logs reveals that SWE-bench tasks average 15,999 total tokens per request, substantially higher than the 817 tokens for LiveCodeBench. Here, total tokens include both prompt and generated tokens, and the average is computed over assistant model requests. This implies that in repository-level tasks, the verification process transitions into a compute-bound regime, ultimately limiting the realizable speedup.

Agentic infinite loops as a confound on acceptance-length measurements — the exact warning that should be read against any very high τ or very high acceptance number, including ours:

> Evaluation Noise from Agentic Infinite Loops.    In addition to model scale and architecture, agentic infinite loops can introduce additional noise into SD acceleration measurements. In SWE-bench, we observe that some models enter repetitive action loops, producing nearly identical Thought and Action sequences over multiple turns. Such loops create artificial context redundancy, which can be exploited by model-free SD methods.

> The occurrence of “Infinite Loops” represents a widespread phenomenon rather than an isolated anomaly in current LLM-based agents. To investigate this further, we conducted a quantitative analysis of the execution trajectories within the SWE-bench Verified dataset. We define an infinite loop as a trajectory in which the agent produces the same output block at least five times while solving a single instance. As illustrated in Figure 3(b), the L31-8B model encounters infinite loops in 226 distinct tasks. In a similar vein, the Qwen3-32B model demonstrates this pattern in 210 tasks. Qwen3-30A3 also exhibits this behavior in 167 tasks. Conversely, L31-70B exhibits a notably lower frequency, with only 89 tasks affected by such repetitive generations. These quantitative findings suggest that repetitive agentic behavior varies substantially across target models and can become a major confounding factor in SD evaluation.

Table 6 — task performance score vs. mean acceptance length (τ) for PLD / Suffix / Eagle-3, reproduced verbatim from the rendered paper, and the paper's own reading of it:

`arxiv.org/html/2604.26469 (rendered Table 6)` lines 480-533:

````text
+---------------------------------------+------------+-------------------+--------------------------------+
| Benchmark                             | Model      | Performance Score | Mean Accept Length (τ)         |
+---------------------------------------+            |                   +----------+----------+----------+
| Metrics                               |            |                   | PLD      | Suffix   | Eagle-3  |
+=======================================+============+===================+==========+==========+==========+
| LiveCodeBench Pass@1 (%)              | Qwen3-32B  | 46.1              | 2.89     | 1.21     | 1.87     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | Qwen3-30A3 | 56.0              | 2.21     | 1.02     | 1.75     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-8B     | 17.0              | 2.36     | 1.30     | 1.82     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-70B    | 30.5              | 2.39     | 1.08     | 1.74     |
+---------------------------------------+------------+-------------------+----------+----------+----------+
| HumanEval Pass@1 (%)                  | Qwen3-32B  | 83.5              | 2.89     | 1.10     | 1.99     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | Qwen3-30A3 | 96.3              | 2.57     | 0.91     | 1.93     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-8B     | 68.9              | 2.81     | 1.13     | 2.11     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-70B    | 72.6              | 2.40     | 1.00     | 1.27     |
+---------------------------------------+------------+-------------------+----------+----------+----------+
| SWE-bench Resolved (%)                | Qwen3-32B  | 12.4              | 2.92     | 3.48     | 1.87     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | Qwen3-30A3 | 10.4              | 3.25     | 4.78     | 0.43     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-8B     | 1.4               | 2.99     | 5.48     | 0.64     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-70B    | 7.2               | 2.69     | 3.39     | 1.07     |
+---------------------------------------+------------+-------------------+----------+----------+----------+
| Defects4J Correct (%)                 | Qwen3-32B  | 18.54             | 3.40     | 3.33     | 2.10     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | Qwen3-30A3 | 18.08             | 3.38     | 3.14     | 1.57     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-8B     | 7.09              | 3.23     | 3.18     | 1.56     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-70B    | 19.68             | 3.77     | 3.55     | 0.35     |
+---------------------------------------+------------+-------------------+----------+----------+----------+
| Aider Polyglot Pass Rate 1 & 2 (%)    | Qwen3-32B  | 18.2 & 41.3       | 3.19     | 1.35     | 1.81     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | Qwen3-30A3 | 11.6 & 25.8       | 3.29     | 1.41     | 1.52     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-8B     | 1.3 & 2.7         | 3.36     | 1.99     | 1.49     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-70B    | 1.8 & 7.1         | 3.40     | 2.72     | 0.48     |
+---------------------------------------+------------+-------------------+----------+----------+----------+
| CanItEdit Pass@1 (%)                  | Qwen3-32B  | 58.1              | 4.07     | 4.11     | 2.35     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | Qwen3-30A3 | 54.8              | 3.83     | 2.93     | 2.22     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-8B     | 43.3              | 4.01     | 3.54     | 2.18     |
|                                       +------------+-------------------+----------+----------+----------+
|                                       | L31-70B    | 53.3              | 4.05     | 3.93     | 1.09     |
+---------------------------------------+------------+-------------------+----------+----------+----------+

````

> Table 6 suggests that, in agentic tasks, lower task performance can coincide with higher acceptance lengths for model-free methods. This pattern is particularly clear on Aider Polyglot: Qwen3-32B achieves the best task score (18.2% and 41.3%) but has lower PLD and Suffix acceptance lengths (τ = 3.19 and τ = 1.35), while L31-70B obtains much lower task scores (1.8% and 7.1%) but higher PLD and Suffix acceptance lengths (τ = 3.40 and τ = 2.72). On SWE-bench, L31-8B has the lowest problem-solving capability (1.4% Resolved) but reaches the highest Suffix Decoding acceptance length (τ = 5.48). Similarly, Qwen3-30A3 achieves a lower task score than Qwen3-32B (10.4% vs. 12.4%) but obtains a higher PLD acceptance length (τ = 3.25 vs. 2.92). These examples suggest that model-free acceptance statistics can be inflated by repetitive or degenerate agentic traces, rather than reflecting meaningful task progress alone.

Cross-domain comparison against MT-bench (natural language) — acceptance length collapses for model-free methods outside code:

> The results show that acceptance lengths drop markedly on MT-bench for model-free methods than for Eagle-3. Across the four models, PLD’s average τ decreases from 3.14 on SE tasks to 2.00 on MT-bench, while Suffix Decoding decreases more sharply from 2.55 to 0.52. Such short acceptance lengths make it difficult for model-free methods to translate drafted tokens into tangible acceleration. By contrast, Eagle-3 exhibits a smaller decrease in τ, suggesting stronger robustness on natural language tasks.

> Furthermore, SE tasks contain substantial text alongside code. For instance, code tokens and text tokens account for 35.17% and 64.83% of the SWE-bench dataset, respectively.⁴⁴ 4 These token proportions are computed from the Qwen3-32B SWE-bench execution traces. Despite this high proportion of text, model-free methods still achieve higher mean acceptance length on SWE-bench than on MT-bench. This suggests that the textual portions in SE tasks are more structured and reusable than natural language.

SE tasks tolerate a MORE AGGRESSIVE draft width than natural-language tasks — the paper's K=3 vs K=5 sweep (method: Eagle-3, since the τ values at K=3 match Table 1's Eagle-3 row exactly), verbatim table and conclusion:

`arxiv.org/html/2604.26469 (rendered Table 5, K=3 vs K=5)` lines 454-462:

````text
+---------------+-------------------+-----------------+-----------------+-----------------+-------------------+-----------------+----------------+
| Configuration | LiveCodeBench     | HumanEval       | SWE-bench       | Defects4J       | Aider Polyglot    | CanItEdit       | MT-bench       |
|               +---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
|               | Speedup | τ       | Speedup | τ     | Speedup | τ     | Speedup | τ     | Speedup | τ       | Speedup | τ     | Speedup | τ    |
+===============+=========+=========+=========+=======+=========+=======+=========+=======+=========+=========+=========+=======+=========+======+
| K = 3         | 1.71×   | 1.87    | 1.94×   | 1.99  | 1.10×   | 1.87  | 1.21×   | 2.10  | 1.29×   | 1.81    | 2.14×   | 2.35  | 1.69×   | 1.56 |
+---------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
| K = 5         | 1.79×   | 2.25    | 2.26×   | 2.55  | 1.53×   | 2.39  | 2.23×   | 2.83  | 1.43×   | 2.27    | 2.56×   | 3.32  | 1.76×   | 1.86 |
+---------------+---------+---------+---------+-------+---------+-------+---------+-------+---------+---------+---------+-------+---------+------+
````

> As presented in Table 5, the aggressive configuration (K = 5) yields consistent performance gains across both SE and natural language tasks. However, the improvements are more pronounced in the SE domain. Across the six SE benchmarks, the mean speedup and mean acceptance length (τ) increase by approximately 25.7% (from 1.57× to 1.97×) and 30.2% (from 2.00 to 2.60), respectively. In contrast, on the natural language task, the gains are relatively modest, with speedup and τ increasing by 4.1% (from 1.69× to 1.76×) and 19.2% (from 1.56 to 1.86), respectively. Notably, on the repository-level SWE-bench task, the speedup increases substantially from 1.10× to 1.53×, and similar gains appear on Defects4J and CanItEdit.

> This observation indicates that due to the higher predictability and contextual redundancy of SE tasks compared to natural language, SD in SE tasks can accommodate more aggressive hyperparameter settings than the standard configurations used in natural language.

The paper's own practical recommendation, verbatim:

> Our experiments indicate that model-free methods often provide practical acceleration in SE tasks, particularly when the workload contains reusable contextual redundancy, such as in repository-level editing and repair. However, their benefits are not universal across all model-task combinations. Due to their simple deployment, zero training cost, and absence of additional GPU memory overhead for draft inference, we still recommend prioritizing these methods for resource-constrained environments, such as local IDE assistants.

## Relevance
This is the most directly comparable independent replication of SuffixDecoding's own 7.8-on-SWE-Bench headline figure (Section 1 above), and it does NOT reproduce it: at K=32, the HIGHEST Suffix Decoding τ anyone measures here is **5.48** (L31-8B on SWE-bench — note this is also the model with the WORST Resolved rate, 1.4%, which the paper explicitly flags as the infinite-loop confound, not genuine productive copying), and PLD's τ on SWE-bench ranges only 2.69-3.25 across all four models at K=5 default. That is a real, numbered contradiction worth sitting with: the same method (Suffix Decoding, same K=32 default, same benchmark family, SWE-bench) produces 7.8 in one paper and a maximum of 5.48 in another, a 30%+ gap, apparently driven by target-model choice (Llama-3.1-8B-Instruct vs. this paper's four different models) and by how much of the measured acceptance is genuine prompt-copying versus agentic-loop repetition. Three mechanisms matter for us directly: (1) the paper's own diminishing-returns finding — speedup stops tracking τ once verification becomes compute-bound in long contexts (SWE-bench averages 15,999 tokens/request vs 817 for LiveCodeBench) — which is exactly the risk in our regime if a wide draft tree makes the verify pass itself the bottleneck; (2) the explicit warning to distrust very high model-free acceptance numbers without checking for repetitive-loop inflation, which is a methodological check we should run against our own 7.1 before trusting it as "real" productive copying; (3) the K=3-vs-K=5 sweep is independent, numbered confirmation that SE/code tasks specifically tolerate a wider draft than the generic default, with τ gains of 19-30% from widening K by less than 2x — small versus our width of 32, suggesting (as in SuffixDecoding's own Fig. 2) the marginal value of extra width keeps shrinking well before 32.

---

### Lookahead (Ant Group) — a trie tree holding EVERY branch seen so far, hierarchical merge of shared prefixes, deployed in Alipay since April 2023

https://arxiv.org/abs/2312.12728 · https://arxiv.org/html/2312.12728v3 (v3)
Yao Zhao, Zhitian Xie, Chenyi Zhuang, Jinjie Gu (Ant Group, Hangzhou). Code: https://github.com/alipay/PainlessInferenceAcceleration

Abstract, verbatim:

> As Large Language Models (LLMs) have made significant advancements across various tasks, such as question answering, translation, text summarization, and dialogue systems, the need for accuracy in information becomes crucial, especially for serious financial products serving billions of users like Alipay. However, for a real-world product serving millions of users, the inference speed of LLMs becomes a critical factor compared to a mere experimental model.
>
> Hence, this paper presents a generic framework for accelerating the inference process, resulting in a substantial increase in speed and cost reduction for our LLM-based scenarios, with lossless generation accuracy. In the traditional inference process, each token is generated sequentially by the LLM, leading to a time consumption proportional to the number of generated tokens. To enhance this process, our framework, named lookahead, introduces a multi-branch strategy. Instead of generating a single token at a time, we propose a Trie-based retrieval and verification mechanism to be able to accept several tokens at a forward step. Our strategy offers two distinct advantages: (1) it guarantees absolute correctness of the output, avoiding any approximation algorithms, and (2) the worst-case performance of our approach is equivalent to the conventional process. We conduct extensive experiments to demonstrate the significant improvements achieved by applying our inference acceleration framework. Our framework is widely deployed in Alipay since April 2023, and obtain remarkable 2.66x to 6.26x speedup. Our code is available at https://github.com/alipay/PainlessInferenceAcceleration.

The mechanism, verbatim — multiple branches retrieved from one trie, verified together, longest-correct-subsequence kept per branch:

> Table 1 summarizes the acceleration techniques discussed previously. To address their limitations, we introduce Lookahead, a groundbreaking framework that incorporates a trie-tree-based retrieving strategy and a multi-branch-based parallel Verification and Accept (VA) strategy. Lookahead leverages a trie tree to record the n-gram tokens of input prompts and generated responses. The draft is retrieved based on the provided context tokens, allowing for extremely fast draft generation. The draft tokens arranged in a logical tree structure are efficiently processed in the Verification and Accept (VA) process. Furthermore, we have implemented an adaptive strategy to optimize the retrieval process, effectively striking a balance between memory and computation requirements. It shows that Lookahead proves its superior performance in accelerating LLMs’ inference, compared with the existing state-of-the-art (SOTA) acceleration method.

The definition that plays the role of our "accepted tokens per round" (they call it EDL, Effective Decoding Length), and the single-branch ceiling it is built to break:

> Definition 1 Following the predicted next token, only the successive validated tokens from the beginning of the branch draft are kept and accepted as the output tokens, whose length is called the effective decoding length, EDL.

> Simply extending the single-branch draft’s length over EDL not only fails to promise a longer output tokens’ length per decoding step l and a higher inference speed V, but also wastes more computation. Apparently, how to achieve a longer EDL, namely the effective decoding length, is the key to further accelerate LLMs’ inference.

Multi-branch and hierarchical multi-branch draft — the exact worked example of merging several branches that share a prefix into one tree before verification:

> In contrast to the single-branch strategy such as (Yang et al., 2023a), which only considers one available draft without considering the existence of other drafts, the multi-branch strategy retrieves multiple drafts simultaneously. These drafts are then efficiently decoded and validated in parallel through the Verification and Accept (VA) process, progressively. The VA process then identifies the correct sub-sequence for each draft and retains the longest sub-sequence as the output tokens. Figure 2 presents the progress overview including various draft retrieving and verifying strategies.

> Fortunately, through careful observation of the context, we have noticed that certain branches in the drafts have common prefix tokens. For example, in Figure 2, the branches [on, my, knee] and [on, a, table] both share the prefix token [on]. By leveraging this observation, we can compress and organize the multi-branch draft using a hierarchical structure, allowing for the inclusion of additional branches while maintaining the same token capacity. To achieve this, we recursively merge the prefix token(s) shared by multiple branches, and then append the remaining tokens of each branch. We refer to these compressed and organized drafts as hierarchical multi-branch draft.

> As illustrated in Figure 2 describing the process of an inference step, through the single-branch strategy, the single draft [on, my, knee] is appended to the next token [sits] directly for the single forward process in VA, which in turn gives the output [on, a]. Being benefited from the parallel multi-branch draft, two branches [on, my, knee] and [on, a, table] are organized in parallel and appended to the next token [sits] for verifying, which outputs [on, a, chair]. By utilizing the hierarchical multi-branch draft, the shared prefix token [on] are merged, after which the token list [my, knee, a, table] are appended. By doing so, given the same decoding length, we are able to save an additional space to append one more token [chair] from another branch [on, a, chair]. Combined with the next token, we collect the token list [sits, on, my, knee, a, table, chair] for decoding through the single forward process and output [on, a, chair, and]. Therefore, compared to the single-branch strategy, our multi-branch strategy, particularly the hierarchical multi-branch draft, offers the advantage of retrieving multiple drafts, resulting in improved EDL from a statistical perspective and thus significantly enhancing LLMs’ inference speed. As illustrated in Figure 3, the position IDs and causal masks in a transformer-based model (Vaswani et al., 2017) are also merged to align with the merged token list. By doing so, we are able to accommodate more branches using hierarchical multi-branch draft, compared to parallel multi-branch draft.

Trie-tree-based draft retrieval — every node is one token, root-to-leaf is one branch, and BOTH the prompt and the generated output are inserted on the fly:

> To enable hierarchical multi-branch draft, we utilize a trie tree (Briandais, 1959; Ferragina et al., 2023), a data structure that is widely used for efficient retrieval and storage purposes and handles prefix matching by organizing nodes as individual characters or words. In our work, each node of the trie tree represents a token ID, and a path from the root to a leaf node represents a branch. The trie tree is initialized when a model is loaded, and it is alive until the model instance is shut down.

> Prompt branch Inserting. Considering that in several typical scenarios (e.g., RAG, Summary), the output are often derived from the prompt, we thus manipulate the prompt into the branches and insert them into the trie tree.

> Generated branch Inserting. Different from LLMA which only uses branches from prompts, we also reuse the branch from generated tokens. We find that, towards various prompts, there may be similar outputs, e.g., "keep exercising" in answering "how to keep healthy" and "how to be stronger". Moreover, we also find that some tokens may be repeated in a response, from example, "numbers.push_back" may occur several times in a code-generation response. To benefit from the repeat, we put the generated branches into the trie tree on-the-fly instead of the final step.

> Node Pruning. To maintain the trie tree within a moderate size, in case that the oversized trie tree results in high memory consumption and slow retrieval performance, we develop a pruning strategy. To achieve this, we decay the branch frequency and remove the nodes with frequency less than 1, when the trie tree exceeds a predetermined capacity. By doing so, we are able to optimize memory consumption and improve retrieval performance in our framework.

The exact multi-stage retrieval and branch-selection policy — THIS is the direct answer to "how many candidates to take per match, and how to break ties": shrink the match-length window until there are enough candidate tokens, keep ALL branches if their count is below a threshold, otherwise keep the highest-frequency ones, and amplify prompt-derived branches:

> Multi-stage Retrieval. We can extract a sub tree from a trie tree by providing a prefix, which is essentially a list of tokens. The sub tree is also a trie tree and can be directly used for the hierarchical multi-branch draft. The length of the prefix influences the number and relevance of the retrieved branches. Short prefixes yield a greater number of branches, while longer prefixes result in more related branches. To strike a balance between the count and correlation of the branches, we adopt a multi-stage retrieval strategy inspired by (Yang et al., 2023a). Specifically, we begin by attempting to match a longer prefix. If the number of tokens associated with the matched branches is significantly smaller than the CDL, we reduce the length of the prefix and retry the matching process until we obtain a substantial number of tokens linked to matched branches. If the count of the matched branches falls below a predefined threshold, we utilize all of them for the VA process. However, if the count exceeds a given size, we choose the tokens with the highest frequency.

> Branch Weighting. For token sorting by frequency, we employ a weighted scheme that considers both prompt and response frequencies. Intuitively, branches from an input prompt may be more related to current generation than that of other responses. To prioritize the importance of branches derived from the input prompt, we amplify their frequency by a significant factor.

Main result table (tokens/s and speedup vs. transformers baseline), reproduced verbatim from the rendered paper — note the LLMA baseline is itself a single-branch n-gram/prompt-lookup method, so the lookahead column is the multi-branch gain over classic PLD at matched conditions:

`arxiv.org/html/2312.12728v3 (rendered Table 2)` lines 264-280:

````text
  model        dataset       device     transformers   vLLM           LLMA           LaDe          lookahead(ours)
  ------------ ------------- ---------- -------------- -------------- -------------- ------------- -----------------
  AntGLM-10B   AntRAG        A100-80G   52.4           52.06(x0.99)   165.4(x3.16)   -             280.9(x5.36)
  AntGLM-10B   AntRAG        A10        20.3           20.29(x1.00)   59.5(x2.93)    -             105.1(x5.18)
  AntGLM-10B   AntRAG        V100-32G   27.3           27.28(x1.00)   64.3(x2.36)    -             118.9(x4.36)
  Llama-7B     Dolly         A100-80G   50.4           91.04(x1.81)   60.7(x1.20)    66.7(x1.32)   106.8(x2.12)
  Llama-7B     Dolly         A10        31.4           32.58(x1.04)   35.8(x1.14)    38.2(x1.22)   55.7(x1.77)
  Llama-7B     GSM8k         A100-80G   41.4           92.09(x2.22)   69.0(x1.67)    94.3(x2.28)   111.3(x2.69)
  Llama-7B     GSM8k         A10        31.4           32.73(x1.04)   41.1(x1.31)    48.9(x1.56)   68.1(x2.17)
  Llama-7B     HumanEval-x   A100-80G   51.1           90.47(x1.77)   66.5(x1.30)    88.3(x1.73)   161.5(x3.16)
  Llama-7B     HumanEval-x   A10        30.9           32.46(x1.05)   42.1(x1.36)    46.0(x1.49)   89.6(x2.90)
  Llama-13B    Dolly         A100-80G   39.9           51.67(x1.29)   59.0(x1.48)    45.2(x1.13)   84.6(x2.12)
  Llama-13B    Dolly         V100-32G   20.5           22.07(x1.08)   23.9(x1.17)    23.2(x1.13)   35.2(x1.72)
  Llama-13B    GSM8k         A100-80G   42.9           52.06(x1.21)   51.8(x1.21)    64.8(x1.51)   103.4(x2.41)
  Llama-13B    GSM8k         V100-32G   22.0           22.43(x1.02)   25.8(x1.17)    27.8(x1.26)   45.6(x2.07)
  Llama-13B    HumanEval-x   A100-80G   35.0           51.49(x1.47)   57.7(x1.65)    66.1(x1.89)   137.3(x3.92)
  Llama-13B    HumanEval-x   V100-32G   21.5           22.33(x1.04)   28.8(x1.34)    27.5(x1.28)   57.0(x2.65)
````

> As depicted in Table 2, our Lookahead obtain significant improvement over other methods, on various models, datasets and devices. The average inference speed of AntGLM-10B towards AntRAG is recorded at 52.4 tokens/s on A100 GPU. However, with the incorporation of LLMA, this speed is elevated to 165.4 tokens/s, resulting in a notable 3.16 times speed improvement. Our Lookahead with hierarchical multi-branch draft via trie tree propels the average inference speed even further, culminating in a 5.36 times speed-up, surpassing LLMA by 70% in terms of acceleration. Lookahead also obtains similar acceleration with AntRAG dataset on less-powerful devices, such as A10 and V100, which are widely used for deployment. Despite of RAG scenarios, Lookahead also consistently demonstrates its remarkable superiority and applicability across diverse datasets, especially with speedup of 3.92 on the HumanEval-x dataset, further emphasizing its practicality in real-world scenarios.

Ablation on which trie-update procedures matter (Table 3), and on trie node capacity (Table 4) — the capacity result sets a concrete, load-bearing default: 16x the decoding length:

`arxiv.org/html/2312.12728v3 (rendered Table 3)` lines 296-299:

````text
  ----------- ------------ ------------ ------------- ------------ -----------
  Condition   W/o prompt   W/o output   W/o pruning   W/o elimin   Lookahead
  Token/s     234.5        202.0        221.4         234.2        280.9
  ----------- ------------ ------------ ------------- ------------ -----------
````

> Table 3 showcases the inference speeds of Lookahead, which exhibit variability depending on different procedures used for updating the trie tree. These procedures include branch inserting with prompt and/or output, branch eliminating, and node pruning. In comparison to the results obtained without a specific procedure, Lookahead attains its optimal performance when all the aforementioned procedures are applied to dynamically update the trie tree, highlighting the necessity of these procedures in updating the trie tree.

`arxiv.org/html/2312.12728v3 (rendered Table 4)` lines 303-307:

````text
  Capacity       1*DL    2*DL    4*DL    8*DL    16*DL   32*DL   64*DL   128*DL
  -------------- ------- ------- ------- ------- ------- ------- ------- --------
  Token/s        254.5   258.7   268.8   273.3   280.7   280.8   279.5   278.3
  Retrieve(ms)   0.81    1.35    1.22    1.46    1.02    1.29    1.75    2.38
  Update(ms)     0.10    0.12    0.11    0.11    0.12    0.12    0.15    0.16
````

> The capacity of the trie tree has a significant impact on the acceleration performance. To determine the optimal capacity, we conduct empirical experiments, as outlined in Table 4. Instead of using a fixed value, we normalize the node capacity in relation to the decoding length, which allows us to identify a suitable hyper-parameter that performs well across different decoding lengths. Based on our findings, we conclude that node capacities of 16 and 32 times the decoding length yield the best results. Consequently, we have set the default capacity in our framework to be 16 times the decoding length. We also examine the retrieving and updating time of trie tree with different capacity, the time is negligible when compared with the forward time.

Why EDL keeps growing with more branches, and where it stops paying, verbatim:

> Generally speaking, it can be observed that as the decoding and branch lengths increase, there is an upward trend in the overall inference speed. As mentioned, with the single-branch strategy, LLMA is unable to further enhance the inference speed while its branch length surpasses 25. This limitation arises due to its inability to ensure a longer EDL. By implementing the multi-branch strategy, Lookahead is able to retrieve multiple branch drafts and thus guarantee a longer EDL after the VA process, as illustrated in Figure 5. Consequently, this leads to a significantly higher inference speed, while maintaining the same branch length as LLMA in Figure 4. Furthermore, the improved EDL as shown in Figure 5 also provides more headroom for increasing the branch length and thereby enhancing the upper limit of the inference speed.

> In addition, Figure 5 shows that being benefited from dealing with more branch drafts, Lookahead using the hierarchical multi-branch draft promises the advantage of a longer EDL and consequently improved inference speed, compared to using the parallel multi-branch draft with the identical branch and decoding lengths. It should be noted that in Figure 5, improving the decoding length promises a longer EDL using Lookahead, however the oversized decoding length that surpasses the CDL fails to promise a higher inference speed, due to a more complex calculation logic for the tensor cores that has been introduced in the preliminary.

## Relevance
This is a second, independent production system (deployed since April 2023, "lossless" by construction) that answers the exact question the task asked: when the same context has several matching branches with different continuations, Lookahead does not pick one — it keeps all of them under a count threshold, merges their shared prefixes into one tree (`hierarchical multi-branch draft`), and verifies the whole tree in one forward pass with merged position IDs and a merged causal mask. Two numbers are directly load-bearing for a node-budget decision on our engine: the trie capacity sweep (Table 4) shows token/s is already within 0.04% of its peak at **16x the decoding length** and gains nothing moving to 32x or beyond (280.7 vs 280.8 vs 279.5 vs 278.3 tokens/s for 16/32/64/128x) — i.e. node budget, not decoding length, is the thing to cap, and 16x is already past the knee; and the branch-selection rule itself (fall back to "all branches" below a threshold, else "highest frequency") is a complete, two-line policy for exactly the "multiple occurrences, multiple continuations" case with no free parameters beyond the threshold and the prompt-frequency amplification factor. Table 2's own comparison is also a clean, isolated measurement of tree-vs-chain at matched everything-else: LLMA (single best-match chain, i.e. classic PLD) gets 3.16x speedup on AntRAG/AntGLM-10B, Lookahead (same trie, same corpus, multi-branch tree) gets 5.36x — a 1.70x further multiplier from branching alone, the largest such isolated tree-over-chain gain found anywhere in this file, though on an internal RAG-style benchmark rather than agentic code.

---

### vLLM-Ascend official benchmark — Suffix Decoding on 4-card Atlas 800T A2 / Qwen3-32B, with per-dataset ACCEPTANCE RATE at num_speculative_tokens=3

https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/features/suffix_speculative_decoding.html (vLLM Ascend project, zh-cn docs, `latest` as fetched 2026-10-20; mirrored in English at the `/en/latest/...` path with identical numbers)

This is the single source in this file that is both (a) official vLLM-Ascend documentation, i.e. the same project as Section 2's hard `npu_fused_infer_attention_score` ceiling, and (b) written natively in Chinese, not translated. Quoted in the original Chinese with the material numbers inline; no translation substitutes for the quote.

简介, verbatim:

> 后缀解码是一种基于模式匹配的推测解码优化技术。它同时从提示词和已生成内容中检索重复序列，利用频率统计来预测最可能的token延续。与传统的推测解码方法不同，后缀解码完全在CPU上运行，无需额外的GPU资源或草稿模型，从而在AI智能体和代码生成等重复性任务中实现卓越的加速效果。

> 本文档提供了在Atlas A2硬件上部署和基准测试vllm-ascend支持的后缀解码推测推理技术的分步指南。该设置使用单个Atlas 800T A2节点，采用4卡部署Qwen3-32B模型实例。基准测试使用涵盖以下类别的真实开源数据集进行：

> 本教程使用的基准测试工具是AISBench，它支持上述所有数据集的性能测试。本教程的最后部分展示了在满足SLO TPOT < 50ms的条件下，不同数据集和并发级别下启用与禁用后缀解码的性能对比。验证表明，启用后缀解码后，Qwen3-32B模型在各种真实数据集上的吞吐量提升了约20%到80%。

The exact deployment config — note `num_speculative_tokens` is set to **3**, not anywhere near the 16-token Ascend ceiling or our width of 32, and the exact flag names:

> 使用以下命令启动容器服务实例。通过--speculative-config参数启用推测推理，其中method设置为suffix。本次测试中，num_speculative_tokens统一设置为3。

`vllm serve command, verbatim` lines 300-316:

````bash
# set the NPU device number:
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# Set the operator dispatch pipeline level to 1 and disable manual memory control in ACLGraph
export TASK_QUEUE_ENABLE=1
# Enable the AIVector core to directly schedule ROCE communication.
export HCCL_OP_EXPANSION_MODE="AIV"
vllm serve /data/Qwen3-32B \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 4 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --speculative-config '{"method": "suffix", "num_speculative_tokens": 3}' \
  --gpu-memory-utilization 0.9 \
  --additional-config '{"pa_shape_list":[48,64,72,80], "weight_prefetch_config":{"enable":true}}' \
  --port 8011
````

The summary table and the full per-dataset results, reproduced verbatim (note the `接受率` column — "acceptance rate" — is a genuine PUBLISHED measured number on real Ascend hardware, at K=3):

`测试结果 summary table, verbatim` lines 358-361:

````text
  数据集类别   典型代表         吞吐量提升 (BS=1-10)   SLO TPOT
  ------------ ---------------- ---------------------- ----------
  高增益       AGIEval, GSM8K   > 50%                  < 50ms
  中低增益     ARC, ShareGPT    20% ~ 30%              < 50ms
````

> 以下是本次评估中六个开源数据集的详细测试结果。与基线相比，启用后缀解码后，在不同并发级别下TPOT和吞吐量性能的提升因数据集而异。以下是结果摘要：

> 以下是原始详细测试结果：

`原始详细测试结果 full table, verbatim` lines 365-390:

````text
  并发数            平均输入   平均输出   请求数   基础TPOT(毫秒)   基础吞吐量(TPS)   后缀TPOT(毫秒)   后缀吞吐量(TPS)   接受率   TPOT增益   TPS增益
  ----------------- ---------- ---------- -------- ---------------- ----------------- ---------------- ----------------- -------- ---------- ---------
  HumanEval                                                                                                                                  
  1                 150        2700       100      55.1             18.1              37.9             26.3              27.0%    45.2%      45.1%
  15                150        2700       100      61.6             233.8             45.8             318.2             27.0%    34.6%      36.1%
  26                150        2700       100      64.7             403.8             50.9             519.2             27.0%    27.2%      28.6%
  ARC                                                                                                                                        
  1                 76         960        100      52.8             18.9              39.5             25.4              23.9%    33.7%      34.6%
  8                 76         960        100      59.1             125.4             47.0             163.1             23.9%    25.7%      30.0%
  15                76         960        100      59.8             245.8             48.9             311.7             23.9%    22.3%      26.8%
  GSM8K                                                                                                                                      
  1                 67         1570       100      55.5             18.0              35.7             28.5              31.1%    55.6%      58.4%
  17                67         1570       100      61.5             279.8             45.4             403.0             31.1%    35.6%      44.0%
  26                67         1570       100      63.9             396.4             50.0             527.6             31.1%    27.8%      33.1%
  ShareGPT                                                                                                                                   
  1                 666        231        327      54.1             18.3              39.2             24.1              23.9%    37.9%      31.5%
  8                 666        231        327      58.8             125.0             46.2             153.2             23.9%    27.1%      22.5%
  14                666        231        327      61.8             227.0             49.9             273.9             23.9%    23.8%      20.7%
  SuperGLUE_BoolQ                                                                                                                            
  1                 207        314        100      54.1             18.4              36.1             26.8              33.4%    49.8%      45.6%
  16                207        314        100      60.0             229.7             43.5             303.9             33.4%    38.0%      32.3%
  32                207        314        100      62.7             396.4             47.8             507.5             33.4%    31.3%      28.0%
  AGIEval                                                                                                                                    
  1                 735        1880       100      53.1             18.7              31.8             34.1              50.3%    66.8%      81.9%
  24                735        1880       100      64.0             381.2             43.3             629.0             50.3%    47.8%      65.0%
  34                735        1880       100      70.0             494.6             50.2             768.4             50.3%    39.4%      55.3%
````

## Relevance
This is the only source in the whole file that measures suffix-decoding acceptance on the exact hardware family (Atlas A2 / Ascend NPU, vLLM-Ascend) our engine targets, rather than on H100/A100. Three things are directly load-bearing: (1) they run `num_speculative_tokens: 3`, i.e. a width of 3 — a full order of magnitude below our 32 and below the 16-token hardware ceiling documented in Section 2, which tells us production Ascend deployments of suffix decoding are not yet pushing width at all; (2) 接受率 (acceptance rate) at that tiny width already reaches 50.3% on AGIEval and 31.1% on GSM8K — these are rate-of-accepted-draft-tokens, not mean-accepted-length, but they establish that even K=3 captures a large fraction of available signal on structured/repetitive datasets, which is consistent with the width-saturation picture in Sections 1, D and E (most of the signal is captured by a short match, width mainly helps the long tail); (3) their own baseline TPOT (52.8-70.0 ms at BS=1-34) and suffix-decoding TPOT (31.8-50.9 ms) are both two orders of magnitude above our 2.x ms target, on a 32B dense model rather than our W4A8C8 MoE, so the comparison is suggestive of real on-Ascend gain shape but not a numeric target to match.

---

### Ascend tree-attention mask mechanics — `npu_fused_infer_attention_score` DOES take an arbitrary tree mask (proven by a shipped AscendC kernel), and the 16-token cap is a NODE-COUNT cap, not a chain-depth cap

https://github.com/sgl-project/sgl-kernel-npu @ a5c12d19fe12dc03aca474852f63cf49f5df8eb2 (committed 2026-10-19, retrieved 2026-10-20) — `csrc/build_tree/` and `python/sgl_kernel_npu/sgl_kernel_npu/sample/verify_tree_greedy.py`
https://github.com/vllm-project/vllm-ascend — `vllm_ascend/attention/mla_v1.py` and `vllm_ascend/attention/attention_v1.py` (as cloned 2026-10-20)

This answers the question directly: SGLang's Ascend backend already runs EAGLE-style dynamic-tree speculative decoding in production through `npu_fused_infer_attention_score`/`npu_fused_infer_attention_score_v2`, by building the tree mask on-device with a dedicated AscendC kernel (`build_tree_efficient`) and walking it at verify time with a Triton kernel compiled for Ascend (`triton.language.extra.cann.extension`). The operator's `atten_mask` argument is a plain tensor; nothing in its signature restricts it to a causal/chain shape.

The kernel's own README states the exact tensor shapes it produces, verbatim:

`csrc/build_tree/README.md` lines 33-48:

````python
import sgl_kernel_npu

torch.ops.npu.build_tree_kernel_efficient(
    parent_list: torch.Tensor,           # int64, [batch_size, topk*(depth-1)+1]
    selected_index: torch.Tensor,        # int64, [batch_size, draft_token_num-1]
    verified_seq_len: torch.Tensor,      # int64, [batch_size]
    tree_mask: torch.Tensor,             # bool,  [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] = [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token]
    positions: torch.Tensor,             # int64, [batch_size, draft_token_num]
    retrive_index: torch.Tensor,         # int64, [batch_size, draft_token_num]
    retrive_next_token: torch.Tensor,    # int64, [batch_size, draft_token_num]
    retrive_next_sibling: torch.Tensor,  # int64, [batch_size, draft_token_num]
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int
) -> None
````

`csrc/build_tree/README.md (parameter table excerpt)` lines 65-72:

````text
## Parameter Description | 参数说明

| Parameter Name (参数名称) | DataType (数据类型) | Description                               | 说明                         |
|:----------------------|:----------------|:------------------------------------------|:---------------------------|
| `parent_list`         | `torch.Tensor`  | parent id of every draft token            | 每个 draft token 的父节点 id     |
| `selected_index`      | `torch.Tensor`  | flat index of sampled token in top-k list | 采样的 token 在 top-k 列表中的扁平索引 |
| `verified_seq_len`    | `torch.Tensor`  | length of already-verified prefix         | 当前已验证序列长度                  |
| `topk`                | `int`           | branching factor per step                 | 每步分支数                      |
````

> | `tree_mask`            | `torch.Tensor`  | true → node must be verified       | true → 该节点需被验证     |

The AscendC kernel that actually writes the tree mask bits and the first-child/next-sibling retrieval links, one query-token-per-core-thread (`tid`), verbatim (this is the device-side mask CONSTRUCTION, the thing the task asked to chase down):

`csrc/build_tree/op_kernel/build_tree_kernel.cpp` lines 120-201:

````cpp
private:
    __aicore__ inline void Compute(int32_t bid, uint32_t tid, int64_t seq_len, int64_t seq_tree_idx,
                                   AscendC::LocalTensor<int64_t> &positions_ub,
                                   AscendC::LocalTensor<int64_t> &retrive_index_ub,
                                   AscendC::LocalTensor<int64_t> &retrive_next_token_ub,
                                   AscendC::LocalTensor<int64_t> &retrive_next_sibling_ub)
    {
        auto bs = bs_offset + bid;
        int token_tree_idx;
        if (tree_mask_mode == sglang::npu_kernel::TreeMaskMode::FULL_MASK) {
            // [seq_lens_sum * num_verify_tokens + num_verify_tokens * num_verify_tokens * bs, ]
            token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len;
        } else {
            // [num_verify_tokens * bs * num_verify_tokens, ]
            token_tree_idx = draft_token_num * draft_token_num * bs + draft_token_num * bs;
        }

        AscendC::LocalTensor<bool> tree_mask_ub = mask_queue.AllocTensor<bool>();
        tree_mask_ub.SetValue(0, true);
        for (int i = 1; i < draft_token_num; i++) {
            tree_mask_ub.SetValue(i, false);
        }

        int64_t position = 0;
        if (tid == 0) {
            positions_ub.SetValue(0, seq_len);

            int64_t retrive_index_offset = bs * draft_token_num;
            for (int64_t i = draft_token_num - 1; i > 0; --i) {
                int64_t current_token_idx = retrive_index_offset + i;
                retrive_index_ub.SetValue(i, current_token_idx);
                int64_t parent_tb_idx = selected_index_gm.GetValue(bid * (draft_token_num - 1) + i - 1) / topk;
                int64_t parent_position = 0;
                if (parent_tb_idx > 0) {
                    int64_t parent_token_idx = parent_list_gm.GetValue(bid * (topk * (depth - 1) + 1) + parent_tb_idx);
                    for (; parent_position < draft_token_num; ++parent_position) {
                        if (selected_index_gm.GetValue(bid * (draft_token_num - 1) + parent_position) ==
                            parent_token_idx) {
                            ++parent_position;
                            break;
                        }
                    }
                }
                if (parent_position == draft_token_num) {
                    continue;
                }

                if (retrive_next_token_ub.GetValue(parent_position) == -1) {
                    retrive_next_token_ub.SetValue(parent_position, i);
                } else {
                    int64_t origin_next_token = retrive_next_token_ub.GetValue(parent_position);
                    retrive_next_token_ub.SetValue(parent_position, i);
                    retrive_next_sibling_ub.SetValue(i, origin_next_token);
                }
            }
            retrive_index_ub.SetValue(0, bs * draft_token_num);
        } else {
            int64_t cur_position = tid - 1;
            while (true) {
                position += 1;
                tree_mask_ub.SetValue(1 + cur_position, true);
                int64_t parent_tb_idx = selected_index_gm.GetValue(bid * (draft_token_num - 1) + cur_position) / topk;
                if (parent_tb_idx == 0) {
                    break;
                }

                int64_t token_idx = parent_list_gm.GetValue(bid * (topk * (depth - 1) + 1) + parent_tb_idx);
                for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
                    if (selected_index_gm.GetValue(bid * (draft_token_num - 1) + cur_position) == token_idx) {
                        break;
                    }
                }
            }
            positions_ub.SetValue(tid, position + seq_len);
        }

        this->mask_queue.EnQue(tree_mask_ub);
        AscendC::LocalTensor<bool> mask_ub = mask_queue.DeQue<bool>();
        AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>(this->draft_token_num * sizeof(bool)), 0, 0, 0};
        AscendC::DataCopyPad<bool>(this->tree_mask_gm[token_tree_idx], mask_ub, copy_params);
        mask_queue.FreeTensor(mask_ub);
    }
````

The kernel entry point — a plain AscendC `__global__ __aicore__` function, called once per decode-verify round:

`csrc/build_tree/op_kernel/build_tree_kernel.cpp` lines 257-267:

````cpp
extern "C" __global__ __aicore__ void build_tree_efficient(GM_ADDR parent_list, GM_ADDR selected_index,
                                                           GM_ADDR verified_seq_len, GM_ADDR tree_mask,
                                                           GM_ADDR positions, GM_ADDR retrive_index,
                                                           GM_ADDR retrive_next_token, GM_ADDR retrive_next_sibling,
                                                           GM_ADDR workspace_in, GM_ADDR tiling_in)
{
    KernalBuildTree op;
    op.Init(parent_list, selected_index, verified_seq_len, tree_mask, positions, retrive_index, retrive_next_token,
            retrive_next_sibling, workspace_in, tiling_in);
    op.Process();
}
````

Verification against that mask — the SGLang-Ascend greedy tree-verify kernel, written in Triton with an Ascend (CANN) extension rather than CUDA, but structurally identical to the CUDA `VerifyTreeGreedy` kernel quoted in Section 2 above (same `retrive_next_token` / `retrive_next_sibling` walk, same single-token-per-request `req_idx` grid):

`python/sgl_kernel_npu/sgl_kernel_npu/sample/verify_tree_greedy.py` lines 1-67:

````python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def verify_tree_greedy_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    num_draft_tokens: tl.constexpr,
    pad_num_draft_tokens: tl.constexpr,
):
    """
    predicts: [tot_num_draft_tokens] -> ((bs * (self.spec_steps + 1))
    accept_index: [bs, num_spec_step + 1]
    accept_token_num: [bs]
    candidates: [bs, num_draft_tokens]
    retrive_index: [bs, num_draft_tokens]
    retrive_next_token: [bs, num_draft_tokens]
    retrive_next_sibling: [bs, num_draft_tokens]
    target_predict: [bs, num_draft_tokens]
    """
    req_idx = tl.program_id(0)
    tokens_offset = tl.arange(0, pad_num_draft_tokens)
    cur_tokens_offset = req_idx * num_draft_tokens + tokens_offset
    mask_draft_mask = tokens_offset < num_draft_tokens
    cur_candidates = tl.load(
        candidates + cur_tokens_offset, mask=mask_draft_mask, other=0
    )
    cur_retrive_index = tl.load(
        retrive_index + cur_tokens_offset, mask=mask_draft_mask, other=0
    )
    cur_target = tl.load(
        target_predict + cur_tokens_offset, mask=mask_draft_mask, other=0
    )

    last_accepted_idx = al.get_element(cur_retrive_index, (0,))
    tl.store(accept_index + req_idx * num_draft_tokens, last_accepted_idx)
    num_accepted = 0

    rejected = False
    for i in range(1, num_draft_tokens):
        if not rejected:
            draft_token = al.get_element(cur_candidates, (i,))
            target_token = al.get_element(cur_target, (i - 1,))

            if draft_token == target_token:
                draft_idx = al.get_element(cur_retrive_index, (i,))
                tl.store(predicts + last_accepted_idx, target_token)
                num_accepted += 1
                tl.store(
                    accept_index + req_idx * num_draft_tokens + num_accepted, draft_idx
                )
                last_accepted_idx = draft_idx
            else:
                rejected = True
    target_token = al.get_element(
        cur_target, (last_accepted_idx - num_draft_tokens * req_idx,)
    )
    tl.store(accept_token_num + req_idx, num_accepted)
    tl.store(predicts + last_accepted_idx, target_token)
````

Now the cap itself. vllm-ascend's own assertion, verbatim, identical in both its MLA and dense attention backends:

`vllm_ascend/attention/mla_v1.py` lines 311-320:

````python
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )
````

and, byte-identical in the non-MLA path:

`vllm_ascend/attention/attention_v1.py` lines 274-283:

````python
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )
````

## Relevance
Two things are now settled with code, not inference. First: Ascend's `npu_fused_infer_attention_score` family DOES support an arbitrary tree mask today — not "structurally possible in principle" but actually shipped, in `sgl-kernel-npu` + SGLang's `ascend_backend.py`, for EAGLE-style dynamic-tree draft models. vllm-ascend's own `ngram`/`suffix` proposers (Section 2) only ever emit a linear chain, but that is a gap in those two proposers' Python, not a hardware or kernel limitation — the exact same attention operator, on the exact same chip family, already verifies a tree elsewhere in the same ecosystem. Porting a tree-shaped ngram/suffix draft (SGLang's own `NGRAM` tree from Section 2, or the Lookahead/REST multi-match tries above) onto vllm-ascend's `ngram`/`suffix` path is therefore an engineering port of existing Ascend plumbing, not new kernel work. Second: `decode_threshold = 1 + num_speculative_tokens <= 16` is counted from `spec_token_num = speculative_config.num_speculative_tokens`, a flat token count with no notion of tree shape anywhere in its derivation — it bounds the FIA call's `TND`-layout query length (how many Q rows this request contributes this round), which is exactly `draft_token_num` in the SGLang/REST/Lookahead formulations above, i.e. a NODE-COUNT cap. A balanced or irregular tree of 15 nodes costs the operator exactly the same as a chain of 15, which means the entire multi-match-tree machinery in Sections C, E and 2 above is usable on this hardware up to the same 15-node (16-token) ceiling already governing our chain drafts — the tree buys higher expected-accepted-length per node, not a bigger node budget.

---

### Cursor "Instant Apply" — speculative edits is deterministic PLD-style drafting applied to full-file rewrites, 13x over vanilla, 9x over their own prior GPT-4 deployment

https://cursor.com/blog/instant-apply (Cursor, research blog)
Aman Sanger, May 14, 2024

The core idea, verbatim:

> Our fast-apply model surpasses GPT-4 and GPT-4o performance and pushes the pareto frontier on the accuracy / latency curve. We achieve speeds of ~1000 tokens** (around 3500 char/s) on our 70b model using a speculative-decoding variant tailored for code-edits, called speculative edits.

> This means a ~13x speedup over vanilla inference using Llama-3-70b and a ~9x speedup over our previous GPT-4 speculative edits deployment.

> By default, we have language models generate the fully rewritten file conditioned on the current file, the conversation history, and the current code block.

Why full-file rewrite instead of a diff — this is the reason prompt-output overlap is so high in their setting, stated directly:

> In this post, we explain how we trained and evaluated our new model. We show why we rewrite the file instead of using diffs and how speculative edits give us such ridiculous speedups.

The mechanism itself, verbatim — this is the "deterministic algorithm instead of a draft model" framing the whole speculative-lookup family rests on, stated independently by a production team with no citation to Saxena/PLD:

> Our biggest win comes from our custom speculative decoding algorithm called "speculative edits". It is equivalent to a full-file-rewrite, while being up to 9x faster.
>
> With code edits, we have a strong prior on the draft tokens at any point in time, so we can speculate on future tokens using a deterministic algorithm rather than a draft model.

> We've worked with Fireworks to deploy our fast-apply model with strong speculative edits support. They have a fantastic inference engine and built out api support for our custom speculation logic.

> This makes the gain from speculative edits larger for Llama-3 than GPT-4, meaning a 4-5x speedup over the next fastest model.

Why it is specifically a custom model, not a frontier API model, verbatim:

> The claude-3-opus-diff apply model is faster and more accurate than gpt-4-turbo-spec, but is still too slow.

> It is not possible to build speculative edits into any of anthropic's models, so we need to train and deploy a performant custom model.

## Relevance
This is the production speculative-edits precedent the task asked for by name, and it is architecturally the same family as PLD/suffix decoding even though Cursor does not cite that lineage: "we have a strong prior on the draft tokens at any point in time, so we can speculate on future tokens using a deterministic algorithm rather than a draft model" is exactly Saxena's PLD insight, applied to full-file-rewrite edits instead of chat completions. No per-round accepted-token-count or acceptance-rate number is published — Cursor reports only end-to-end speedup (13x vanilla, 9x their prior GPT-4 deployment, ~1000 tok/s on a 70B model) — so this cannot be compared numerically to our 7.1. The one transferable design decision is their choice of REPRESENTATION: they rewrite the whole file rather than emit a diff, explicitly because "models have seen more complete files than diffs in training" and "models struggle with line-number counting in diffs" (Morph's gloss) — i.e. they maximize prompt-output literal overlap by construction, which is the same lever our 7.1 is already riding on an agentic-coding model that "copies its input heavily." It is independent, non-academic confirmation that the industry's actual highest-leverage move for this workload is maximizing copy-the-input overlap, not a cleverer lookup policy.

---

## Gaps

Looked for and could not find:

- **A published acceptance-length number clearly ABOVE 7.8 on a genuinely agentic benchmark.** SuffixDecoding's own 7.8 (SWE-Bench, Llama-3.1-8B, H100) remains the ceiling among primary sources with a stated tree/corpus mechanism. The closest independent approach is the vLLM-production study's InstructCoder-specific n-gram measurement — mean τ **7.27**, median 6.57 (Table A.3) — which lands almost exactly on our 7.1, but that number's own draft-length configuration is not disambiguated in the paper (the main text's default is `num_speculative_tokens=3`, but InstructCoder is separately noted as also tested at 5; neither obviously explains a mean of 7.27, so treat this figure as real but not fully traceable to a stated K). Working against a ceiling above 7.8: the ISSTA 2026 SE-tasks study, which specifically tried to replicate Suffix Decoding's own regime (K=32, SWE-bench) across four different target models, tops out at τ=5.48 — 30% BELOW 7.8, and on the model with the worst task-resolution rate, which the paper itself flags as likely infinite-loop inflation rather than genuine copying. So the state of the art is contested, not settled: one lab's 7.8 has not been reproduced by an independent replication at the same nominal configuration, and the closest match to our own 7.1 (InstructCoder's 7.27) comes with an unresolved K. Nothing in any source reports sustained acceptance in the 10-15 range, and on Ascend `num_speculative_tokens + 1 ≤ 16` caps it regardless.
- **GLM-5.2's own MTP head documentation.** vllm-ascend's reference GLM-5.2 config uses `"method": "deepseek_mtp"` with `num_speculative_tokens: 3` (see FILE 1), i.e. it borrows the DeepSeek MTP path rather than a GLM-specific head; no GLM-5.2 MTP architecture document or per-head acceptance measurement was found. The `dspark` dynamic method's own note says GLM support "is being added incrementally".
- **Per-benchmark acceptance tables from SuffixDecoding's appendix.** The appendix section headings exist in the HTML ("SWE-Bench: Mean accepted tokens (tokens/step)", "SWE-Bench: Mean Acceptance Rate", "AgenticSQL: …", "Spec-Bench: …") but the figure data itself is in plotted images, not text, so the per-sub-task breakdowns could not be quoted.
- **Measured acceptance for `dflash` / `dspark` / Block Verify / Entropy Verify.** vllm-ascend documents the mechanisms and the parameters but publishes no acceptance-length or throughput deltas for any of them.
- **SpecInfer / Sequoia / HASS's own tree-mask construction code and the Sequoia expected-accepted-length DP algebra.** Resolved for n-gram/trie-based trees instead (REST's BFS-over-weighted-Trie, SGLang's `NGRAM` recency/frequency expansion, Lookahead's hierarchical prefix-merge, and the Ascend AscendC `build_tree_efficient` kernel that writes the mask bits directly) — all quoted verbatim above. But SpecInfer's/Sequoia's/HASS's own published code for the mask itself, and Sequoia's dynamic-programming optimal-tree-topology derivation, were not retrieved; what is on record for Sequoia here is only the qualitative claim (its own figure caption) that "Sequoia trees' average number of generated tokens after verification continues to grow with tree size, while existing tree structures asymptote" — not the DP or the mask code.
- **Token Recycling, PLD+, Ouroboros, ANPD, SAM-Decoding primary numeric sources beyond what Section "SAM-Decoding" above already has.** Token Recycling still appears only as a baseline inside SuffixDecoding's and this file's other sources (3.2 mean accepted tokens/step on AgenticSQL per Section 1; SGLang lists it as a baseline comparator too); no primary paper with its own acceptance-vs-node-count table was retrieved. PLD+, Ouroboros and ANPD were targeted by search but returned no primary source with usable numbers this round.
- **A genuine Zhihu or X/Twitter thread (including replies) with measured 投机解码/n-gram acceptance numbers on code tasks.** Multiple Chinese-language queries were run (`知乎 投机解码 接受率`, `投机解码 接受率 n-gram prompt lookup 代码补全`, `n-gram 投机解码`). They surfaced only mirrors/translations of the English academic and engineering sources already quoted above, plus one genuine Chinese-native primary source — vLLM-Ascend's own `zh-cn` suffix-decoding tutorial (quoted in full above, with real 接受率 numbers from an on-Ascend benchmark) — which is official documentation, not a Zhihu/X post. No independent Zhihu article or X reply thread with its own measured number was found.
