# Ultra-low-latency decode techniques, 2025–2026

One hundred public items, seventy arXiv papers and thirty X posts, dated on or after 2025-01-01. Each record is a decode-latency technique: speculative decoding and acceptance length, MTP, megakernel or launch fusion, decode attention, quantization on the decode path, or a collective or prefill/decode transfer on that path. The quote is the source's words. The number is that source's claim under the regime it states. Nothing here was remeasured on this repo's hardware.

Relevance uses the baseline printed in `docs/reference/README.md`: TPOT 10.8528 ms, about 3008 kernels/round, launch floors AI_VECTOR_CORE 0.70 us, AI_CORE 2.30 us, MIX_AIV 12.04 us, MIX_AIC 13.58 us, prompt-lookup acceptance about 7.1, TP reduce 5.35 ms.

## r001. SpecSearch: Towards Hardware-Aware Efficient Speculative Decoding

- url: https://arxiv.org/abs/2611.22296
- authors: Selin Yildirim; Ali Zeyrek; Deming Chen
- date: 2026-11-17
- mechanism: We present SpecSearch, a hardware-aware search engine that jointly optimizes decoding strategy, draft length, and precision across workload--GPU pairs.
- regime: Named in the abstract: Qwen3-4B, H100, L40, MI300X, vLLM.
- number: Additionally, it retains $97.36\%$ of the throughput achieved by SpecSearch that utilizes full calibration.
- quote: Additionally, it retains $97.36\%$ of the throughput achieved by SpecSearch that utilizes full calibration.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r002. ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding

- url: https://arxiv.org/abs/2611.13442
- authors: Jaeyong Song; Daniel Wang; Pratyush Patel; Jonny Kong; Preyas Shah; Shobhit Kanaujia
- date: 2026-11-10
- mechanism: In this paper, we propose work-conserving speculation (WCS), a design that jointly allocates draft and verification budgets per request in LLM serving, and present ForeSpark as its prototype.
- regime: Named in the abstract: vLLM.
- number: It also tolerates higher load, sustaining up to 1.51x the request rate of DSpark under latency service level objectives (SLOs).
- quote: It also tolerates higher load, sustaining up to 1.51x the request rate of DSpark under latency service level objectives (SLOs).
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r003. Callo: Serving Cache-Miss Requests with Speculative Decoding on Invalidated KV under Prefill-Decode Disaggregation

- url: https://arxiv.org/abs/2611.18459
- authors: Jianjun Ou; Matthew Lentz
- date: 2026-11-13
- mechanism: We present Callo, a disaggregated serving system that leverages invalidated KV for speculative drafts to reduce this latency, while verifying drafts against the exact KV after prefill before committing tokens.
- regime: Named in the abstract: vLLM.
- number: We evaluated Callo on a two-GPU testbed using SWE-agent sessions across three model families, observing all-request speedups of 1.05-1.19x over vLLM at moderate load.
- quote: We evaluated Callo on a two-GPU testbed using SWE-agent sessions across three model families, observing all-request speedups of 1.05-1.19x over vLLM at moderate load.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r004. SpecRelay: Amortize or Avoid the Round Trip in Edge-Cloud Speculative Decoding

- url: https://arxiv.org/abs/2611.18963
- authors: Jai Jain; Chihyo Ahn; Yong Liu; Charles Chen; Duc N. M Hoang; Ajay Kumar Jaiswal; Fatih Ilhan; Han-Byul Kim; Lauren Hannah; Richa Dixit; Minsik Cho; Yingyan Celine Lin; Hyesoon Kim
- date: 2026-11-13
- mechanism: To recover this loss, we present SpecRelay, an edge-cloud LLM inference system that addresses this cost at two levels: it amortizes round trips when cloud verification is necessary and avoids them when local reasoning suffices.
- regime: The abstract does not name a device, batch, or model family.
- number: At matched target-model accuracy and 250 ms of added round-trip latency, the full system with the router delivers 1.95x and 1.55x the throughput of the same baseline on SPROUT and OlympiadBench, respectively, while sending only 47% and 39% of the queries to the cloud.
- quote: At matched target-model accuracy and 250 ms of added round-trip latency, the full system with the router delivers 1.95x and 1.55x the throughput of the same baseline on SPROUT and OlympiadBench, respectively, while sending only 47% and 39% of the queries to the cloud.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r005. MMFlash: Faster Speculative Decoding for Vision-Language Models

- url: https://arxiv.org/abs/2611.18482
- authors: Sicheng Feng; Zigeng Chen; Song Wang; Gongfan Fang; Xinchao Wang
- date: 2026-11-13
- mechanism: Motivated by this observation, we propose MMFlash, a simple and effective framework that adaptively upweights visually dependent tokens that the drafter predicts poorly.
- regime: The abstract does not name a device, batch, or model family.
- number: Lossless decoding speedups of up to 3.58x, outperforming EAGLE-3 and DFlash by up to 195.4% and 17.1%.
- quote: Across various multimodal benchmarks, MMFlash achieves lossless decoding speedups of up to $3.58\times$, outperforming EAGLE-3 and DFlash by up to 195.4% and 17.1%, respectively.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r006. DForkTree: Accelerating Speculative Decoding with Fork-Aware Block Diffusion Draft Trees

- url: https://arxiv.org/abs/2611.12356
- authors: Zhicheng Zheng; Cheng Zhang; Chenkai Tang; Tianxiang Zhao
- date: 2026-11-09
- mechanism: In this paper, we introduce DForkTree, a training-free method that re-anchors each branch on its own tokens.
- regime: Named in the abstract: Qwen3.5-4B.
- number: On Qwen3.5-4B and 9B, DForkTree outperforms all baselines on every benchmark, by up to 41.5% over DFlash and 15.1% over DDTree, with average speedups of up to 4.75x over autoregressive decoding.
- quote: On Qwen3.5-4B and 9B, DForkTree outperforms all baselines on every benchmark, by up to 41.5% over DFlash and 15.1% over DDTree, with average speedups of up to 4.75x over autoregressive decoding.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r007. Phrase-Level Speculative Decoding on a Frozen Target

- url: https://arxiv.org/abs/2611.12464
- authors: Madhusudhana Naidu
- date: 2026-11-09
- mechanism: We change the unit. A small MLP head reads the frozen target's hidden state and predicts a one- to three-word phrase as one entry in a separate draft vocabulary; greedy longest-match expansion maps it back into the target's own tokens, and N candidates are verified in one forward pass behind a prefix-tree mask, so the change is orthogonal to tree depth and width.
- regime: Named in the abstract: Qwen3-4B, batch 1.
- number: On one A5000 at batch 1, against HuggingFace generate(), a 21.94M head (0.545% of Qwen3-4B) reaches 2.391x on GSM8K, 2.444x on MATH-500 and 1.544x on MT-Bench.
- quote: On one A5000 at batch 1, against HuggingFace generate(), a 21.94M head (0.545% of Qwen3-4B) reaches 2.391x on GSM8K, 2.444x on MATH-500 and 1.544x on MT-Bench.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r008. Towards Green Text Generation: Memory-Based Speculative Decoding

- url: https://arxiv.org/abs/2611.16714
- authors: Antal van den Bosch
- date: 2026-11-12
- mechanism: We introduce Olifant, a draft model built on a memory-based classifier.
- regime: Named in the abstract: Llama.
- number: It predicts the next token from a four-token window in under 0.3 ms on CPU, requiring no GPU, no backpropagation, and no accelerator memory.
- quote: It predicts the next token from a four-token window in under 0.3 ms on CPU, requiring no GPU, no backpropagation, and no accelerator memory.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r009. Multi-Future Drafting: Mixture Block Drafters for Tree-Verified Speculative Decoding

- url: https://arxiv.org/abs/2611.13017
- authors: Haoyu Huang; Jialiang Kang; Linlin Yang; Xiao Yang; Baochang Zhang
- date: 2026-11-09
- mechanism: We propose Multi-Future Drafting (MFD), which restores the coupling with a latent variable instead of the chain rule: $K$ learned mode embeddings at the drafter's input draft $K$ coherent blocks in the same pass, and a mixture DDTree lets the verifier pick the future the target takes, losslessly and with $K+1$ extra vectors.
- regime: Named in the abstract: Qwen, batch size 1, Qwen3-4B, Qwen3-8B.
- number: On four Qwen targets from 4B to 27B parameters, at batch size 1, MFD reaches up to 7.8$\times$ speedup, 5-8% faster than the best single-mode tree trained identically and, on Qwen3-4B and Qwen3-8B, 9-16% faster than the released DARTree trained on the same prompts.
- quote: On four Qwen targets from 4B to 27B parameters, at batch size 1, MFD reaches up to 7.8$\times$ speedup, 5-8% faster than the best single-mode tree trained identically and, on Qwen3-4B and Qwen3-8B, 9-16% faster than the released DARTree trained on the same prompts.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r010. D3Tree: Learning to Prune Block-Diffusion Draft Trees for Efficient Verification

- url: https://arxiv.org/abs/2611.11371
- authors: Hong Nie; Zhangchi Gao; Jun Yu; Bo Wang; Jianqing Sun; Shengping Liu
- date: 2026-11-08
- mechanism: We present D3Tree, which learns which nodes of a block-diffusion draft tree are worth verifying.
- regime: The abstract does not name a device, batch, or model family.
- number: Across six vision-language verifiers and 17 text and multimodal benchmarks, D3Tree is fastest in 70 of 72 load settings, with 3.2-4.0x speedup over autoregressive decoding for one request and 1.3-1.8x at 64 concurrent requests, where fixed-budget trees fall below autoregressive speed.
- quote: Across six vision-language verifiers and 17 text and multimodal benchmarks, D3Tree is fastest in 70 of 72 load settings, with 3.2-4.0x speedup over autoregressive decoding for one request and 1.3-1.8x at 64 concurrent requests, where fixed-budget trees fall below autoregressive speed.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r011. Give the Drafter a Second Chance to Guess: Parallel Reconsideration for Speculative Decoding

- url: https://arxiv.org/abs/2611.05724
- authors: Hoang Anh Duy Le; Huibing Dong; Alireza Khadem; Ziteng Sun; Xiang Song; Yizheng Huang; Soroush Radpour; Anshumali Shrivastava; Ye He
- date: 2026-11-03
- mechanism: To address this, we introduce Parallel Reconsideration: guided by its own uncertainty, the drafter reconsiders its hesitant guesses with extra passes that run in parallel with the one-pass draft, adding negligible latency.
- regime: The abstract does not name a device, batch, or model family.
- number: Across models and benchmarks, on average, Parallel Reconsideration improves over the baselines by 5.3% to 10.2% on acceptance length and by 3.5% to 9.2% on speedup.
- quote: Across models and benchmarks, on average, Parallel Reconsideration improves over the baselines by 5.3% to 10.2% on acceptance length and by 3.5% to 9.2% on speedup.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r012. Hybrid Spark: Adaptive Speculative Decoding That Lets the Content Choose the Drafter

- url: https://arxiv.org/abs/2611.05611
- authors: Ekagra Ranjan; Bharat Venkitesh
- date: 2026-11-03
- mechanism: We present \sys, a framework that runs a model-based and a model-free drafter on every step and gives each request the better of their chains, within adaptive verification and entirely on the GPU, which keeps asynchronous scheduling intact.
- regime: Named in the abstract: vLLM.
- number: Instantiated with \dspark and an n-gram drafter in vLLM, \sys achieves 1.10--1.62\x the output throughput of \dspark on code editing at batch size~4 and 1.04--1.27\x at batch size~128, while staying at 0.97--1.01\x of it on chat at temperature~0.
- quote: Instantiated with \dspark and an n-gram drafter in vLLM, \sys achieves 1.10--1.62\x the output throughput of \dspark on code editing at batch size~4 and 1.04--1.27\x at batch size~128, while staying at 0.97--1.01\x of it on chat at temperature~0.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r013. HybridSpec: Throughput-Aware Speculative Decoding for Hybrid GDN-MoE Models

- url: https://arxiv.org/abs/2611.04055
- authors: Haiyu Wang; Sai Qian Zhang; Zechun Liu; Steven Li
- date: 2026-11-02
- mechanism: We present HybridSpec, which selects the verification length on the GPU from draft-side acceptance estimates and measured target latency.
- regime: Named in the abstract: Qwen3.6-35B-A3B, Qwen3-4B.
- number: With three GDN-MoE target models in single-request greedy decoding, HybridSpec is $1.75\times$-$1.88\times$ faster than autoregressive decoding and $1.30\times$-$1.33\times$ faster than DFlash.
- quote: With three GDN-MoE target models in single-request greedy decoding, HybridSpec is $1.75\times$-$1.88\times$ faster than autoregressive decoding and $1.30\times$-$1.33\times$ faster than DFlash.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r014. One-Sided Expert-Parallel All-to-All for Mixture-of-Experts Inference over NVLink

- url: https://arxiv.org/abs/2611.13188
- authors: Bo Li; Chulian Zhang; Or Biri; Anthony Chang; Anton Korzh; Noam Duvdevani; XingFei Xi; Daniel Stokes; Zhenhuan Chen; Ludwig Schneider; Artem Y. Polyakov; Julien Demouth; Chandler Zhou
- date: 2026-11-09
- mechanism: We present NVLinkOneSided, a one-sided EP all-to-all for GPUs within a single NVLink domain.
- regime: The abstract does not name a device, batch, or model family.
- number: On Blackwell GPUs, NVLinkOneSided completes dispatch and combine 1.2-3.0$\times$ faster than DeepEP, with the largest speedups at small batch sizes.
- quote: On Blackwell GPUs, NVLinkOneSided completes dispatch and combine 1.2-3.0$\times$ faster than DeepEP, with the largest speedups at small batch sizes.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this sits on the collective in the decode step. Our TP reduce is 5.35 ms and 100% serial, so a shorter collective would move that term only on the interconnect the source measured.

## r015. Weights That Stay: On-Chip Megakernels for Serial Heads in Speculative Decoding

- url: https://arxiv.org/abs/2611.00506
- authors: Xinwei Qiang; Ziyi Xu; Zaifeng Pan; Chang Chen; Yufei Ding
- date: 2026-10-31
- mechanism: We present Weights That Stay, an execution design that keeps projection shards on chip for the entire serial draft walk.
- regime: Named in the abstract: H100, B200, SGLang, Qwen3-4B, Qwen3-8B.
- number: At the head level, the DSpark walk is up to 11.6$\times$ faster than the fastest measured SGLang reference path.
- quote: At the head level, the DSpark walk is up to 11.6$\times$ faster than the fastest measured SGLang reference path.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r016. SharpDraft: Accelerating Long-Context Speculative Decoding with Cardinality-Aware Query Scaling

- url: https://arxiv.org/abs/2610.05106
- authors: Jeonghoon Park; Seongwoon Jo; Jongwon Lee; Taesik Gong
- date: 2026-10-04
- mechanism: We introduce SharpDraft, a training-free method that counteracts this effect through cardinality-aware Query scaling, without the computational overhead of online adaptation.
- regime: The abstract does not name a device, batch, or model family.
- number: Across AIME-26, GPQA-Diamond, and LongGenBench Diary, SharpDraft achieves $2.59$-$3.19\times$ geometric-mean end-to-end speedups over target-only autoregressive decoding when applied to DFlash, PARD, and EAGLE 3.1.
- quote: Across AIME-26, GPQA-Diamond, and LongGenBench Diary, SharpDraft achieves $2.59$-$3.19\times$ geometric-mean end-to-end speedups over target-only autoregressive decoding when applied to DFlash, PARD, and EAGLE 3.1.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r017. AgSpec: Pushing the Limits of Retrieval-Based Speculative Decoding in Coding Agent Pipelines

- url: https://arxiv.org/abs/2610.01108
- authors: Sumin Lee; Sukmin Cho; Seungjae Lim; Youngjin Kwon
- date: 2026-10-01
- mechanism: We present AgSpec, a framework that supplies the corpus and draft-length policies that existing retrieval engines lack in coding-agent pipelines.
- regime: Named in the abstract: batch size 1, batch size 16.
- number: On two repository-level multi-agent coding benchmarks, AgSpec outperforms five retrieval-based drafters and EAGLE-3 in most evaluated settings, raising generation throughput over autoregressive decoding up to 4.37$\times$ at batch size 1 and 4.76$\times$ at batch size 16.
- quote: On two repository-level multi-agent coding benchmarks, AgSpec outperforms five retrieval-based drafters and EAGLE-3 in most evaluated settings, raising generation throughput over autoregressive decoding up to 4.37$\times$ at batch size 1 and 4.76$\times$ at batch size 16.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r018. SEED: Self-Speculative Decoding via Implicit Encoder-Decoder

- url: https://arxiv.org/abs/2609.36590
- authors: Hankun Lin; Patrick Pynadath; Ruqi Zhang
- date: 2026-09-29
- mechanism: We propose self-speculative encoder-decoder (SEED), a self-speculative method that obtains high-quality drafts cheaply by reusing the deep contextual representations already computed during verification.
- regime: The abstract does not name a device, batch, or model family.
- number: Experiments across multiple benchmarks show that SEED achieves up to 2.7$\times$ average speedup on 4B-scale models, outperforming both early-exit and MTP-style self-speculative baselines and running 28% faster than the state-of-the-art EAGLE-3, while preserving or even improving the generation quality of standard autoregressive fine-tuning.
- quote: Experiments across multiple benchmarks show that SEED achieves up to 2.7$\times$ average speedup on 4B-scale models, outperforming both early-exit and MTP-style self-speculative baselines and running 28% faster than the state-of-the-art EAGLE-3, while preserving or even improving the generation quality of standard autoregressive fine-tuning.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r019. RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding

- url: https://arxiv.org/abs/2609.21827
- authors: Qiao Hu; Yepeng Weng; Bo Zhang; Takehisa Yairi
- date: 2026-09-18
- mechanism: We resolve this by decoupling these roles: RheoSampling assigns a token sampled from the draft distribution a proxy probability for tree expansion and pruning alongside its true sampling probability for verification.
- regime: The abstract does not name a device, batch, or model family.
- number: Improvements in acceptance rate and speedup over state-of-the-art dynamic-tree methods. The abstract states no numeric factor.
- quote: Experiments across LLMs and benchmarks demonstrate improvements in acceptance rate and speedup over state-of-the-art dynamic tree methods.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): RheoSampling keeps stochastic sampling inside a dynamic draft tree. The abstract reports a higher acceptance rate and a speedup but no numeric factor, so it does not move the 10.8528 ms TPOT or say whether acceptance exceeds ~7.1. It does not change the ~3008 kernel count, the 13.58 us MIX floor, or the 5.35 ms TP reduce.

## r020. To Copy or Not to Copy: Controlling Speculative Decoding via Intrinsic Model Signals

- url: https://arxiv.org/abs/2609.20186
- authors: Roy Eisenstadt; Ido Cohen; Edo Cohen-Karlik; Lior Wolf; Itamar Zimerman
- date: 2026-09-17
- mechanism: We introduce SwitchSD, an adaptive framework that treats copying as a latent control signal of the LLM.
- regime: Named in the abstract: Llama, Qwen.
- number: Neural drafts (e.g., EAGLE3) provide robust performance across diverse text settings, while copy-based methods achieve higher speedups in copy-intensive regimes by generating candidates faster and exploiting long repetition spans for near-perfect speculation.
- quote: Neural drafts (e.g., EAGLE3) provide robust performance across diverse text settings, while copy-based methods achieve higher speedups in copy-intensive regimes by generating candidates faster and exploiting long repetition spans for near-perfect speculation.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r021. ResiSpec: Enhancing Multi-Candidate Speculative Sampling via Residual Distribution Shaping

- url: https://arxiv.org/abs/2608.24411
- authors: Zhi-Kai Chen; Jun-Jie Tao; Wei-Xiang Mao; De-Chuan Zhan; Han-Jia Ye
- date: 2026-08-25
- mechanism: To resolve this, we propose ResiSpec, a framework that strategically reforms the proposal distribution during verification to anchor the residual target mass within the draft model's high-confidence regions.
- regime: The abstract does not name a device, batch, or model family.
- number: By mathematically re-aligning the verification process without compromising output exactness, ResiSpec prevents candidate obsolescence and achieves up to 1.92$\times$ speedup over state-of-the-art multi-candidate methods.
- quote: By mathematically re-aligning the verification process without compromising output exactness, ResiSpec prevents candidate obsolescence and achieves up to 1.92$\times$ speedup over state-of-the-art multi-candidate methods.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r022. Oilbird: Training-Free Speculative Decoding with Keys the Verifier Already Computes

- url: https://arxiv.org/abs/2608.03839
- authors: Tao Jin; Phuong Minh Nguyen; Zhenzhu Yan; Teeradaj Racharak; Naoya Inoue
- date: 2026-08-04
- mechanism: We therefore propose a second, semantic draft source: the same pool, re-keyed by the hidden state the verifier has already computed at each committed token, together with a merge that lets it ride inside an existing lexical drafter's tree.
- regime: The abstract does not name a device, batch, or model family.
- number: Oilbird reaches 4.4x autoregressive decoding speed on API-Bank, against 3.9x for the strongest training-free baseline in our harness and 2.0x for EAGLE-3.
- quote: Oilbird reaches 4.4x autoregressive decoding speed on API-Bank, against 3.9x for the strongest training-free baseline in our harness and 2.0x for EAGLE-3.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r023. AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding

- url: https://arxiv.org/abs/2608.02989
- authors: Shuang Liang; Hao Mark Chen; Zhiwen Mo; Qianzhou Wang; Guoyu Li; Lingxiao Ma; Wayne Luk
- date: 2026-08-04
- mechanism: We introduce AcceptMoE, a verifier-side expert selector that combines target-router scores with offline-estimated commitment probabilities and automatically adjusts the number of eligible experts for each verification block, eliminating the need for a user-specified expert budget.
- regime: Named in the abstract: SGLang.
- number: Served with SGLang at batch size one, it reaches 1.290 times the throughput of this baseline with all expert weights in GPU memory, and 2.06 times under physical expert offloading, while reducing host-to-device traffic by 73.6 percent to 77.1 percent.
- quote: Served with SGLang at batch size one, it reaches 1.290 times the throughput of this baseline with all expert weights in GPU memory, and 2.06 times under physical expert offloading, while reducing host-to-device traffic by 73.6 percent to 77.1 percent.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r024. AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding

- url: https://arxiv.org/abs/2607.25852
- authors: Hong Liu; Rui Cen; Junhan Shi; Guangshuo Qin; Jiebin Zhang; Tianyu Liu; Runzhi Fan; Guoliang Zhao; Ruobing Xie; Kai Zhang; Song Liu; Guanghua Yu; Jianchen Zhu
- date: 2026-07-28
- mechanism: We present AngelSpec, a unified training framework for MTP and block-parallel speculative decoding that addresses this heterogeneity at three levels.
- regime: The abstract does not name a device, batch, or model family.
- number: Across the Hy3 series, DFly raises the average accepted length on Hy3-A21B by roughly 30% and attains the highest average throughput at every tested concurrency from 4 to 64, a 1.98-2.40x speedup over autoregressive decoding and 10.5-11.8% higher throughput than DFlash.
- quote: Across the Hy3 series, DFly raises the average accepted length on Hy3-A21B by roughly 30% and attains the highest average throughput at every tested concurrency from 4 to 64, a 1.98-2.40x speedup over autoregressive decoding and 10.5-11.8% higher throughput than DFlash.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r025. Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context

- url: https://arxiv.org/abs/2607.21535
- authors: Alagappan Valliappan
- date: 2026-07-23
- mechanism: We apply a StreamingLLM-style window plus attention sink to the draft only (Windowed-MTP); target verification remains full-context, so the target distribution is preserved.
- regime: Named in the abstract: SGLang, Qwen-35B.
- number: On three models from two hybrid families (Gated DeltaNet and Mamba2) at 1M context on one GPU in SGLang, it provides an acceptance-free 1.28-1.44x per-decode-step speedup over native MTP.
- quote: On three models from two hybrid families (Gated DeltaNet and Mamba2) at 1M context on one GPU in SGLang, it provides an acceptance-free 1.28-1.44x per-decode-step speedup over native MTP.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r026. SpecLA: Efficient Speculative Decoding for Linear-Attention Models

- url: https://arxiv.org/abs/2607.16673
- authors: Zhibin Wang; Xuying Han; Zhaohua Yang; Fuliang Liu; Xue Li; Rong Gu; Sheng Zhong; Chen Tian
- date: 2026-07-18
- mechanism: This paper presents SpecLA, a speculative decoding runtime for stateful linear-attention models.
- regime: Named in the abstract: H100.
- number: On an NVIDIA H100 with a public GDN-1.3B target, SpecLA achieves up to 1.70x end-to-end speedup over autoregressive decoding.
- quote: On an NVIDIA H100 with a public GDN-1.3B target, SpecLA achieves up to 1.70x end-to-end speedup over autoregressive decoding.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r027. Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding

- url: https://arxiv.org/abs/2609.02897
- authors: Oszkár Urbán; Young D. Kwon; Stylianos I. Venieris; Cecilia Mascolo
- date: 2026-07-03
- mechanism: We introduce AdaptiveSpec, a training-free per-step speculative decoding method that adapts both decisions from internal signals already produced during decoding.
- regime: Named in the abstract: SGLang, DeepSeek-R1-Distill-Llama-8B, Llama-3.1-8B-Instruct, Qwen3-8B.
- number: On SGLang, AdaptiveSpec improves throughput over EAGLE-3 by up to 56%, recovering 93% to fully lossless task accuracy on GSM8K, MATH-500, and HumanEval.
- quote: Implemented on the SGLang production-grade serving engine, AdaptiveSpec improves throughput over the state-of-the-art autoregressive speculative decoding method EAGLE-3 by up to 56%, recovering 93% to fully lossless task accuracy across GSM8K, MATH-500, and HumanEval on three target models (DeepSeek-R1-Distill-Llama-8B, Llama-3.1-8B-Instruct, Qwen3-8B).
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r028. SwiftSpec: Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding

- url: https://arxiv.org/abs/2506.11309
- authors: Ziyi Zhang; Ziheng Jiang; Chengquan Jiang; Menghan Yu; Size Zheng; Haibin Lin; Henry Hoffmann; Xin Liu
- date: 2025-06-12
- mechanism: This paper introduces SwiftSpec, a system that targets ultra-low latency for LLM decoding.
- regime: Named in the abstract: Llama3-70B.
- number: Across 5 model families and 6 datasets, SwiftSpec achieves an average of 1.75x speedup over state-of-the-art speculative decoding systems and, as a highlight, serves Llama3-70B at 348 tokens/s on 8 Nvidia Hopper GPUs, making it the fastest known system for low-latency LLM serving at this scale.
- quote: Across 5 model families and 6 datasets, SwiftSpec achieves an average of 1.75x speedup over state-of-the-art speculative decoding systems and, as a highlight, serves Llama3-70B at 348 tokens/s on 8 Nvidia Hopper GPUs, making it the fastest known system for low-latency LLM serving at this scale.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r029. StreamServe: Adaptive Speculative Flows for Low-Latency Disaggregated LLM Serving

- url: https://arxiv.org/abs/2604.09562
- authors: Satyam Kumar; Arpit Singh Gautam; Kailash Talreja; Saurabh Jha
- date: 2026-02-11
- mechanism: We introduce StreamServe, a disaggregated prefill decode serving architecture that combines metric aware routing across compute lanes with adaptive speculative decoding that tunes speculation depth online from runtime signals.
- regime: Named in the abstract: A800, vLLM.
- number: Across these workloads, StreamServe reduces latency by 11 to 18 times relative to tensor parallel vLLM baselines and reaches throughput up to 2235 tokens per second on summarization tasks.
- quote: Across these workloads, StreamServe reduces latency by 11 to 18 times relative to tensor parallel vLLM baselines and reaches throughput up to 2235 tokens per second on summarization tasks.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r030. EntMTP: Accelerating LLM Inference with Entropy Guided Multi Token Prediction

- url: https://arxiv.org/abs/2606.27550
- authors: Carrie Chen
- date: 2026-06-25
- mechanism: To address this, we propose Entropy-guided Multi-Token Prediction (EntMTP), a training-free scheduler that toggles between tree-based attention topologies from a set of task-specific pareto-optimal trees conditioned on a running estimate of local generation entropy.
- regime: The abstract does not name a device, batch, or model family.
- number: When evaluated across Humaneval, ShareGPT, GSM8k, and Litbench benchmarks, EntMTP consistently achieves a 1.15x speedup against Hydra and peak speedup of 1.36x against Medusa baselines respectively.
- quote: When evaluated across Humaneval, ShareGPT, GSM8k, and Litbench benchmarks, EntMTP consistently achieves a 1.15x speedup against Hydra and peak speedup of 1.36x against Medusa baselines respectively.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r031. Text Speeding Itself Up: Recency and Restraint in Session-Local Drafting

- url: https://arxiv.org/abs/2611.19455
- authors: Antal van den Bosch
- date: 2026-11-14
- mechanism: We compare a recency-weighted, backing-off, confidence-gated session memory against prompt lookup decoding in its reference form, which copies from the earliest earlier match, and in a variant that copies from the most recent one, with paired per-prompt bootstrap intervals throughout.
- regime: Named in the abstract: Mistral-7B.
- number: Under a repetition penalty, long-generation speedups shrink from 1.4-1.7x to 1.1-1.2x, but recency still helps and the length floor is the only drafter that saves energy in both domains.
- quote: Under a repetition penalty, long-generation speedups shrink from 1.4-1.7x to 1.1-1.2x, but recency still helps and the length floor is the only drafter that saves energy in both domains.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r032. Beneath the Tokens: A Performance Engineering Study of Multi-Token Prediction in GPU-Accelerated LLM Inference

- url: https://arxiv.org/abs/2609.35188
- authors: Suwesh Prasad Sah
- date: 2026-09-28
- mechanism: This study evaluates two-token multi-token prediction (MTP) against autoregressive decoding in a controlled single-request deployment on an NVIDIA A10G GPU.
- regime: Named in the abstract: A10G.
- number: One A10G, one request: output throughput 1.91x to 2.19x and time to first output down 10.0-14.2%. Median mean acceptance 2.370 to 2.595.
- quote: MTP increased output throughput by \(1.91\times\) to \(2.19\times\) across all prompts and reduced time to first output by 10.0--14.2\%.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): two-token MTP on one A10G request raises throughput 1.91x-2.19x, with mean acceptance 2.370-2.595, below ~7.1. That does not move the 10.8528 ms TPOT. It does not change the ~3008 kernel count, the 13.58 us MIX floor, or the 5.35 ms TP reduce.

## r033. DFlash: Block Diffusion for Flash Speculative Decoding

- url: https://arxiv.org/abs/2602.06036
- authors: Jian Chen; Yesheng Liang; Zhijian Liu
- date: 2026-02-05
- mechanism: In this paper, we introduce DFlash, a speculative decoding framework that employs a lightweight block diffusion model for parallel drafting.
- regime: The abstract does not name a device, batch, or model family.
- number: Experiments show that DFlash achieves over 6x lossless acceleration across a range of models and tasks, delivering up to 2.5x higher speedup than the state-of-the-art speculative decoding method EAGLE-3.
- quote: Experiments show that DFlash achieves over 6x lossless acceleration across a range of models and tasks, delivering up to 2.5x higher speedup than the state-of-the-art speculative decoding method EAGLE-3.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r034. EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test

- url: https://arxiv.org/abs/2503.01840
- authors: Yuhui Li; Fangyun Wei; Chao Zhang; Hongyang Zhang
- date: 2025-03-03
- mechanism: In this paper, we introduce EAGLE-3, which abandons feature prediction in favor of direct token prediction and replaces reliance on top-layer features with multi-layer feature fusion via a technique named training-time test.
- regime: Named in the abstract: SGLang, batch size of 64.
- number: The results show that EAGLE-3 achieves a speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2.
- quote: The results show that EAGLE-3 achieves a speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r035. Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding

- url: https://arxiv.org/abs/2609.36173
- authors: Haohui Zhang; Keyu Chen; Haocheng Sun; Weibo Gu; Ruizhi Qiao; Xing Sun; Bo Jiang
- date: 2026-09-28
- mechanism: We therefore propose DSpine, a drafter with causal conditioning injection throughout the backbone: at every layer, gated adjacent injection writes each predecessor's predicted feature into its successor, so the causal conditioning chain unfolds over network depth while all positions update in parallel.
- regime: Named in the abstract: SGLang, Qwen3-4B, Qwen3-8B., Qwen3-8B.
- number: Across seven math, code, and chat benchmarks, DSpine achieves the longest acceptance length at both temperatures on Qwen3-4B and Qwen3-8B.
- quote: Across seven math, code, and chat benchmarks, DSpine achieves the longest acceptance length at both temperatures on Qwen3-4B and Qwen3-8B.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r036. Speculative Pipeline Decoding: Higher-Accuracy Drafting with Hidden Latency via Pipeline Parallelism

- url: https://arxiv.org/abs/2605.30852
- authors: Yijiong Yu; Huazheng Wang; Shuai Yuan; Ruilong Ren; Ji Pei
- date: 2026-05-29
- mechanism: We propose Speculative Pipeline Decoding (SPD), which partitions the target LLM into $n$ pipeline stages so that $n$ tokens of a single sequence advance in parallel.
- regime: The abstract does not name a device, batch, or model family.
- number: Experiments show that SPD achieves higher theoretical and wall-clock speedup than EAGLE-3 at moderate pipeline width, while more aggressive widths still leave room for further gains.
- quote: Experiments show that SPD achieves higher theoretical and wall-clock speedup than EAGLE-3 at moderate pipeline width, while more aggressive widths still leave room for further gains.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r037. DynaDFlash: Dynamic Feature Projection for Block Diffusion Speculators

- url: https://arxiv.org/abs/2611.19085
- authors: Aadi Kulshrestha; Reese Chong
- date: 2026-11-14
- mechanism: We trained both a DFlash baseline for the Qwen 3 of block size 16, and a dynamic model that computes gates for each layer to affect the magnitude of hidden features before projection.
- regime: Named in the abstract: Qwen.
- number: We also found this lead to a 4.1% higher throughput for a subset of the same dataset in the context of speculative decoding.
- quote: We also found this lead to a 4.1% higher throughput for a subset of the same dataset in the context of speculative decoding.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r038. LoopSkim: Lossless Multi-Depth Speculative Decoding for Looped Language Models

- url: https://arxiv.org/abs/2611.19896
- authors: Muhan Xiao; Shijian Li
- date: 2026-11-15
- mechanism: We introduce LoopSkim, in which the depth of the pipeline is the verifier.
- regime: The abstract does not name a device, batch, or model family.
- number: Without training, its speedup over AR averages 1.73x that of LoopSpec across greedy decoding and sampling, and its checks between the first loops and full depth speed it up on every model.
- quote: Without training, its speedup over AR averages 1.73x that of LoopSpec across greedy decoding and sampling, and its checks between the first loops and full depth speed it up on every model.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r039. LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers

- url: https://arxiv.org/abs/2611.19649
- authors: Siqi Zhu; Suozhi Huang; Yi Zhao; Jiaxuan You
- date: 2026-11-14
- mechanism: We introduce LoopFill, a training-free speculative decoding algorithm for looped LLM that combines multi-token suffix drafts with shallow-loop predictions in a single draft tree.
- regime: The abstract does not name a device, batch, or model family.
- number: The speedups exceed those of LoopAhead, the strongest baseline, by \(40\%\) on Ouro-2.6B and \(25\%\) on Huginn-3.5B.
- quote: The speedups exceed those of LoopAhead, the strongest baseline, by \(40\%\) on Ouro-2.6B and \(25\%\) on Huginn-3.5B.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r040. Memory-Efficient Speculative Decoding with Quantized Draft KV Cache

- url: https://arxiv.org/abs/2611.20565
- authors: Saksham Gupta; Michael Li
- date: 2026-11-16
- mechanism: Hence, following KIVI, we propose K4V4 quantization with per-channel keys and per-token values.
- regime: Named in the abstract: Qwen.
- number: Mean acceptance drops from 57.93% to 57.14%, only 1.36% relative.
- quote: Mean acceptance drops from 57.93% to 57.14%, only 1.36% relative.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r041. D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding

- url: https://arxiv.org/abs/2611.06425
- authors: Baoyou Chen; Ruiqiao Mei; Guancong Lin; Tao Song; Weihao Yuan; Qun Liu; Siyu Zhu
- date: 2026-11-04
- mechanism: We introduce D3D, a standalone diffusion drafter that removes this inter-round dependency.
- regime: Named in the abstract: Qwen3-8B, Qwen3-32B.
- number: On GSM8K with Qwen3-32B, speedup reaches 7.48$\times$.
- quote: On GSM8K with Qwen3-32B, speedup reaches 7.48$\times$.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r042. RECAP: Context-Aware Prediction Recycling with Request-Local and Persistent Memories for Multimodal Speculative Decoding

- url: https://arxiv.org/abs/2611.05894
- authors: Hengjie Zhu; Dayan Wu; Xinze Liu; Peng Fu; Zheng Lin; Weiping Wang
- date: 2026-11-04
- mechanism: We introduce \method, which stores target predictions under one-, two-, and three-token keys: contextual memories live only for the current request and are thus implicitly conditioned on its image, while token-level memory persists across requests to retain coverage.
- regime: Named in the abstract: Qwen2.5-VL-7B.
- number: Without training or an auxiliary draft model, \method\ achieves mean speedups over autoregressive decoding of $1.66\speed$ and $1.56\speed$ across seven single-turn benchmarks on Qwen2.5-VL-7B and LLaVA-1.6-7B, exceeding the strongest training-free baseline, Token Recycling ($1.51\speed$ and $1.37\speed$).
- quote: Without training or an auxiliary draft model, \method\ achieves mean speedups over autoregressive decoding of $1.66\speed$ and $1.56\speed$ across seven single-turn benchmarks on Qwen2.5-VL-7B and LLaVA-1.6-7B, exceeding the strongest training-free baseline, Token Recycling ($1.51\speed$ and $1.37\speed$).
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a longer accepted draft banks more tokens per target forward. Our prompt-lookup acceptance is already ~7.1, so this only cuts the 10.8528 ms/token if verification does not add launches on top of ~3008 kernels/round. The source's figure is not a measurement on this Ascend path.

## r043. Fuse-All MoE Inference Kernel with Load Balancing on TPUs

- url: https://arxiv.org/abs/2611.17857
- authors: Rupeng Liu; Bangsheng Tang; Zongwei Zhou; Chengji Yao; Yejin Lee
- date: 2026-11-13
- mechanism: To address these bottlenecks, we introduce Fuse-All MoE, a TPU kernel design that jointly optimizes expert computation, output communication, and load balancing.
- regime: The abstract does not name a device, batch, or model family.
- number: Across 32-512 decode tokens, symmetric EP8TP2 reduces kernel latency by up to 21.3% relative to fused EP16.
- quote: Across 32-512 decode tokens, symmetric EP8TP2 reduces kernel latency by up to 21.3% relative to fused EP16.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): Fuse-All is a TPU MoE kernel that fuses expert compute, output communication, and load balance. A 21.3% kernel-latency cut at 32-512 decode tokens does not move the 10.8528 ms TPOT. It does not change the ~3008 kernel count, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r044. VelServe: Taming Decode Jitters in LLM Inference with Tail-Calibrated Scheduling

- url: https://arxiv.org/abs/2611.18392
- authors: Ishaan Datta; Alind Khare; Koushik Kumar Dey; Ratnesh Jamidar; Anand Kumar Singh
- date: 2026-11-13
- mechanism: We present \sysname, a scheduler that calibrates how much prefill work each decode step carries using the measured tail of its own iteration latency.
- regime: Named in the abstract: Qwen3-8B, DeepSeek-V4-Flash, DeepSeek-V2-Lite..
- number: On three attention families and varying workloads, \sysname sustains 2.4--2.9$\times$ the SLO goodput of the strongest baseline on Qwen3-8B and DeepSeek-V4-Flash, and at least 4.4$\times$ on DeepSeek-V2-Lite.
- quote: On three attention families and varying workloads, \sysname sustains 2.4--2.9$\times$ the SLO goodput of the strongest baseline on Qwen3-8B and DeepSeek-V4-Flash, and at least 4.4$\times$ on DeepSeek-V2-Lite.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this is a scheduler that sizes the prefill admitted into each decode step from that step's latency tail. The 2.4-2.9x figure is SLO goodput, not a shorter token, so it does not move the 10.8528 ms TPOT. It does not change the ~3008 kernel count, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r045. FlashOffloading: Low-Latency KV Cache Offloading for Long-Context LLM Serving

- url: https://arxiv.org/abs/2611.07542
- authors: Xuan Wang; Zhuanhao Wu; Tingzhu Fang; Xushi Li; Sicheng Zhang; Zhenyu Yang; Feng Liu; Zhaohui Fan; Qiang Xie; Huayi Zou; Yuncong Liu; Zeyu Song; Ke Zhu; Yanjia Zeng; Jingchun Gao; Yan Shi; Kui Luo; Gezi Li; Liangjun Feng; Cheng Li
- date: 2026-11-05
- mechanism: We present FlashOffloading, a host-DRAM-centric inference architecture that fully offloads the KV cache in both the prefill and decode phases.
- regime: GLM-5.2 with MTP3 speculative decoding, Ascend910C, prefill-decode disaggregated. The constraint is a 30 ms TPOT SLO.
- number: GLM-5.2 with MTP3 on Ascend910C PD-disaggregation: prefill throughput up to 1.38x and decode throughput up to 1.91x under a 30 ms TPOT constraint.
- quote: In experiments deploying GLM-5.2 with multi-token prediction (MTP3) speculative decoding on an Ascend910C prefill--decode (PD)-disaggregated platform, FlashOffloading improves prefill throughput by up to 1.38\(\times\) and decode throughput by up to 1.91\(\times\) under a 30 ms TPOT constraint.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this is KV offload, not a longer draft. Decode throughput rises 1.91x only under a 30 ms TPOT SLO on Ascend910C, almost 3x our 10.8528 ms and on a different PD stack, so it does not move this baseline. The ~3008 kernel count, 13.58 us MIX floor, ~7.1 acceptance, and 5.35 ms TP reduce are untouched.

## r046. MegaOps: A Kernel Overlapping Framework with PyTorch-like Modularity and Megakernel Performance

- url: https://arxiv.org/abs/2611.16248
- authors: Jinkun Lin; Byungsoo Oh; Rachee Singh
- date: 2026-11-12
- mechanism: We develop MegaOps, a framework that launches dependent kernels on separate CUDA streams and enforces each dependency on individual tiles of the intermediate tensor that the producer kernel writes and the consumer kernel reads.
- regime: Named in the abstract: Llama3, Qwen3, H100, A100, L40.
- number: 1.09x-1.10x decode throughput and 0.98x-1.10x prefill throughput versus the best baseline, matching hand-tuned ThunderKittens megakernels.
- quote: Our MegaOps framework matches the throughput of hand-tuned ThunderKittens megakernels, and achieves 1.09$\times$-1.10$\times$ the decode throughput and 0.98$\times$-1.10$\times$ the prefill throughput of the best-performing baseline, while reusing existing Triton, cuBLAS, and cuDNN kernels.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): MegaOps keeps separate kernels and overlaps them by tile, so it does not cut the ~3008 launch count. A 1.09x-1.10x decode-throughput gain on H100, A100, and L40 does not move the 10.8528 ms TPOT. The 13.58 us MIX floor, ~7.1 acceptance, and 5.35 ms TP reduce are a different path.

## r047. Understanding GPU Megakernels: Why Residency Can Mislead

- url: https://arxiv.org/abs/2611.00495
- authors: Xinwei Qiang; Ziyi Xu; Chang Chen; Zaifeng Pan; Zhongkai Yu; Yue Guan; Yufei Ding
- date: 2026-10-31
- mechanism: We study the benefits and costs of these capabilities by reconstructing two LLM megakernels, ThunderKittens (TK) and VDCores, as separate kernels on H100 using their original operator code.
- regime: Named in the abstract: H100.
- number: By overlapping startup and operand loading with programmatic dependent launch (PDL), the reconstructions reduce decode-step latency by 5.7% in TK and achieve comparable performance in VDCores when both executions enforce the required memory ordering.
- quote: By overlapping startup and operand loading with programmatic dependent launch (PDL), the reconstructions reduce decode-step latency by 5.7% in TK and achieve comparable performance in VDCores when both executions enforce the required memory ordering.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this cuts kernel launches on the decode step. Our round pays ~3008 launches and an 11.5 ms floor, with MIX_AIC at 13.58 us, so fewer launches are in scope. The source's GPU and launch path are not this A3 MIX kernel.

## r048. MegaFlux: Skew-Resilient MoE Megakernels via Pipelined Expert Replication

- url: https://arxiv.org/abs/2610.00671
- authors: Jianzhu Yao; Siva Kumar Sastry Hari; Vignesh Balaji; Sana Damani; Insu Jang; Pramod Viswanath; Christos Kozyrakis
- date: 2026-09-30
- mechanism: We present MegaFlux, which makes expert replication a runtime decision and pipelines the communication induced by replication within persistent MoE execution.
- regime: Named in the abstract: TensorRT-LLM, B200, vLLM, DeepSeek-V4-Pro.
- number: Integrated into vLLM for DeepSeek-V4-Pro prefill, MegaFlux delivers $1.13$--$1.26\times$ median end-to-end speedups over fixed placement.
- quote: Integrated into vLLM for DeepSeek-V4-Pro prefill, MegaFlux delivers $1.13$--$1.26\times$ median end-to-end speedups over fixed placement.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this cuts kernel launches on the decode step. Our round pays ~3008 launches and an 11.5 ms floor, with MIX_AIC at 13.58 us, so fewer launches are in scope. The source's GPU and launch path are not this A3 MIX kernel.

## r049. Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap

- url: https://arxiv.org/abs/2609.21483
- authors: Ziyu Huang; Yangjie Zhou; Chenhao Zhu; Peng Yu; Zihan Liu; Jinyu Liu; Shulai Zhang; Xingxun Tang; Hongzhe Yan; Xinhao Luo; Minyi Guo; Xiu Lin; Yinghao Yu; Guodong Yang; Liping Zhang; Shixuan Sun; Jingwen Leng
- date: 2026-09-18
- mechanism: We present Weave, to our knowledge the first MoE overlap system that performs fine-grained dynamic SM scheduling - deciding per layer and per GPU by routing results at runtime.
- regime: Named in the abstract: H100.
- number: On 4x H100 SXM GPUs across six mainstream MoE models, Weave achieves a 2.89x geometric-mean MoE-layer speedup and a 1.33x geometric-mean end-to-end speedup over five state-of-the-art baselines.
- quote: On 4x H100 SXM GPUs across six mainstream MoE models, Weave achieves a 2.89x geometric-mean MoE-layer speedup and a 1.33x geometric-mean end-to-end speedup over five state-of-the-art baselines.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this cuts kernel launches on the decode step. Our round pays ~3008 launches and an 11.5 ms floor, with MIX_AIC at 13.58 us, so fewer launches are in scope. The source's GPU and launch path are not this A3 MIX kernel.

## r050. mKernel: Fast Multi-GPU, Multi-Node Fused Kernels

- url: https://arxiv.org/abs/2609.13585
- authors: Ziming Mao; Yihan Zhang; Shawn Wei Chew; Shuang Ma; Costin Raiciu; Yang Zhou; Scott Shenker; Ion Stoica
- date: 2026-09-11
- mechanism: We present mKernel, a library of multi-GPU, multi-node fused kernels that overlap computation, intra-node NVLink communication, and inter-node RDMA at tile granularity.
- regime: Named in the abstract: H200.
- number: On two 16-GPU H200 clusters, mKernel achieves speedups of up to 1.72x on GEMM+AllReduce and $1.88\times$ on Ring Attention.
- quote: On two 16-GPU H200 clusters, mKernel achieves speedups of up to 1.72x on GEMM+AllReduce and $1.88\times$ on Ring Attention.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this cuts kernel launches on the decode step. Our round pays ~3008 launches and an 11.5 ms floor, with MIX_AIC at 13.58 us, so fewer launches are in scope. The source's GPU and launch path are not this A3 MIX kernel.

## r051. Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference

- url: https://arxiv.org/abs/2605.11581
- authors: Wenxin Dong; Mingqing Hu; Guanghui Yu; Qiang Fu; Peng Xu; Hui Xu; Yue Xing; Xuewu Jiao; Shuanglong Li; Lin Liu
- date: 2026-05-12
- mechanism: Building on this insight, we propose Ada-MK: (1) a three-dimensional shared-memory constraint model combined with K-dimension splitting that reduces peak shared memory usage by 50%; (2) MLIR-based fine-grained DAG offline search that solidifies the optimal execution path, completely eliminating runtime branching; and (3) a heterogeneous hybrid inference engine that embeds MegaKernel as a plugin into TensorRT-LLM, combining high-throughput Prefill with low-latency Decode.
- regime: Named in the abstract: TensorRT-LLM, L20, vLLM.
- number: On an NVIDIA L20, Ada-MK improves single-batch throughput by up to 23.6% over vanilla TensorRT-LLM and 50.2% over vLLM, achieving positive gains across all tested scenarios--the first industrial deployment of MegaKernel in a commercial online advertising system.
- quote: On an NVIDIA L20, Ada-MK improves single-batch throughput by up to 23.6% over vanilla TensorRT-LLM and 50.2% over vLLM, achieving positive gains across all tested scenarios--the first industrial deployment of MegaKernel in a commercial online advertising system.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this cuts kernel launches on the decode step. Our round pays ~3008 launches and an 11.5 ms floor, with MIX_AIC at 13.58 us, so fewer launches are in scope. The source's GPU and launch path are not this A3 MIX kernel.

## r052. AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis

- url: https://arxiv.org/abs/2606.09682
- authors: Jaber Jaber; Osama Jaber
- date: 2026-06-08
- mechanism: AutoMegaKernel (AMK) compiles a HuggingFace Llama-family model into a single persistent cooperative CUDA kernel that runs the whole forward pass in one launch, with no per-model hand-written CUDA.
- regime: Named in the abstract: Llama-family, L4, L40S, A10G, RTX 5090, A100, H100, Llama-1.1B..
- number: Batch-1 W8A16 megakernel versus CUDA-graphed cuBLAS bf16: L4 up to 1.33x, L40S 1.25-1.27x, A10G up to 1.08x, RTX 5090 1.19-1.23x.
- quote: A search-found int8 (W8A16) megakernel beats CUDA-graphed cuBLAS bf16 at batch-1 decode across NVIDIA's datacenter inference fleet: L4 up to 1.33x, the current-gen L40S 1.25-1.27x, A10G up to 1.08x at scale, and the consumer RTX 5090 1.19-1.23x.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this is one persistent kernel for a whole forward pass. Up to 1.33x at batch-1 decode versus CUDA-graphed cuBLAS bf16 is on L4 through RTX 5090, not this A3 path, so it does not move the 10.8528 ms TPOT. It does not change the ~3008 kernel count on this stack, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r053. CoRun: Padding is Simple and Efficient for Deterministic LLM Inference

- url: https://arxiv.org/abs/2608.14376
- authors: Shiju Zhao; Jiacheng Yang; Qihang Chen; Junhao Hu; Jiaqi Zheng; Guihai Chen; Xusheng Chen
- date: 2026-08-14
- mechanism: Leveraging this property, we present CoRun, a scheduling-based system that achieves deterministic inference without requiring batch invariance.
- regime: Named in the abstract: Qwen, DeepSeek.
- number: Throughput +15-324% over batch-invariant kernels, TTFT -51.8%, TPOT -48.6% on average.
- quote: Experiments on LLMs with diverse architectures, including Qwen and DeepSeek, show that CoRun ensures determinism while improving throughput by 15-324 % over batch-invariant approaches, reducing time-to-first-token by 51.8 % and time-per-output-token by 48.6 % on average.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): CoRun is fixed-shape batched decode plus isolated prefill, a determinism schedule, not a launch cut and not a KV offload. Its average TPOT drop of 48.6% is against batch-invariant kernels on Qwen and DeepSeek, so it does not move the 10.8528 ms TPOT. The ~3008 kernel count, 13.58 us MIX floor, ~7.1 acceptance, and 5.35 ms TP reduce stay as measured here.

## r054. Hybrid JIT-CUDA Graph Optimization for Low-Latency Large Language Model Inference

- url: https://arxiv.org/abs/2604.23467
- authors: Divakar Kumar Yadav; Tian Zhao
- date: 2026-04-25
- mechanism: This paper presents a hybrid runtime framework that combines Just-In-Time (JIT) compilation with CUDA Graph execution to reduce launch overhead while preserving runtime flexibility during autoregressive decoding.
- regime: Named in the abstract: TensorRT-LLM.
- number: Experimental results show that the hybrid runtime reduces Time-to-First-Token (TTFT) by up to 66.0% and achieves lower P99 latency compared with TensorRT-LLM in this regime.
- quote: Experimental results show that the hybrid runtime reduces Time-to-First-Token (TTFT) by up to 66.0% and achieves lower P99 latency compared with TensorRT-LLM in this regime.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this cuts kernel launches on the decode step. Our round pays ~3008 launches and an 11.5 ms floor, with MIX_AIC at 13.58 us, so fewer launches are in scope. The source's GPU and launch path are not this A3 MIX kernel.

## r055. flash-comm: Self-Synchronizing Collective Communication for LLM Decoding

- url: https://arxiv.org/abs/2611.06634
- authors: Carrie Wang; Jack Li; Jason Zhao
- date: 2026-11-04
- mechanism: We present flash-comm, a communication library that achieves efficient transport for LLM decoding by fusing readiness synchronization with data movement.
- regime: The abstract does not name a device, batch, or model family.
- number: In end-to-end decoding, flash-comm reduces time per output token (TPOT) by 1.7-6.8%.
- quote: In end-to-end decoding, flash-comm reduces time per output token (TPOT) by 1.7-6.8%.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this sits on the collective in the decode step. Our TP reduce is 5.35 ms and 100% serial, so a shorter collective would move that term only on the interconnect the source measured.

## r056. Understanding and Improving Communication Performance in Multi-node LLM Inference

- url: https://arxiv.org/abs/2511.09557
- authors: Prajwal Singhania; Siddharth Singh; Lannie Dalton Hough; Akarsh Srivastava; Harshitha Menon; Charles Fredrick Jekel; Abhinav Bhatele
- date: 2025-11-12
- mechanism: In this work, we present a detailed performance study of multi-node distributed inference using LLMs on GPU-based supercomputers.
- regime: Named in the abstract: Llama.
- number: NVRAR achieves up to 1.9$\times$-3.6$\times$ lower latency than NCCL for message sizes between 128 KB and 2 MB on HPE Slingshot and InfiniBand interconnects.
- quote: NVRAR achieves up to 1.9$\times$-3.6$\times$ lower latency than NCCL for message sizes between 128 KB and 2 MB on HPE Slingshot and InfiniBand interconnects.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this sits on the collective in the decode step. Our TP reduce is 5.35 ms and 100% serial, so a shorter collective would move that term only on the interconnect the source measured.

## r057. ReKV: Overlapping Quantized KV Transfer and Suffix Recomputation

- url: https://arxiv.org/abs/2611.08770
- authors: Guangjian Dong
- date: 2026-11-05
- mechanism: ReKV transfers a quantized prefix KV cache and one BF16 boundary state, then reconstructs a contiguous suffix while the prefix is in flight.
- regime: Named in the abstract: H100, Qwen2.5-7B, Qwen2.5-3B, Qwen3-8B..
- number: In a model-shaped synthetic benchmark on cross-node H100s, the Q8 configuration meeting separately evaluated quality criteria reduces 8K handoff latency by $2.36\times$, $2.84\times$, and $3.65\times$ over BF16 transfer for Qwen2.5-7B, Qwen2.5-3B, and Qwen3-8B.
- quote: In a model-shaped synthetic benchmark on cross-node H100s, the Q8 configuration meeting separately evaluated quality criteria reduces 8K handoff latency by $2.36\times$, $2.84\times$, and $3.65\times$ over BF16 transfer for Qwen2.5-7B, Qwen2.5-3B, and Qwen3-8B.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): ReKV overlaps a quantized KV handoff with suffix recomputation before the first decode token. The 2.36x-3.65x cut is 8K handoff latency on cross-node H100s, not the steady 10.8528 ms TPOT, so it does not move that figure. It does not change the ~3008 kernel count, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r058. FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling

- url: https://arxiv.org/abs/2504.03775
- authors: Weiqing Li; Guochao Jiang; Xiangyong Ding; Zhangcheng Tao; Chuzhan Hao; Chenfeng Xu; Yuewei Zhang; Hao Wang
- date: 2025-04-03
- mechanism: In this paper, we propose FlowKV, a novel disaggregated inference framework, which reduces the average transmission latency of KV cache by 96%, from 0.944s to 0.053s, almost eliminating the transfer time relative to the total request latency by optimizing the KV cache transfer.
- regime: The abstract does not name a device, batch, or model family.
- number: In this paper, we propose FlowKV, a novel disaggregated inference framework, which reduces the average transmission latency of KV cache by 96%, from 0.944s to 0.053s, almost eliminating the transfer time relative to the total request latency by optimizing the KV cache transfer.
- quote: In this paper, we propose FlowKV, a novel disaggregated inference framework, which reduces the average transmission latency of KV cache by 96%, from 0.944s to 0.053s, almost eliminating the transfer time relative to the total request latency by optimizing the KV cache transfer.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this moves KV or prefill/decode placement. It changes the 10.8528 ms TPOT only when the transfer or the attention read stays on the token critical path, which is the source's regime, not this profile.

## r059. FastAFD: When Does Attention-FFN Disaggregation Pay on Rack-Scale NVLink?

- url: https://arxiv.org/abs/2611.07149
- authors: Yichao Fu; Yuxuan Zhang; Minshen Zhang; Ruitian Wang; Junda Chen; Hao Zhang
- date: 2026-11-04
- mechanism: We build and tune FastAFD, an AFD engine for GB200 NVL72 that overlaps the MoE of one microbatch with the attention of the other, and compare it against the best vLLM wide EP setting.
- regime: Named in the abstract: GB200, vLLM, Qwen3, Qwen3.5.
- number: AFD throughput up to 1.17x over wide expert parallelism on Qwen3 and 1.16x on Qwen3.5, matching it on MiniMax-M2.5.
- quote: Experiments show that AFD has a throughput advantage of up to 1.17x over wide EP on Qwen3 and 1.16x on Qwen3.5, and matches it on MiniMax-M2.5.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this overlaps MoE with attention by splitting them across GPUs on GB200 NVL72. Up to 1.17x decode throughput versus wide expert parallelism does not move the 10.8528 ms TPOT. It is not a longer draft, so the ~7.1 acceptance is not the lever, and it does not change the ~3008 kernel count, the 13.58 us MIX floor, or the 5.35 ms TP reduce.

## r060. Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

- url: https://arxiv.org/abs/2509.16495
- authors: Mert Hidayetoglu; Aurick Qiao; Michael Wyatt; Jeff Rasley; Yuxiong He; Samyam Rajbhandari
- date: 2025-09-20
- mechanism: We adapt SP to inference, and combine it with TP to get the best of both worlds. Our solution: Shift Parallelism. Shift Parallelism dynamically switches across TP and SP, and minimizes latency in low traffic without losing throughput in high traffic.
- regime: The abstract does not name a device, batch, or model family.
- number: The efficient GPU communications of Shift Parallelism yields up to i) 1.51x faster response in interactive workloads and ii) 50% higher throughput in batch workloads, compared to a TP-only solution.
- quote: The efficient GPU communications of Shift Parallelism yields up to i) 1.51x faster response in interactive workloads and ii) 50% higher throughput in batch workloads, compared to a TP-only solution.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): Shift Parallelism switches between tensor and sequence parallelism as load changes. Up to 1.51x faster response versus TP-only does not move the 10.8528 ms TPOT. It does not change the ~3008 kernel count, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r061. AFORE: Attention-FFN Disaggregation with Overlapped Reconfiguration of Experts

- url: https://arxiv.org/abs/2610.03203
- authors: Wenshuang Li; Youhe Jiang; You Peng; Jiawei Jiang; Binhang Yuan
- date: 2026-10-02
- mechanism: To solve this problem, we present AFORE, a timely expert reconfiguration system for AFD-based MoE serving.
- regime: The abstract does not name a device, batch, or model family.
- number: Compared with static placement, AFORE improves throughput by 29.8% on average and reduces P95 inter-token latency by 18.2% on average.
- quote: Compared with static placement, AFORE improves throughput by 29.8% on average and reduces P95 inter-token latency by 18.2% on average.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this sits on the collective in the decode step. Our TP reduce is 5.35 ms and 100% serial, so a shorter collective would move that term only on the interconnect the source measured.

## r062. FOSCAR: Accelerating Decode Attention over 2- and 4-Bit KV Caches through Affine Folding

- url: https://arxiv.org/abs/2611.09722
- authors: Donglin Zhuang; Pragaash Ponnusamy; Xiaoxia Wu; Shuaiwen Leon Song; Zhongzhu Zhou
- date: 2026-11-06
- mechanism: Instead of applying scale and offset to each KV value, FOSCAR uses affine folding to apply them to the other operand or the output of decode-attention matrix products, depending on the quantization scheme (token-wise or channel-wise).
- regime: Named in the abstract: H100, B200, TensorRT-LLM, SGLang.
- number: H100 geo-mean INT2 and INT4 speedups 3.36x and 2.83x over FlashInfer BF16, and 2.03x and 1.71x over TensorRT-LLM FP8.
- quote: On H100, FOSCAR INT2 and INT4 kernels achieve geo-mean speedups of $3.36\times$ and $2.83\times$ over FlashInfer BF16, and $2.03\times$ and $1.71\times$ over TensorRT-LLM FP8.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): FOSCAR folds 2-bit and 4-bit KV scales into the decode-attention matmul. The 3.36x and 2.83x kernel speedups are on H100 versus FlashInfer BF16, so they do not move the 10.8528 ms TPOT or the A3 W4A8 path. They do not change the ~3008 kernel count, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r063. BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm-System Co-Design

- url: https://arxiv.org/abs/2611.09825
- authors: Dayou Du; Woon Yee Ng; Guangming Sheng; Nick Lau; Jianyi Cheng; Luo Mai
- date: 2026-11-06
- mechanism: We present BitDecoding-FP4, an algorithm--system co-design for accurate and efficient native FP4 KV decoding.
- regime: End-to-end serving on NVIDIA B300. Models named in the abstract: Qwen3-4B, Qwen3-30B-A3B.
- number: On NVIDIA B300, Qwen3-4B and Qwen3-30B-A3B, 2.4-3.0x higher peak serving throughput than BF16 KV.
- quote: In end-to-end serving on NVIDIA B300 with Qwen3-4B and Qwen3-30B-A3B, it delivers 2.4-3.0$\times$ higher peak serving throughput than BF16 KV.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this changes how many bytes the decode step moves. Our 10.8528 ms TPOT already includes the profiled W4A8 kernels; a foreign quant kernel's speedup does not transfer onto the A3 path.

## r064. TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference

- url: https://arxiv.org/abs/2611.07256
- authors: Hanzhi Zhang; Qiao Zhang; Qinglei Cao; Heng Fan; Yan Huang; Kewei Sha; Yunhe Feng
- date: 2026-11-05
- mechanism: We therefore introduce TriAttention, which assigns query-key (QK) tiles to skipping, 4-bit, or 8-bit computation, using 4-bit to retain quantization-tolerant interactions at lower cost.
- regime: The abstract does not name a device, batch, or model family.
- number: Experiments show that TriAttention achieves end-to-end generation speedups of up to 7.88x over the Torch attention and 2.65x over the BF16 baseline.
- quote: Experiments show that TriAttention achieves end-to-end generation speedups of up to 7.88x over the Torch attention and 2.65x over the BF16 baseline.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this changes how many bytes the decode step moves. Our 10.8528 ms TPOT already includes the profiled W4A8 kernels; a foreign quant kernel's speedup does not transfer onto the A3 path.

## r065. Coverage-Verified Sparse Attention: Closed-Loop Quality Control for Long-Context LLM Decoding

- url: https://arxiv.org/abs/2611.10634
- authors: Yan-Cheng Li; Chun-Yi Lee
- date: 2026-11-07
- mechanism: Coverage-Verified Sparse Attention (CVSA) turns coverage into a training-free, closed-loop verify-then-correct decode loop that accepts high-coverage drafts and recovers only the heads that fall short, with no sampling, no draft model, and no speculative-decoding infrastructure.
- regime: The abstract does not name a device, batch, or model family.
- number: On LongBench at 7B and 70B, CVSA comes within $1.13$ points of dense attention, statistically matching it at 70B, and it loses $0.8$ points on RULER; measured median decode-step speedup reaches $6.21\times$ at 128K context.
- quote: On LongBench at 7B and 70B, CVSA comes within $1.13$ points of dense attention, statistically matching it at 70B, and it loses $0.8$ points on RULER; measured median decode-step speedup reaches $6.21\times$ at 128K context.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): CVSA is sparse decode attention with a coverage check, and the abstract says it uses no draft model. A 6.21x median decode-step speedup at 128K context does not move the 10.8528 ms TPOT. It does not change the ~3008 kernel count, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r066. MOIRA: Mass-Oriented Indexing with Ragged Attention for Long-Context Decoding

- url: https://arxiv.org/abs/2610.04313
- authors: Dich Nhat Minh Nguyen; Tran Dang Duong Nguyen
- date: 2026-10-03
- mechanism: We propose MOIRA, a training-free sparse decode path in vLLM whose budget adapts per KV head and per layer.
- regime: Named in the abstract: vLLM, H200.
- number: On an H200, at RULER's 128k context, MOIRA with $γ=0.99$ matches dense accuracy while reading about 30% of the pages and reduces the time per output token (TPOT) by 2.2-2.5$\times$ relative to dense FA3; with $γ=0.98$ it reduces TPOT by 2.7$\times$ and stays within the noise of dense.
- quote: On an H200, at RULER's 128k context, MOIRA with $γ=0.99$ matches dense accuracy while reading about 30% of the pages and reduces the time per output token (TPOT) by 2.2-2.5$\times$ relative to dense FA3; with $γ=0.98$ it reduces TPOT by 2.7$\times$ and stays within the noise of dense.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): MOIRA reads fewer KV pages on a sparse decode path. A 2.2-2.5x TPOT cut versus dense attention at 128k on an H200 does not move the 10.8528 ms TPOT. It is not a cut of the ~3008 launches, and it does not change the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r067. PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention

- url: https://arxiv.org/abs/2607.24593
- authors: Hong Liu; Yuan Cheng; Lin Niu; Yi Su; Yufei Xue; Anmin Liu; Guanghua Yu; Jianchen Zhu
- date: 2026-07-27
- mechanism: We exploit these properties in PIVOT, Proxy Indexing Via One full-prefix Traversal, a training-free, drop-in replacement for the DSA indexer that shares one prefix scan across a group of nearby queries.
- regime: Named in the abstract: DeepSeek, DeepSeek-V3.2, GLM-5.1.
- number: On DeepSeek-V3.2 and GLM-5.1 across LongBench and RULER, PIVOT matches the accuracy of the dense DSA indexer while accelerating it by up to 4x and reducing end-to-end latency by up to 1.6x at long context.
- quote: On DeepSeek-V3.2 and GLM-5.1 across LongBench and RULER, PIVOT matches the accuracy of the dense DSA indexer while accelerating it by up to 4x and reducing end-to-end latency by up to 1.6x at long context.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): PIVOT replaces the sparse-attention indexer with one shared prefix scan. Up to 1.6x lower end-to-end latency at long context does not move the 10.8528 ms TPOT. It is not a longer draft, so the ~7.1 acceptance is not the lever, and it does not change the ~3008 kernel count, the 13.58 us MIX floor, or the 5.35 ms TP reduce.

## r068. Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding

- url: https://arxiv.org/abs/2609.00097
- authors: Zhigeng Liu; Zhiyuan Ning; Ruixiao Li; Xiaoran Liu; Yuerong Song; Min Zhang; Ziwei He; Xipeng Qiu
- date: 2026-08-31
- mechanism: To overcome the inherent trade-offs between the memory overhead of metadata-based metrics and the computational inefficiency of adaptive selection strategies, we present Faster Flash Decoding (FFD), a novel hardware-algorithm co-design framework designed to break the memory wall in long-context decoding.
- regime: The abstract does not name a device, batch, or model family.
- number: Offering a training-free and plug-and-play solution, FFD also enables the reuse of scanning results for computation, achieving up to 11.6x kernel-level speedup and scaling to 256K context length, with 2.37x end-to-end throughput improvement.
- quote: Offering a training-free and plug-and-play solution, FFD also enables the reuse of scanning results for computation, achieving up to 11.6x kernel-level speedup and scaling to 256K context length, with 2.37x end-to-end throughput improvement.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): FFD is sparse long-context decode attention, not a quant kernel. Up to 11.6x kernel speedup and 2.37x end-to-end throughput do not move the 10.8528 ms TPOT. They do not change the ~3008 kernel count, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r069. AVSG: Accelerated Vectorized Sparse Gather for Efficient KV Cache Offload in Sparse-Attention LLM Serving

- url: https://arxiv.org/abs/2609.37538
- authors: Wenwei Kuang; Xiangyu Wang; Chong Wu; Jun Wang; Weijie Zhang; Brian K Chen; Longwen Lan; Ken Zhang
- date: 2026-09-29
- mechanism: We present AVSG, an operator that addresses these challenges for DSA while preserving its exact token selections.
- regime: Single NPU. End-to-end on a serving stack with a real-world production dataset. Operator figures on that NPU: vectorized hash matching 2.80x versus scalar matching, an 8K-entry buffer at 94.83% hit rate. The abstract does not name the NPU SKU.
- number: On an NPU serving stack, TPOT -39% and output throughput 1.27x versus the same KV offload without HBM-buffer reuse.
- quote: These gains translate into end-to-end improvement: on a serving stack processing a real-world production dataset, AVSG reduces time per output token by 39% and increases output throughput by 1.27x relative to the same KV offload layout without HBM-buffer reuse.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this is KV offload with device-buffer reuse, not a longer draft. A 39% TPOT cut and 1.27x output throughput are on a single-NPU serving stack, so they do not move the 10.8528 ms figure, the ~3008 kernel count, the 13.58 us MIX floor, the ~7.1 acceptance, or the 5.35 ms TP reduce.

## r070. UltraQuant: 4-bit KV Caching for Context-Heavy Agents

- url: https://arxiv.org/abs/2606.20474
- authors: Inesh Chakrabarti; David Limpus; Aditi Ghai Rana; Bowen Bao; Spandan Tiwari; Thiago Crepaldi; Ashish Sirasao
- date: 2026-06-18
- mechanism: Third, we present serving optimizations on AMD GPUs, including optimized decode-attention kernels and UltraQuant, an FP4 approximation path that uses FP8 queries, FP4 KV tensors, UE8M0 group scales, and native scaled-MFMA support on CDNA4.
- regime: Named in the abstract: vLLM, Qwen3-235B.
- number: 2.71x (MiniMax-M2.5) and 4.38x (Qwen3-235B) qualified-request throughput versus BF16, at half the KV bytes.
- quote: On an adaptive-SLO replay of production Claude Code traces, UltraQuant sustains 2.71x (MiniMax-M2.5) and 4.38x (Qwen3-235B) the qualified-request throughput of the BF16 baseline, matching or exceeding hardware FP8 KV while using half the KV bytes.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): this changes how many bytes the decode step moves. Our 10.8528 ms TPOT already includes the profiled W4A8 kernels; a foreign quant kernel's speedup does not transfer onto the A3 path.

## r071. Speculative decoding plus concurrency on one DGX Spark

- url: https://x.com/JRB_AI/status/2122957227229139227
- handle: @JRB_AI
- authors: JRB
- date: 2026-11-18
- mechanism: A small drafter guesses the next tokens; the main model checks every guess in the same weight read, keeps the matches, and adds its own token at the first miss. Concurrency then uses that one read to check the drafts of several conversations.
- regime: Ripple 65B, dense, one DGX Spark, decode only. One conversation, then eight.
- number: 296 ms instead of 245 ms, about 8 tokens kept per round instead of 1; 4.1 tok/s to 26.5 tok/s, then 368 ms and about 49 tokens kept, 26.5 tok/s to 133.5 tok/s.
- quote: 296 ms instead of 245 ms, but about 8 tokens kept per round instead of 1.
- relevance: Against the README baseline (TPOT 10.8528 ms, prompt-lookup acceptance ~7.1): the post's ~8 tokens kept per round is the same lever as our acceptance, on a bandwidth-bound dense model. Eight-way concurrency raises tokens per read and does not shrink one request's step the way a megakernel would against ~3008 kernels/round. Not measured on this Ascend path.

## r072. DFlash2 and built-in MTP on one DGX Spark

- url: https://x.com/JRB_AI/status/2122600183116796357
- handle: @JRB_AI
- authors: JRB
- date: 2026-11-17
- mechanism: The same test is run without and with a drafter. DFlash2 is the drafter on three models; two others use built-in MTP. Output is matched token for token.
- regime: One DGX Spark. Qwen3.8-27B, Ripple 65B, Qwen3.6-35B-A3B with DFlash2; Qwen3.8 Flash-Next and Nemotron 3.5 Super with built-in MTP.
- number: Qwen3.8-27B 11.9 to 78.2 tok/s (6.60x); Ripple 65B 4.1 to 25.9 (6.37x); Qwen3.6-35B-A3B 63.4 to 237.3 (3.74x). MTP: 26.3 to 72.5 (2.75x) and 20.8 to 50.2 (2.42x).
- quote: Qwen3.8-27B: 11.9 to 78.2 tok/s (6.60x)
- relevance: Against the README baseline (TPOT 10.8528 ms, acceptance ~7.1): these are end-to-end tok/s ratios on a DGX Spark, not an acceptance length and not an A3 kernel count. A 6.6x tok/s ratio would matter here only if the extra verify did not add to ~3008 kernels/round. Not remeasured.

## r073. DSpine in llama.cpp on an RTX 3090

- url: https://x.com/ai_hakase_/status/2122979004755988729
- handle: @ai_hakase_
- authors: Ai-Hakase
- date: 2026-11-18
- mechanism: DSpine is Tencent's speculative decoder (draft in parallel, condition through depth). The post reports a llama.cpp port (PR 31941) and a community benchmark, and notes that 15 drafts keep a recurrent state that caps context.
- regime: NVIDIA RTX 3090 24GB, Qwen3.8-27B Q4_K_M, code and math, llama.cpp. 15 drafts limit context to 65,536 tokens; 7 drafts extend it to 98,304.
- number: About 125 tok/s on code and math, 4.5x versus off and 2x versus DFlash2; about 90 tok/s on mixed chat.
- quote: コード・数学タスクにおいて約125 tok/s（非適用時の4.5倍、DFlash2の2倍）を記録したそうですね。
- relevance: Against the README baseline (TPOT 10.8528 ms, acceptance ~7.1): this is a llama.cpp measurement on a 3090, separate from the DSpine paper's SGLang table. 4.5x tok/s is not a new acceptance length and was not run on the A3 stack. Context pressure from 15 drafts is the cost the post states.

## r074. MTP acceptance depends on the prompt

- url: https://x.com/MythomancerX/status/2122856489232515254
- handle: @MythomancerX
- authors: Patrick C.
- date: 2026-11-18
- mechanism: MTP speculative decoding guesses tokens ahead. Code and JSON are predictable, so more guesses are accepted; prose rejects more, and rejected guesses are thrown away.
- regime: Qwen3.8-27B Q4_K_M, llama.cpp, one RTX 3090, 128k context, q4_0 KV cache, MTP on, temperature 0.2, two runs.
- number: Code 78.50 and 78.72 tok/s; JSON 79.27 and 78.16; English 61.17 and 64.26; short story 58.09 and 61.54. Acceptance 92 to 94% on code and JSON, 55 to 67% on prose.
- quote: On code and JSON 92 to 94% of the guesses got accepted. On prose it was 55 to 67%.
- relevance: Against the README baseline (prompt-lookup acceptance ~7.1, TPOT 10.8528 ms): the post's lever is acceptance, and it moves tok/s by about 25% between code and prose on the same 3090. Our 7.1 is a mean tokens/round, not a percent, so the two figures are not interchangeable. Not an A3 measurement.

## r075. One-launch MegaQwen decode megakernel

- url: https://x.com/elliotarledge/status/2122374330399211618
- handle: @elliotarledge
- authors: Elliot Arledge
- date: 2026-11-16
- mechanism: One cooperative megakernel runs every decode step of all four layers in a single launch, 188 blocks, five grid barriers per layer.
- regime: KernelBench-CUDA, RTX PRO 6000, Qwen 4 Max session, MegaQwen Decode, about 2k context in the in-session test.
- number: 4.7% of roofline. An earlier version timed 114.5 microseconds per step; the shipped rewrite times 172 microseconds per step.
- quote: what shipped times 172 microseconds per step in the same kind of test.
- relevance: Against the README baseline (~3008 kernels/round, MIX_AIC floor 13.58 us, TPOT 10.8528 ms): one launch for four layers is the same launch-count idea, on a four-layer bench model rather than our 78-layer decode. 172 us/step is not comparable to 10.8528 ms until the model, batch, and kernel count match. Not measured here.

## r076. W4A16 one-launch Kimi-Linear decode megakernel

- url: https://x.com/elliotarledge/status/2120631976181035177
- handle: @elliotarledge
- authors: Elliot Arledge
- date: 2026-11-11
- mechanism: One launch holds 376 persistent blocks in lockstep across three KDA layers. Int4 GEMV turns each weight nibble into bf16 with a bit-OR, and zero-points fold once per 32-element group, keeping activations in fp32 (W4A16). A single-word software barrier flips its top bit when the last block arrives.
- regime: KernelBench-Mega, RTX PRO 6000, Kimi-Linear decode, 2k context.
- number: 23.8x the optimized PyTorch baseline, 0.25 ms per token at 2k context.
- quote: It wrote a one-launch W4A16 Kimi-Linear Decode megakernel for RTX PRO 6000 at 23.8x the optimized PyTorch baseline (0.25 ms per token at 2k context).
- relevance: Against the README baseline (TPOT 10.8528 ms, MIX_AIC 13.58 us, ~3008 kernels/round): 0.25 ms/token is one launch of a small bench model, not our round. The W4A16 nibble expand is a different quant path from the A3 W4A8 MSD kernels already profiled. The 23.8x is against that session's PyTorch baseline, so it does not move the 10.8528 ms TPOT. Acceptance ~7.1 and the 5.35 ms TP reduce are not in this post.

## r077. Native sparse attention kernel, megakernel skipped

- url: https://x.com/elliotarledge/status/2122854017097511334
- handle: @elliotarledge
- authors: Elliot Arledge
- date: 2026-11-18
- mechanism: Hand-written CUDA for KernelBench. On MegaQwen Decode the session skipped the megakernel. Native sparse attention is reported as its own kernel time.
- regime: KernelBench-CUDA, RTX PRO 6000, GPT-6.2 Astra session.
- number: 0.123 ms on Native Sparse Attention; 5.8% of roofline on MegaQwen Decode.
- quote: 0.123 ms on Native Sparse Attention
- relevance: Against the README baseline (SparseFlashAttention 6.973 ms, TPOT 10.8528 ms): 0.123 ms is one sparse-attention kernel on a bench shape, not a full decode round and not our A3 SparseFlashAttention. Skipping the megakernel is the post's own limit on that item. Not remeasured.

## r078. MXFP8 linears, less CUDA-graph padding, fused MoE all-reduce

- url: https://x.com/itschef_ai/status/2122335157508288584
- handle: @itschef_ai
- authors: Chef_AI
- date: 2026-11-16
- mechanism: Skip CUDA-graph padding in MoE routing, move BF16 linears to MXFP8, retune graphs and the pipeline split. Sequence-parallel MoE then runs across pipeline stages.
- regime: Kimi K3.1 Pro kernels. Device is not named in the post.
- number: Decode 3,240 to 9,405 tok/s. Sequence-parallel MoE all-reduce 865 to 19 ms. Prefill +29%.
- quote: We skipped CUDA-graph padding in MoE routing, moved BF16 linears to MXFP8, retuned graphs and PP split.
- relevance: Against the README baseline (TPOT 10.8528 ms, TP reduce 5.35 ms): the decode tok/s move is the claim that touches our number, and the 865 to 19 ms all-reduce is a collective on their MoE path, not our 5.35 ms TP reduce. Graph padding is a launch-shape issue next to ~3008 kernels/round only if their graph is the same kind of launch. Not measured on A3.

## r079. Rubin decode path with fused all-reduces

- url: https://x.com/vllm_project/status/2121008103735087502
- handle: @vllm_project
- authors: vLLM
- date: 2026-11-12
- mechanism: Bandwidth-bound data stays in each SM's memory domain, and SM107's wider MMA does more work per issue. The decode path fuses all-reduces into the kernels behind them, puts the shared expert inside the routed GEMMs, and starts loads before the wait.
- regime: vLLM on Rubin, SM107. DEP decode steps. Batch and model are not named in this post.
- number: DEP2 prefill steps about 28% faster, and DEP decode steps about 25% faster.
- quote: DEP decode steps about 25% faster.
- relevance: Against the README baseline (TP reduce 5.35 ms, MIX_AIC floor 13.58 us, TPOT 10.8528 ms): fusing the all-reduce into the producer is the same idea as cutting a serial 5.35 ms reduce, but 25% is on Rubin DEP steps, not on A3. A fused collective still has to be cheaper than 13.58 us of MIX launch plus the reduce it replaces. The 25% is not on this stack, so it does not move the 10.8528 ms TPOT or the ~3008 kernel count. Acceptance ~7.1 is not stated.

## r080. NVLink-multicast fusion and a long-context decode kernel

- url: https://x.com/vllm_project/status/2120992730134520226
- handle: @vllm_project
- authors: vLLM
- date: 2026-11-12
- mechanism: All-reduce, conv, residual, attention-residual mix, and RMSNorm are fused into one NVLink-multicast kernel, plus programmatic dependent launch. A separate Gluon kernel covers relative attention at long context.
- regime: vLLM, Inkling-2-Small recipe thread. The decode kernel is reported at 100K context. GPU count is not in this post.
- number: Gluon relative-attention decode kernel: 15.1 to 4.8 ms per decode step at 100K context. Sequence-parallel prefill is 22–26% on 8K, which is prefill.
- quote: a Gluon relative-attention decode kernel: 15.1 → 4.8 ms per decode step at 100K context
- relevance: Against the README baseline (TPOT 10.8528 ms, SparseFlashAttention 6.973 ms): 4.8 ms is one attention kernel at 100K context, already in the same order as our whole-round 10.8528 ms, so it does not transfer onto a short-context A3 round. The fused multicast kernel is a collective-plus-epilogue claim next to our 5.35 ms TP reduce, on NVLink, not HCCL.

## r081. Fixed work per decode step on an M5 Max

- url: https://x.com/TeamPrecisit/status/2120966183256031499
- handle: @TeamPrecisit
- authors: Precisit
- date: 2026-11-12
- mechanism: They split a decode step into fixed work and model work, and ask how much a megakernel could save before writing one.
- regime: M5 Max. A 0.6B model and an 8B model.
- number: About 1.1 ms of fixed work per decode step: 68% of a 0.6B step, 12% of an 8B step.
- quote: We measured about 1.1 ms of fixed work per decode step on an M5 Max.
- relevance: Against the README baseline (~3008 kernels/round, MIX floors 0.70–13.58 us, TPOT 10.8528 ms): 1.1 ms of fixed work is the launch-overhead question on an M5 Max. Our launch floor is already itemized near 11.5 ms because the kernel count is ~3008, so the fraction is larger than their 12% of an 8B step. Their 1.1 ms does not replace that accounting.

## r082. MTP depth 5 and 8-bit KV on an M3 Ultra

- url: https://x.com/GrimGreysson/status/2122863259699347493
- handle: @GrimGreysson
- authors: Grimm
- date: 2026-11-18
- mechanism: MTP at depth 5, with 8-bit KV and a BF16 n-gram table streamed off SSD, on a requantized 4-bit checkpoint.
- regime: Qwen3.8-Flash-Next, 1M context, mlx-serve 26.10.2, M3 Ultra 256GB, YaRN 4x. Cold 1M prompt.
- number: Cold 1M prompt: ~719 tok/s prefill, ~85 tok/s decode, ~116 GiB peak.
- quote: Cold 1M prompt: ~719 tok/s prefill, ~85 tok/s decode, ~116 GiB peak
- relevance: Against the README baseline (TPOT 10.8528 ms, acceptance ~7.1): ~85 tok/s decode is about 11.8 ms/token on an M3 Ultra at 1M context with MTP depth 5, close to our 10.8528 ms only as a unit conversion, not as the same model or kernel path. Acceptance is not stated, so it does not say whether depth 5 beats ~7.1. The unit match does not move the 10.8528 ms TPOT, and the post says nothing about ~3008 kernels, the 13.58 us MIX floor, or the 5.35 ms TP reduce.

## r083. SGLang megakernel for Muse Spark Open

- url: https://x.com/lmsysorg/status/2120566740560261237
- handle: @lmsysorg
- authors: LMSYS Org
- date: 2026-11-11
- mechanism: One kernel launch per layer overlaps compute, memory loads, and communication. Weights are prefetched into shared memory and L2, the instruction cache is warmed, and data itself is the readiness signal for cross-GPU communication. The draft model is fused.
- regime: Muse Spark Open, 8x GB300, SGLang. The post states the batch-1 tradeoff in the linked writeup's framing; this status post gives the launch counts and tok/s.
- number: 517 to 1,109 tokens/s. Launches per decode step drop from 1,450 to 97. Drafting is 2.6x faster.
- quote: Total launches per decode step drop from 1,450 to 97.
- relevance: Against the README baseline (~3008 kernels/round, MIX_AIC 13.58 us, TPOT 10.8528 ms): 1,450 to 97 launches is the same lever as our kernel count, on 8x GB300 rather than A3. Their residual 97 launches would still be expensive at our MIX floor if each stayed a MIX launch. 1,109 tok/s is their batch regime. It does not move the 10.8528 ms TPOT. Acceptance ~7.1 and the 5.35 ms TP reduce are not in this post.

## r084. EXL3 plus MTP, and an engine swap on the same card

- url: https://x.com/YH31377807/status/2122996623345389919
- handle: @YH31377807
- authors: YH
- date: 2026-11-18
- mechanism: EXL3 3 bpw weights plus MTP on a 3090. On an Arc Pro B70 the same model moves from llama.cpp to vLLM plus custom XPU kernels.
- regime: Qwen3.8-27B. RTX 3090 with EXL3 3 bpw + MTP. Intel Arc Pro B70, vLLM plus custom kernels versus llama.cpp Q4_K_M. Mac Studio M5 Max 40-core, llama.cpp UD-Q5_K_XL, no drafter.
- number: 3090: 96.2 tok/s prose, 141.1 code. B70 vLLM 91.2 tok/s versus llama.cpp 25.0 tok/s. M5 Max 25 tok/s with no drafter.
- quote: RTX 3090, EXL3 3 bpw + MTP: 96.2 tok/s prose, 141.1 code
- relevance: Against the README baseline (TPOT 10.8528 ms, acceptance ~7.1): 96.2 tok/s is about 10.4 ms/token on a 3090 with MTP, a unit match to 10.8528 ms and not a transfer. The 25 to 91 tok/s gap on one B70 is an engine and kernel change, not evidence about our ~3008 A3 launches. Acceptance is not given, and the unit match does not move the 10.8528 ms TPOT. The 13.58 us MIX floor and the 5.35 ms TP reduce are not in this post.

## r085. MTP with almost no accepts slows the step

- url: https://x.com/tolewis/status/2122879970548625474
- handle: @tolewis
- authors: Timothy Lewis
- date: 2026-11-18
- mechanism: The drafter guesses ahead and the model checks. When almost none of the guesses are accepted, every step is spent verifying drafts that are thrown out.
- regime: Qwen 3.8 27B on tensorfold, a B70 pair. Four checks: 1 of 122, 2 of 288, 1 of 159, 0 of 107.
- number: 119 tok/s down to 17.6 tok/s the first time MTP was turned on. Under 1% of guesses accepted. 6.7x slower than leaving MTP off.
- quote: 119 tok/s down to 17.6 the first time i turned mtp on
- relevance: Against the README baseline (acceptance ~7.1, TPOT 10.8528 ms): this is the failure mode of speculation. Our ~7.1 tokens/round is why the draft pays; their under-1% accept rate made the verify pure overhead and slowed tok/s 6.7x. It does not move our number, and it says a drafter must not be added when acceptance is near zero.

## r086. EXL3 3bpw plus MTP on a 35B MoE

- url: https://x.com/fngly343149/status/2122878740078428479
- handle: @fngly343149
- authors: 福叔
- date: 2026-11-18
- mechanism: Local-inference tuning: EXL3 3 bpw weights plus MTP on a 3090, and vLLM speculative decoding for a DeepSeek flash model.
- regime: Qwen3.6-35B-A3B, EXL3 3 bpw + MTP, one RTX 3090. Separately, DeepSeek-V4.1-Flash, vLLM speculative decoding, 4x H200.
- number: 3090: 251.7 tok/s, peak 357.9. 4x H200: prefill 17546 tok/s, decode 229.1 tok/s.
- quote: 3090 上 Qwen3.6-35B-A3B（EXL3 3bpw 加 MTP）跑到 251.7 tok/s，峰值 357.9
- relevance: Against the README baseline (TPOT 10.8528 ms, acceptance ~7.1): 251.7 tok/s is about 4.0 ms/token on a 3090 MoE with MTP, which would beat 10.8528 ms only if the model, batch, and acceptance matched. They are not stated as our GLM-5.2 A3 round. The H200 decode figure is a different stack. Not remeasured.

## r087. Speculative decode on two SRAM chips

- url: https://x.com/deedydas/status/2040083405841568115
- handle: @deedydas
- authors: Deedy
- date: 2026-04-03
- mechanism: The post reports a system that moves speculative decode onto two 2GB SRAM chips (Corsairs) added beside a standard GPU setup.
- regime: gpt-oss-120b, two 2GB SRAM/chip Corsairs plus a standard GPU setup. The post does not name the GPU count.
- number: 10x latency and more than 1400 tokens/sec.
- quote: They achieved 10x latency and >1400 tokens/sec by moving speculative decode onto two 2GB SRAM/chip Corsairs, a small cost on top of a standard GPU setup on gpt-oss-120b.
- relevance: Against the README baseline (TPOT 10.8528 ms, acceptance ~7.1): a 10x latency cut from putting the draft on SRAM is a memory-placement claim, not a result on A3 HBM. 1400 tok/s is their stack. It does not move the 10.8528 ms TPOT until the draft's bytes move on this device. The post does not speak to ~3008 kernels, the 13.58 us MIX floor, or the 5.35 ms TP reduce.

## r088. A band of specialist drafters

- url: https://x.com/yuxiangw_cs/status/1982013575934964075
- handle: @yuxiangw_cs
- authors: Yu-Xiang Wang
- date: 2025-10-25
- mechanism: Speculative decoding with a band of specialist drafters instead of one draft model. The post calls the gain careful math plus engineering, and points at a NeurIPS line of work in the thread's framing of token-level speculation.
- regime: Llama-3.1-8B-Instruct and Qwen-32B. The post does not name the GPU or the batch.
- number: 4.99x on Llama-3.1-8B-Instruct, 4.93x on Qwen-32B, beating EAGLE3 by nearly 2x.
- quote: we got 4.99× on Llama-3.1-8B-Instruct, 4.93× on Qwen-32B — beating EAGLE3 by nearly 2x.
- relevance: Against the README baseline (acceptance ~7.1, TPOT 10.8528 ms): 4.99x is a wall-clock ratio on those two models, not an acceptance length and not our kernel path. Beating EAGLE-3 does not say the accept run is longer than ~7.1. Not measured on A3.

## r089. KV cache sized after CUDA graph capture

- url: https://x.com/AIQuanting/status/2122882231387685201
- handle: @AIQuanting
- authors: AI Quanting
- date: 2026-11-18
- mechanism: vLLM 0.34 sizes the KV cache from memory that is free after CUDA graph capture and warmup, instead of a fixed reserve. The post also notes a shared RoPE cache and an expert weight filter.
- regime: vLLM 0.34. MiniMax-M3 on H200 for the cache-size example. Decode-latency cost stated for some H200 models at concurrency 128.
- number: KV cache for MiniMax-M3 on H200 grows from 433k to 672k tokens. Per-token decode latency can rise 6 to 18 percent at concurrency 128.
- quote: Per-token decode latency can rise 6 to 18 percent on some H200 models at concurrency 128.
- relevance: Against the README baseline (TPOT 10.8528 ms): this technique buys KV capacity and the post says it can make per-token decode slower, so it does not cut 10.8528 ms. A 6 to 18 percent rise would move our TPOT the wrong way if the same sizing were applied. Not measured on A3.

## r090. DeepSeek-V4.1 PD disaggregation on H200

- url: https://x.com/lmsysorg/status/2122900113832239405
- handle: @lmsysorg
- authors: LMSYS Org
- date: 2026-11-18
- mechanism: SGLang prefill-decode disaggregation for DeepSeek-V4.1-Flash: a cheaper sparse-attention indexer, bigger prefill chunks, leaner MoE communication, and late layers that compute only the last 128 tokens.
- regime: 8x H200. Eight cases: 64K in / 512 out, 50/70/90% prefix hit, concurrency 4–32.
- number: E2E 15%–64% lower and TTFT 41%–79% lower versus the original baseline. The 128-token mode cuts TTFT by up to a further 46.7%. TPOT is 3%–52% higher. GSM8K 96.74% to 95.38%.
- quote: The cost: TPOT 3%–52% higher, GSM8K 96.74% → 95.38%
- relevance: Against the README baseline (TPOT 10.8528 ms): the post's own cost is a higher TPOT, bought with lower TTFT. That does not cut our decode round. The late-128-token mode is a prefill/attention shortcut, and the quality dip is the trade the post states. Not an A3 measurement.

## r091. Decode step time left in inter-stage waits

- url: https://x.com/benblainex/status/2121720455904088398
- handle: @benblainex
- authors: Ben Blaine
- date: 2026-11-14
- mechanism: With CUDA graphs, the remaining decode time is on the GPU: waits between dependent stages (completion, ramp, all-reduce latency) rather than streaming weights. Fewer dependent stages, from fusions such as an 80-to-1 mHC kernel and fused routing, are what pay.
- regime: A model of Baseten's DeepSeek-V4.1 engine. The sibling post in the thread states 1,740 tok/s on 4x GB300. This post does not restate the GPU count.
- number: About half of each decode step is waiting between dependent stages. The fusion cited is 80 to 1 for the mHC kernel.
- quote: ~half of each decode step is waiting between dependent stages (completion, ramp, all-reduce latency), not streaming weights.
- relevance: Against the README baseline (TP reduce 5.35 ms serial, ~3008 kernels/round, TPOT 10.8528 ms): half the step in inter-stage and all-reduce waits is the same shape as a serial 5.35 ms reduce plus launch gaps. Their fusion cuts stage count on a GB300 model of V4.1, not on this A3 profile. Not remeasured.

## r092. Modeled DeepSeek-V4.1 decode at 1,740 tok/s

- url: https://x.com/aercrist/status/2121715821281984741
- handle: @aercrist
- authors: Aercrist
- date: 2026-11-14
- mechanism: They model an existing ultrafast DeepSeek-V4.1 engine to see where decode time goes, and argue purpose-built silicon could take the remaining headroom.
- regime: Baseten deepseek-ultrafast, 4x GB300, DeepSeek-V4.1-Flash decode.
- number: 1,740 tok/s on 4x GB300, and about 2x headroom left in the model.
- quote: We modeled @baseten's deepseek-ultrafast (1,740 tok/s on 4x GB300) to find where the time goes.
- relevance: Against the README baseline (TPOT 10.8528 ms): 1,740 tok/s is about 0.57 ms/token on 4x GB300, a different model and device from our 10.8528 ms A3 round. The 2x headroom is their model of that engine, not a kernel we can drop onto ~3008 A3 launches. Not remeasured.

## r093. Chipset ACS off speeds decode-sized all-reduces

- url: https://x.com/taf2/status/2121790872446521712
- handle: @taf2
- authors: taf2
- date: 2026-11-15
- mechanism: Turning ACS off on a Broadcom Gen5 chipset shortens all-reduces. The post applies that to a DeepSeek run.
- regime: Broadcom Gen5 chipset. An 80 MiB prefill all-reduce is quoted separately; decode-sized all-reduces are the decode claim. GPU is not named.
- number: 80 MiB prefill all-reduce 8.24 to 3.66 ms (56%). Decode-sized all-reduces 3–15% faster.
- quote: decode-sized all-reduces got 3–15% faster.
- relevance: Against the README baseline (TPOT 10.8528 ms, ~3008 kernels/round, MIX_AIC launch floor 13.58 us, prompt-lookup acceptance ~7.1, TP reduce 5.35 ms): a 3–15% cut on decode-sized all-reduces is only a fraction of the 5.35 ms reduce, so it does not move the 10.8528 ms TPOT. The 8.24 to 3.66 ms figure is prefill-sized (80 MiB), not our per-layer decode payload, and it does not change the ~3008 kernel count. Not measured on A3.

## r094. Prefill-decode disaggregation, request path

- url: https://x.com/RaghavMehtaM1/status/2122347402342395924
- handle: @RaghavMehtaM1
- authors: Raghav Mehta
- date: 2026-11-16
- mechanism: Split prefill and decode onto separate GPUs. A router picks one prefill worker and one decode worker, the decode side reserves KV blocks, prefill runs, KV moves over RDMA, and decode joins the running batch. SGLang does the transfer from a background thread so the scheduler does not block.
- regime: A 13B on one A100 for the before/after rate. KV size example is one 512-token request on OPT-66B.
- number: About 1.6 req/s on one A100 before latency targets were missed, then 3.3 per GPU after the split. One 512-token OPT-66B request carries about 1.13 GB of KV.
- quote: a 13b model on one a100 served about 1.6 req/s before users started missing latency targets. move the prompt work and the generation onto separate gpus and the same hardware does 3.3 per gpu
- relevance: Against the README baseline (TPOT 10.8528 ms): disaggregation stops prefill from stalling decode, which is a concurrency latency, not a shorter single-token kernel. 1.13 GB of KV per 512 tokens is transfer traffic, off our steady 10.8528 ms loop unless that copy sits on the critical path. Not this A3 stack.

## r095. MTP patch on a Spark GGUF build

- url: https://x.com/termuxSolutions/status/2122971043010683216
- handle: @termuxSolutions
- authors: Alen Požun
- date: 2026-11-18
- mechanism: The post points at a llama.cpp patch (hy4-flash-spark.patch) that adds architecture support, CUDA options, and MTP, built for CUDA arch 121a.
- regime: AngelSlim GGUF card, SPARK-4Bit-Fast, one 128GB Spark. llama.cpp support still an open PR. Prompt figure is at 4K.
- number: About 32–33 tok/s decode and about 800 tok/s prompt at 4K.
- quote: SPARK-4Bit-Fast at ~32-33 tok/s decode and ~800 tok/s prompt (4K) on a single 128GB Spark.
- relevance: Against the README baseline (TPOT 10.8528 ms, acceptance ~7.1): 32 tok/s is about 31 ms/token, slower than 10.8528 ms, on a Spark with MTP in the patch. The post does not give acceptance, so it does not say the MTP depth beats ~7.1. Not an A3 result.

## r096. 4-bit Qwen with speculative decoding on a 5090

- url: https://x.com/eres2k/status/2123003203797000438
- handle: @eres2k
- authors: Erwin
- date: 2026-11-18
- mechanism: vLLM serves a 4-bit Qwen fine-tune with speculative decoding.
- regime: One RTX 5090, Linux, vLLM, 4-bit Qwen fine-tune.
- number: About 270 tok/s.
- quote: One RTX 5090 on Linux. vLLM runs a 4-bit Qwen fine-tune with speculative decoding: ~270 tok/s.
- relevance: Against the README baseline (TPOT 10.8528 ms, acceptance ~7.1): ~270 tok/s is about 3.7 ms/token on a 5090. The post does not give acceptance or kernel count, so it cannot be lined up with ~7.1 or ~3008 launches. It is not an A3 measurement and does not move 10.8528 ms.

## r097. Lookahead reasoning on top of token speculation

- url: https://x.com/haozhangml/status/1970607910846898488
- handle: @haozhangml
- authors: Hao Zhang
- date: 2025-09-23
- mechanism: Token-level speculative decoding's accept probability falls as the draft block grows. Lookahead reasoning adds a second layer: draft future reasoning steps, verify them in one batched pass, keep the semantically correct steps, and still run token-level speculation inside each kept step.
- regime: The post names the GPU trend H200 to B200 to Rubin CPX as the reason to draft longer blocks. It does not give a model, batch, or a measured speedup in this status text.
- number: No numeric speedup in this status text. The load-bearing claim is that a whole k-token draft's full accept probability drops exponentially in k, so token-only speedups plateau.
- quote: the probability that a whole k-token draft is fully accepted drops exponentially as k grows—so speedups plateau fast.
- relevance: Against the README baseline (acceptance ~7.1, TPOT 10.8528 ms): our ~7.1 is already a long accept run, which is the regime where the post says token-only drafts stop paying. Step-level verify would matter here only if it raised tokens per target forward without adding to ~3008 kernels/round. This post states no measured TPOT, so it does not move 10.8528 ms.

## r098. Flash-decoding written from scratch on a 3060

- url: https://x.com/Abhishekcur/status/2076346185561055361
- handle: @Abhishekcur
- authors: Abhishek
- date: 2026-07-12
- mechanism: Decode kernels written from scratch: GEMV, RMSNorm, SiLU, softmax, attention. Flash-decoding is online softmax plus split-KV. Naive attention cannot launch at long context because of the shared-memory cap.
- regime: RTX 3060. GEMV bandwidth versus compute. Flash-decoding at 16K context. 161 checks against an fp64 CPU reference.
- number: GEMV about 92% of DRAM peak and about 1% of compute. Split-KV flash-decoding about 80% of peak at 16K context. Naive-to-optimized softmax about 210x on a logits-sized matrix.
- quote: Online softmax plus split-KV reaches ~80% of peak at 16K context, a length where my naive kernel cannot even launch (shared memory cap).
- relevance: Against the README baseline (SparseFlashAttention 6.973 ms of which 1.382 ms is bytes, TPOT 10.8528 ms): 80% of a 3060's peak at 16K says decode attention can be bandwidth-bound once the launch fits. Our SparseFlashAttention is already far above its byte floor, so a split-KV kernel is in scope only as a candidate, not as a transferred 80%. Not measured on A3.

## r099. Adaptive DFlash speculation on Strix Halo

- url: https://x.com/TeksEdge/status/2092673436904489362
- handle: @TeksEdge
- authors: David Hendrickson
- date: 2026-08-26
- mechanism: A llama.cpp fork replaces a fixed DFlash or MTP draft length with adaptive speculation: it watches acceptance and decides how far ahead to speculate.
- regime: Qwen3.8-27B FP4 on Strix Halo, ROCmFP4 + DFlash2, highly structured output. The post says the fork is not merged, and that independent tests are often 30–45 tok/s.
- number: Bare decode 14.0 tok/s, fixed DFlash n=3 41.6 tok/s, adaptive DFlash 65.6 tok/s, acceptance around 96%, about 4.7x bare decode.
- quote: Instead of hard-coding DFlash/MTP to predict a fixed number of tokens, it watches acceptance in real time and dynamically decides how far ahead to speculate.
- relevance: Against the README baseline (acceptance ~7.1, TPOT 10.8528 ms): adaptive draft length is a control on the same accept lever as ~7.1. 65.6 tok/s is about 15 ms/token on Strix Halo for structured output, not our A3 round, and the post's own caveat is that other workloads land at 30–45 tok/s. Not remeasured.

## r100. MTP, DFlash2, and context lookup on one 3090

- url: https://x.com/Oluwaphilemon1/status/2114179323326255353
- handle: @Oluwaphilemon1
- authors: FHILY
- date: 2026-10-25
- mechanism: The post reports TeksEdge's stack: speculative decoding, DFlash2, context lookup that proposes sequences already in the prompt, and a faster verify of those proposals. Lookup uses blocks of up to 16 tokens.
- regime: Qwen3.8-27B, one RTX 3090 24GB. The 381 tok/s figure is document reproduction; ordinary chat is stated separately. A 25,000-token reproduction is the lookup case.
- number: 82 tok/s baseline, optimized MTP 114, DFlash2 plus context lookup 138, optimized lookup verification 381. About 15 of 16 lookup tokens accepted. Ordinary chat about 133 tok/s. On a 25,000-token reproduction, about 260 to 382 tok/s.
- quote: TeksEdge reports processing blocks of up to 16 tokens and accepting roughly 15 tokens per block in its setup.
- relevance: Against the README baseline (prompt-lookup acceptance ~7.1, TPOT 10.8528 ms): 15 of 16 on document reproduction is a higher accept run than ~7.1, and only on text that is already in the prompt. The post says ordinary chat is about 133 tok/s, so the 381 figure does not replace 10.8528 ms. Not an A3 measurement. This is a report of TeksEdge's numbers, not a second copy of an arXiv item in this file.

## Gaps

Search anchors. arXiv API `submittedDate:[202501010000 TO 202611182359]`, forty hits per query, newest first: `speculative decoding`; `multi-token prediction` with inference or decoding or serving; `TPOT` or `time per output token` or `inter-token latency` or `decode latency`; `megakernel` or `persistent kernel` or `cuda graph`; `flash decoding` or `FlashMLA` or `decode attention` or `paged attention`; `W4A8` or weight-only or activation quantization; `allreduce` or disaggregation with latency; `EAGLE` or `Medusa` or `speculative sampling`; titles `low-latency` plus LLM; `KV cache` with decode and latency or quantization. A second `id_list` fetch picked up DFlash (2602.06036), EAGLE-3 (2503.01840), DSpine (2609.36173), speculative pipeline decoding (2605.30852), Ada-MK (2605.11581), AutoMegaKernel (2606.09682), NVRAR (2511.09557), LazyWeights (2611.17108), and Chimera (2611.21677), which the forty-hit cap had dropped. X keyword and semantic search, 2025-01-01 through 2026-11-18: speculative decoding, MTP, megakernel, kernel launch, TPOT, all-reduce, flash-decoding, and measured tok/s with a named model.

Excluded. arXiv ids already cited in the other files in this directory, so they are not counted again: 2311.08252, 2312.12728, 2405.04434, 2406.06858, 2411.04975, 2411.10666, 2505.14022, 2505.21411, 2506.04667, 2509.23565, 2512.22219, 2602.15763, 2604.13327, 2604.26469, 2605.07881, 2605.13915, 2606.05951, 2607.08973, 2607.16100, 2608.10103, 2609.04244, 2609.12379, 2609.27453. Pre-2025 work, including SuffixDecoding 2411.04975, FLUX 2406.06858, and FlashAttention-2 2405.04434. Training recipes whose contribution is a loss rather than an inference mechanism: REAL (2611.14399), D-REAL (2611.10432), TVP (2611.00588), verification-aware training (2608.30135), AdaMTP (2608.00434). Off-topic hits from the same queries: video generation, VLA and robot policies, ASR and TTS, recommendation, algal-bloom segmentation, orbital-link feasibility with no decode kernel, and agent-tool speculation that does not change the token loop. X posts that only restate an item already filed: the Muse Spark echoes of the LMSYS status, MMFlash announcements of 2611.18482, and repeated retellings of the same 3090 lookup result beyond the one post kept. Roadmap posts with no measurement. Grok confirmations. HyFlash (2611.07263) and the training-free relaxed speculative-decoding note (2607.08690) had no latency, speedup, or acceptance figure in the abstract, so they were dropped.

Subtopics that returned nothing usable. A 2025-or-later prompt-lookup or suffix-decoding paper with a controlled mean acceptance above the ~7.1 already filed from pre-2025 SuffixDecoding. Context-lookup posts in this set report about 15 of 16 accepts on document reproduction, which is a different drafter and not that paper. A public Ascend A3 batch-1 megakernel paper whose TPOT sits next to 10.8528 ms. FlashOffloading (2611.07542) is Ascend910C under a 30 ms TPOT SLO, not this kernel stack. An in-kernel Ascend all-reduce newer than the SiFAR and near-speed-of-light collective papers already quoted in `low-latency-tp-reduce.md`. New collectives in this file are NVIDIA, TPU, or chipset ACS. A W4A8 multi-scale-dequant paper newer than 2605.13915, which is already the subject of `w4a8-msd-int4-mmad.md`. The quant items here are FP4, FP8, KV-cache bits, or weight-only DRAM decoding.
