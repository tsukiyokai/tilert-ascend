# TileLang on Ascend — source material

> STATUS: deprioritized. The main agent has decided on hand-written AscendC, not TileLang.
> This file is kept because it is the best public documentation of *which AscendC primitives
> matter and how they compose* — TileLang-Ascend is a thin wrapper, and its README exposes the
> flag/pipe/scope/buffer vocabulary verbatim, plus real on-chip address maps from a working
> sparse-attention kernel. Read it as AscendC reference, not as a toolchain proposal.

---

## 1. tile-ai org inventory (as of 2026-10-20)

Fetched via authenticated GitHub API, `gh api orgs/tile-ai/repos`.

```
tile-ai/tilelang              8658 stars  pushed 2026-10-20  Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels
tile-ai/TileRT                1869 stars  pushed 2026-09-24  Tile-Based Runtime for Ultra-Low-Latency LLM Inference
tile-ai/tilelang-ascend        417 stars  pushed 2026-10-19  Ascend TileLang adapter
tile-ai/tilelang-puzzles       388 stars  pushed 2026-08-11  Learning TileLang with 10 puzzles!
tile-ai/tilescale              196 stars  pushed 2026-08-25  Tile-based language built for AI computation across all scales
tile-ai/TileOPs                195 stars  pushed 2026-10-19  High-performance LLM operator library built on TileLang.
tile-ai/TileFoundry            129 stars  pushed 2026-10-19
tile-ai/DeepStack               65 stars  pushed 2026-10-05  DeepStack: ... MICRO 2026 AE artifact.
tile-ai/AttentionEngine         54 stars  pushed 2025-05-19
tile-ai/tilelang-metax          43 stars  pushed 2026-10-16
tile-ai/tilelang-lsp            39 stars  pushed 2026-08-04
tile-ai/tilelang-mlir-ascend    30 stars  pushed 2026-10-19  MLIR-based TileLang Ascend Adapter
tile-ai/tilelang-benchmark      26 stars  pushed 2026-06-10
tile-ai/tvm                     21 stars  pushed 2026-10-07
tile-ai/TileSight               18 stars  pushed 2026-10-13
tile-ai/tilelang-musa           14 stars  pushed 2026-09-30
tile-ai/tilelang-sunrise        14 stars  pushed 2026-10-12
tile-ai/BitBLAS                 10 stars  pushed 2026-01-28
tile-ai/Ladder                   9 stars  pushed 2025-02-08
tile-ai/tilelang-hygon           8 stars  pushed 2026-10-20
tile-ai/Liger-tilelang           8 stars  pushed 2025-03-07
```

Two Ascend adapters exist, with two distinct backend routes. From the tilelang-ascend README:

> The compiler backend supports two technical routes: [Ascend C & PTO](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto) and [AscendNPU IR](https://github.com/tile-ai/tilelang-ascend/tree/npuir).

Live branches (`gh api repos/tile-ai/tilelang-ascend/branches`): `ascendc_pto`, `npuir`,
`npuir-dev`, `ascendc_pto_citest_0421`, `example_test`, plus revert/patch branches.

### Relevance

`tile-ai/TileRT` is the parent project we are an Ascend port of; its own description is
"Tile-Based Runtime for Ultra-Low-Latency LLM Inference", and its README states the
architecture we are reproducing.

---

## 2. TileRT's own statement of its architecture

URL: https://github.com/tile-ai/TileRT (README)
Author: tile-ai
Date: README as of 2026-09-24 push; news entries dated through 2026-09-24

> **TileRT** is a project designed to serve large language models (LLMs) in ultra-low-latency scenarios. Its goal is to push the latency limits of LLMs without compromising model size or quality—enabling models with hundreds of billions of parameters to achieve millisecond-level time per output token (TPOT).
>
> Unlike traditional inference systems optimized for high-throughput batch processing, TileRT prioritizes **responsiveness**, which is critical for applications such as high-frequency trading, interactive AI, real-time decision-making, long-running agents, and AI-assisted coding, where the latency of individual requests matters most.
>
> To achieve this, TileRT introduces a **tile-level runtime engine**. Leveraging a compiler-driven approach, LLM operators are decomposed into fine-grained tile-level tasks, while the runtime dynamically reschedules computation, I/O, and communication across multiple devices in a highly overlapped manner. This design minimizes idle time and improves hardware utilization.
>
> The project is actively evolving, and the underlying compiler techniques will be gradually shared with the community as they are integrated into **TileLang** and **TileScale**.

Published GLM-5.2/5.3 result, verbatim:

> - 🏆 **2026-09-24 · [TileRT Takes the Top Spot on AgentX](https://www.tilert.ai/blog/tilert-amd-agentx.html)**. On SemiAnalysis's InferenceX **AgentX** leaderboard, TileRT powers **GLM-5.3** to **469 tok/s** of single-user generation on **million-token agent sessions** with **8× AMD Instinct MI355X** — **#1 in single-user performance**. Running the original **FP8** weights, it leads the second-place NVIDIA GB300 NVL72 FP4 submission by more than 100 tok/s.

> <sub><em>GLM-5.2/5.3-FP8 token generation speed on 8× AMD Instinct MI350X with TileRT v0.1.6 (greedy decoding, bf16 KV and indexer caches). Output length 1K, input length 1K–1M. Bars compare TileRT without MTP, with MTP at depth 3 and average acceptance length 3.2, and the peak under best-case MTP acceptance (4.0).</em></sub>

> - 🔀 **2026-07-14 · [v0.1.5](https://github.com/tile-ai/TileRT/releases/tag/v0.1.5.post2) Released**. Introduce [**PD (prefill–decode) disaggregation**](https://www.tilert.ai/blog/tilert-vllm-disaggregation.html) — vLLM prefill + TileRT decode, behind an OpenAI-compatible endpoint. Supported on GLM-5/5.1 and DeepSeek-V3.2.

> - 🚀 **2026-01-26 · [v0.1.2-alpha.1](https://github.com/tile-ai/TileRT/releases/tag/v0.1.2-alpha.1)**. **Multi-Token Prediction (MTP)** is now available in TileRT! With mtp=3, we achieve decoding rates of up to **590 tokens/s** under synthetic workloads.

### Relevance

Two numbers bear on our speculative-decoding plan directly: GLM-5.3 on 8x MI355X reaches
469 tok/s single-user with "MTP at depth 3 and average acceptance length 3.2", and the
published best case is acceptance 4.0 — against our measured prompt-lookup acceptance of 7.1
tokens/round. Our drafting is already better than TileRT's own MTP on the same model family;
the gap to 2.x ms TPOT is therefore in the round time, not in acceptance.

---

## 3. tilelang-ascend — the AscendC vocabulary, exposed

URL: https://github.com/tile-ai/tilelang-ascend (README, default branch)
Mirror of the same content: https://ascendnpu-ir.gitcode.com/en/developer_guide/conversion/tile_lang_interface.html
Author: tile-ai, with acknowledgement to "Huawei's HiSilicon, ICT, Compiler and Programming Language Lab and the Peking University Kunpeng & Ascend Center"
Date: README news entries run 2025-09-29 through 2026-04-24

### 3.1 Memory-hierarchy mapping and the two vector-expression styles

> ## Comparison with NVIDIA Backend Implementation
>
> GPUs primarily feature a three-level memory hierarchy that can be analogously mapped to NPU hardware architecture as follows:
>
> **Memory Hierarchy Mapping:**
> - `global memory` ↔ `global memory`
> - `shared memory` ↔ `L1 buffer on cube core and unified buffer on vector core`
> - `register memory` ↔ `L0A/B/C buffer`
>
> **Memory Management:**
> TileLang-Ascend provides memory allocation primitives similar to the GPU version. For example, `alloc_{L1/ub/...}` functions allow on-chip memory allocation in a manner comparable to GPU programming.
>
> **Multiple Styles for Vector Operations:**
> TileLang-Ascend offers multiple ways to express vector computations on the NPU, from high-level automatic vectorization to fine-grained tile primitives:
>
> ```python
> # Style 1: T.Parallel — automatic vectorization
> for (i, j) in T.Parallel(M, N):
>     C_ub[i, j] = A_ub[i, j] + B_ub[i, j]
>
> # Style 2: Tile primitives — explicit tile-level operations
> T.tile.add(C_ub, A_ub, B_ub)
> ```
>
> **Automatic Cube/Vector Scope Separation:**
> Cube and vector cores on A2/A3 exchange data through global memory/L2 cache. In **Developer mode**, the compiler automatically separates Cube/Vector scopes and inserts synchronization. In **Expert mode**, you explicitly specify execution scopes using `T.Scope("C")/T.Scope("V")` and manage synchronization via `T.set_cross_flag` / `T.wait_cross_flag`.

This last paragraph is the single most important sentence for a hand-written megakernel:
*Cube and vector cores on A2/A3 exchange data through global memory / L2 cache.* There is no
shared on-chip buffer between AIC and AIV. Any AIC→AIV handoff inside one kernel round-trips
L2 at minimum.

### 3.2 The full intra-core flag protocol, from the high-performance GEMM example

Verbatim from the README's "High Performance GEMM Example. (Layout, L2 Cache Swizzling, and
Pipelining, etc.)". The `set_flag`/`wait_flag` pipe names (`mte1`, `mte2`, `m`, `fix`) are the
AscendC pipes; `S1`/`S2` are the L1 and L0 multi-buffer depths.

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, K_L1, S1, S2, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    core_num = 20

    @T.macro
    def init_flag():
        T.set_flag("mte1", "mte2", 0)
        T.set_flag("mte1", "mte2", 1)
        T.set_flag("m", "mte1", 0)
        T.set_flag("m", "mte1", 1)
        T.set_flag("fix", "m", 0)

    @T.macro
    def clear_flag():
        T.wait_flag("mte1", "mte2", 0)
        T.wait_flag("mte1", "mte2", 1)
        T.wait_flag("m", "mte1", 0)
        T.wait_flag("m", "mte1", 1)
        T.wait_flag("fix", "m", 0)

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)

            T.annotate_layout({
                A_L1: make_zn_layout(A_L1),
                B_L1: make_zn_layout(B_L1),
            })

            A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                init_flag()

                for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
                    T.use_swizzle(
                        i * core_num + cid, M, N, K, block_M, block_N, off=3, in_loop=True)
                    bx = cid // n_num
                    by = cid % n_num

                    loop_k = T.ceildiv(K, K_L1)

                    T.wait_flag("mte1", "mte2", 0)
                    T.copy(A[bx * block_M, 0], A_L1[0, :, :])
                    T.copy(B[0, by * block_N], B_L1[0, :, :])
                    T.set_flag("mte2", "mte1", 0)
                    T.wait_flag("fix", "m", 0)
                    for k in T.serial(loop_k):
                        if k < loop_k - 1:
                            T.wait_flag("mte1", "mte2", (k + 1) % S1)
                            T.copy(A[bx * block_M, (k + 1) * K_L1], A_L1[(k + 1) % S1, :, :])
                            T.copy(B[(k + 1) * K_L1, by * block_N], B_L1[(k + 1) % S1, :, :])
                            T.set_flag("mte2", "mte1", (k + 1) % S1)

                        loop_kk = T.ceildiv(K_L1, block_K)

                        for kk in T.serial(loop_kk):
                            if kk == 0:
                                T.wait_flag("mte2", "mte1", k % S1)
                            T.wait_flag("m", "mte1", kk % S2)
                            T.copy(A_L1[k % S1, 0, kk * block_K], A_L0[kk % S2, :, :])
                            T.copy(B_L1[k % S1, kk * block_K, 0], B_L0[kk % S2, :, :])
                            if kk == 3:
                                T.set_flag("mte1", "mte2", k % S1)
                            T.set_flag("mte1", "m", kk % S2)
                            T.wait_flag("mte1", "m", kk % S2)

                            if k == 0 and kk == 0:
                                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, init=True)
                            else:
                                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0)

                            T.set_flag("m", "mte1", kk % S2)

                    T.set_flag("m", "fix", 0)
                    T.wait_flag("m", "fix", 0)
                    T.copy(C_L0, C[bx * block_M, by * block_N])
                    T.set_flag("fix", "m", 0)

                clear_flag()

                T.barrier_all()

    return main
```

### 3.3 A real on-chip address map — the only concrete UB/L1/L0C byte offsets I found

From the "Automatic Buffer Reuse" section, quoting the manual
`T.annotate_address` map that the new pass replaces. This is a working sparse-attention kernel
(heads=128, dim=512, tail_dim=64, topk=2048), so the offsets are real.

```python
    # T.annotate_address({
    #             # L1 address
    #             q_l1: 0,
    #             q_tail_l1: 65536,
    #             kv_l1: 73728,
    #             kv_tail_l1: 139264,
    #             acc_s_l1: 139264,

    #             # L0C address
    #             acc_s_l0c: 0,
    #             acc_o_l0c: 0,

    #             ## ub address
    #             acc_o: 0,
    #             sumexp: 65536,
    #             m_i: 65664,
    #             indices_ub_: 65792,
    #             kv_ub: 66048,
    #             kv_tail_ub: 67072,
    #             acc_s_ub: 66048,
    #             m_i_prev: 74240,
    #             acc_s_ub_: 74368,
    #             tmp_ub: 74368,
    #             sumexp_i_ub: 98944,
    #             acc_s_half: 98944,
    #             acc_o_ub: 98944,
    #             acc_o_half: 98944
    #         })
```

Enabled by:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True
}
```

### 3.4 The sparse-attention operator signature and workspace set

This is the closest public analogue to our `SparseFlashAttention` (6.973 ms/round measured vs
1.382 ms of bytes), including the exact `topk=2048` we use for `index_topk`.

```python
# Specify workspace positions in parameter list via workspace_idx
@tilelang.jit(out_idx=[3], workspace_idx=[4,5,6,7,8])
def sparse_attention_fwd(...):
    @T.prim_func
    def main(
            # --- Input tensors ---
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),

            # --- Auto-allocated output (index 3 in out_idx) ---
            Output: T.Tensor(o_shape, dtype),

            # --- Auto-allocated workspaces (indices 4-8 in workspace_idx) ---
            # These are temporary buffers managed by the runtime
            workspace_1: T.Tensor([block_num, BI, D], dtype),
            workspace_2: T.Tensor([block_num, BI, D_tail], dtype),
            workspace_3: T.Tensor([block_num, H_per_block, BI], accum_dtype),
            workspace_4: T.Tensor([block_num, H_per_block, BI], dtype),
            workspace_5: T.Tensor([block_num, H_per_block, D], accum_dtype),
    ):

    ...

# Instantiate sparse attention function
func = sparse_attention_fwd(
    heads=128,
    dim=512,
    tail_dim=64,
    topk=2048,
    kv_stride=1,
)
```

### 3.5 Cube↔Vector handoff through workspace, and the "Workspace Reduction" pass

> ### Workspace reduction
>
> Workspace Reduction is **automatically enabled** in the compilation pipeline. No manual configuration required. It automates workspace allocation for cross-core data transfer (L0C → UB), eliminating the need for users to manually write two-stage GM copy scripts and manage workspace buffers.
>
> ```python
> @T.prim_func
> def flash_attn(...):
>     with T.Kernel(..., threads=2, is_npu=True) as (cid):
>         # Cube computation
>         T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
>
>         # Single copy statement — Workspace Reduction automatically handles GM transfer
>         T.copy(acc_s_l0c, acc_s_ub)
>
>         # Vector computation
>         T.tile.exp(acc_s_ub, acc_s_ub)
>         T.reduce_max(acc_s_ub, m_i, dim=-1)
>         T.tile.sub(acc_s_ub, acc_s_ub, m_i)
> ```

The explicit, unsugared form of the same handoff, from the inter-core pipeline example
(`flash_attn_bshd_pipeline.py`) — note `workspace_1[cid, ...]` is in GM:

```python
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])

    T.tile.fill(acc_s_ub, 0.0)
    T.copy(m_i, m_i_prev)
    T.copy(
        workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
        acc_s_ub_)
    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
    ...
```

### 3.6 Software pipelining constraints

> ### T.Pipelined
> [T.Pipelined] enables automatic pipeline scheduling to achieve intra-core computation and data movement overlap, as well as inter-core pipeline overlap between Cube and Vector units.
>
> **Usage:**
> ```python
> for i in T.Pipelined(loop_range, num_stages=N):
>     # loop body
> ```
>
> **Usage Constraints:**
> - `T.Pipelined` **does not support nesting**. You cannot place a `T.Pipelined` loop inside another `T.Pipelined` loop.
> - For inter-core pipelining, place both Cube and Vector operations in a single `T.Pipelined` loop.
> - For intra-core pipelining only, use separate `T.Pipelined` loops for Cube and Vector respectively.

### 3.7 Automatic cross-core flag insertion (CrossCoreSetFlag / CrossCoreWaitFlag)

> ### Automatic insert synchronization flags between AIC and AIV, such as CrossCoreSetFlag / CrossCoreWaitFlag.
>
> Two switches need to be turned on:
> ```python
> pass_configs = {
>     tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
>     tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
> }
> ```
>
> Here is an example:
> - [FlashAttention](./examples/flash_attention/flash_attn_bhsd_cc_sync.py): Implementations of FlashAttention without inserting synchronization flags manually.

> ### Automatic insertion of synchronization instruction
>
> We have supported automatic insertion of synchronization instructions within the core, which can be enabled by setting the TL_ASCEND_AUTO_SYNC attribute in the JIT's pass_configs parameter.

### 3.8 The AIC:AIV ratio control, and `vid` elimination

> ### Vid reduction & Auto CV Ratio
>
> The parameter `threads` needs to be set. (Only 1 or 2 are allowed).When setting thread parameters, the return value will only have cid and no vid：
> ```python
> with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
> ```
> Therefore, ub application and transfer will no longer need to consider core allocation, and the usage will change as follows:
> ```python
> # UB allocation original form
> c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
> # New form after UB reduce elimination
> c_ub = T.alloc_shared((block_M, block_N), dtype)
> # UB moved to its original form
> T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
> # New form after UB transport elimination
> T.copy(c_ub, C[bx * block_M, by * block_N])
> ```

The `threads` value is the AIV-per-AIC count, i.e. the MIX ratio: `threads=2` is the 1:2 mode
named `MIX_AIC_1_2` in CANN's SuperKernel enum (see `megakernel-persistent-decode.md` §4.7).

### 3.9 Feature timeline, verbatim — tells us what landed when

> ## Latest News
> - 04/24/2026 🚀: Released DeepSeek V4 kernels [DeepSeek-V4](./examples/deepseek_v4/)
> - 03/28/2026 🚀: Released high-performance [Flash Attention](./examples/flash_attention/) & [Sparse Flash Attention](./examples/sparse_flash_attention/) benchmark and optimization guide, see [PR#698](https://github.com/tile-ai/tilelang-ascend/pull/698) and [PR#665](https://github.com/tile-ai/tilelang-ascend/pull/665).
> - 03/16/2026 🚀: Introduced [wheel package installation](...), enabling easy setup via `pip install`!
> - 03/12/2026 ✨: New [ACLGraph](./examples/aclgraph/) integration example for graph-level optimization on Ascend NPU.
> - 03/03/2026 ✨: Supported [shared memory (shmem) with put/get API](https://github.com/tile-ai/tilelang-ascend/pull/194) for inter-core communication.
> - 01/29/2026 📖: Published the [TileLang-Ascend Programming Guide](./docs/TileLang-Ascend%20Programming%20Guide.md) with comprehensive development tutorials.
> - 01/23/2026 🚀: Landed [PTO backend](https://github.com/tile-ai/tilelang-ascend/pull/341) as a new code generation target.
> - 01/21/2026 ✨: New [torch_tl_ascend](./examples/torch_tl_ascend/) PyTorch integration example for seamless NPU + PyTorch workflows, see [PR#292](https://github.com/tile-ai/tilelang-ascend/pull/292).
> - 01/15/2026 ✨: Introduced [T.Pipelined](https://github.com/tile-ai/tilelang-ascend/pull/274) for software pipelining on Ascend NPU.
> - 12/08/2025 ✨: Supported [T.Parallel](...) for automatic vectorization, check out [PR#113](https://github.com/tile-ai/tilelang-ascend/pull/113) for details.
> - 11/25/2025 ✨: Enabled [automatic buffer reuse](...) to reduce on-chip memory footprint, see [PR#101](https://github.com/tile-ai/tilelang-ascend/pull/101).
> - 11/17/2025 🛠️: Shipped debug tools `T.printf` and `T.dump_tensor` for [printing and dumping](...) device-side buffers.
> - 11/07/2025 ✨: Enabled [automatic intra-kernel synchronization insertion](...), see [PR#74](https://github.com/tile-ai/tilelang-ascend/pull/74).
> - 10/28/2025 🚀: Optimized tl_templates performance and delivered a high-performance [GEMM kernel](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/gemm/example_gemm_intrinsic.py).
> - 09/29/2025 🎉: tilelang-ascend is now open source! A DSL for high-performance AI workloads on Ascend NPUs.

> ## Tested Devices
> Although tilelang-ascend aims to be portable across a range of Ascend devices, it has been specifically tested and validated on the following NPUs: A2 and A3.

### 3.10 Example inventory — the kernels that already exist on Ascend

`gh api repos/tile-ai/tilelang-ascend/contents/examples?ref=ascendc_pto`:

```
HISA                              causal_conv1d              gqa_fwd_varlen            quant_batch_matmul
aclgraph                          compile_flags              lightning_indexer         reduce
activation                        cross_entropy_loss         linear_attention_and_rnn  shmem
add_rms_norm_dynamic_quant        deepseek_v4                mha_sink_fwd_bhsd         simple_fusion
attention_sink                    developer_mode             mhc_post                  softmax
autotune                          dispatch_combine           moe_token_permute         sort
batch_gemm                        dynamic_quant              normalization             sparse_flash_attention
bench_test.sh                     elementwise                pad                       tail_mask
cann-bench                        exception_dump_test        pipeline                  tile_kernels
                                  flash_attention            pos_embedding             topk_selector
                                  fused_sigmoid_gating_delta_rule                      torch_tl_ascend
                                  fusedmoe                   print                     unsorted_segment_sum
                                  gemm / gemm_aot / gemv     generative_recommendation xattention / xllm_kernels
```

Sizes of the most relevant files:

```
examples/fusedmoe/example_fusedmoe.py            50779 bytes
examples/fusedmoe/test_fusedmoe.py               52615 bytes
examples/deepseek_v4/lightning_indexer.py        83300 bytes
examples/deepseek_v4/sparse_attention.py          9537 bytes
examples/deepseek_v4/int8_gemm.py                 5857 bytes
examples/deepseek_v4/act_quant.py                 4930 bytes
examples/deepseek_v4/hc_split_sinkhorn.py         8527 bytes
examples/shmem/shmem_mm_test_put_nbi.py           3193 bytes
examples/shmem/shmem_mm_test_get_nbi.py           3197 bytes
examples/shmem/shmem_mm_test_ub_put_nbi.py        3308 bytes
examples/shmem/shmem_mm_test_ub_get_nbi.py        3310 bytes
examples/simple_fusion/matmul_add.py              2683 bytes
examples/aclgraph/rms_rope_aclgraph.py            9568 bytes
examples/tile_kernels/{mhc,moe,quant}/            (directories)
```

`examples/shmem/` with `put_nbi` / `get_nbi` naming is a non-blocking one-sided
put/get API on Ascend — relevant to `low-latency-tp-reduce.md`; PR #194 is the reference.

## Relevance

Three things here bear directly on our hand-written AscendC work regardless of the TileLang
decision. First, the explicit statement that AIC and AIV exchange data only through GM/L2 on
A2/A3 — so our megakernel's inter-stage activation handoff costs an L2 round trip, the Ascend
analogue of the 250 us of 600 us Hazy Research spends on "store activations, await consistency,
load them". Second, the §3.3 address map is the only concrete on-chip byte geometry found: L1
offsets up to 139264 and UB offsets up to 98944 in a live heads=128/dim=512/topk=2048 sparse
attention kernel, which is the same topk our `index_topk` uses. Third, the set_flag/wait_flag
pipe quadruple (`mte2→mte1`, `mte1→m`, `m→mte1`, `m→fix`, `fix→m`) with modulo-S1/S2 buffer
indices is the complete software-pipelining idiom we must write by hand, and `T.Pipelined`'s
"does not support nesting" constraint tells us the compiler cannot nest it either.

## Gaps

- Did not read `examples/fusedmoe/example_fusedmoe.py` (50 KB) or
  `examples/deepseek_v4/lightning_indexer.py` (83 KB) in full. These are the two most relevant
  bodies of code on the repo and should be fetched and read directly rather than summarized:
  `https://raw.githubusercontent.com/tile-ai/tilelang-ascend/ascendc_pto/examples/fusedmoe/example_fusedmoe.py`
- Did not fetch `docs/TileLang-Ascend Programming Guide.md`, which is the official programming
  model document, nor `examples/flash_attention/fa_opt/docs/optimization_guide_zh.md` (the
  num_stages / double-buffer tuning guide).
- Did not fetch PR #194 (shmem put/get), PR #698 / #665 (FA and sparse-FA optimization), or
  issue #3 (the development plan).
- `tile-ai/tilelang-mlir-ascend` README was fetched but not read; it is the second, MLIR-based
  Ascend adapter and may carry a different op inventory.
- No TileRT-Ascend-specific material found anywhere public: `tile-ai/TileRT` is the CUDA/ROCm
  runtime, and no Ascend backend for it is announced. Nothing on a TileLang-generated Ascend
  *megakernel* (one kernel across layers) exists in any of these repos as far as the example
  inventory shows.
