# Low-latency TP reduction — verbatim source material

Collected 2026-10-20 for the GLM-5.2-W4A8C8 / Ascend A3 / TP16 / batch=1 ultra-low-latency effort.
Context this material is judged against: a decode round is 53.05 ms unprofiled; our one-shot push "SuperKernel" TP reduction costs 5.35 ms/round and is 100% serial with compute, while every HCCL collective fully overlaps and is free; per-call launch floors AI_VECTOR_CORE 0.70 us, AI_CORE 2.30 us, MIX_AIV 12.04 us, MIX_AIC 13.58 us; ~3008 kernels/round.

Everything below the heading is quoted verbatim from the named source. Chinese text is left in Chinese.

---

## Part 1 — One-shot / push-based allreduce for tiny messages

### vLLM `custom_all_reduce.cuh` — one-shot (`cross_device_reduce_1stage`) and two-shot kernels

https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh
(vLLM project, `main` as of 2026-10-20; fetched from `raw.githubusercontent.com/vllm-project/vllm/main/csrc/custom_all_reduce.cuh`)

```cuda
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}
```

The strategy selection in the same file (`CustomAllreduce::allreduce`):

```cuda
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }
...
    if (force_1stage) {                                 \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else if (fully_connected_) {                    \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }
```

CUDA-graph note from the same class (why peer pointers cannot be kernel args at capture time):

```cuda
  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
```

## Relevance
This is the exact structure of our SuperKernel push reduce: barrier, read all 16 peer buffers, sum, barrier. Note the crossover thresholds: at world_size <= 8 vLLM leaves one-shot at **256 KB**, and at 16 ranks (our TP16) the `fully_connected_` branch has no clause at all — upstream vLLM does not use a custom one-shot beyond 8 ranks. Our 16-die one-shot is outside the regime anyone upstream tuned.

---

### vLLM `custom_collective_common.cuh` — the signal/flag exchange and barrier cost

https://github.com/vllm-project/vllm/blob/main/csrc/custom_collective_common.cuh
(vLLM project, `main` as of 2026-10-20)

```cuda
namespace vllm {
constexpr int kMaxCustomCollectiveRanks = 16;

// Maximal number of blocks in allreduce kernel.
constexpr int kMaxBlocks = 36;

// Default number of blocks in allreduce kernel.
#ifndef USE_ROCM
inline constexpr int defaultBlockLimit = 36;
inline CUpointer_attribute rangeStartAddrAttr =
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#else
inline constexpr int defaultBlockLimit = 16;
inline hipPointer_attribute rangeStartAddrAttr =
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#endif

// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;

// Two sets of peer counters are needed for two syncs: starting and ending an
// operation. The reason is that it's possible for peer GPU block to arrive at
// the second sync point while the current GPU block haven't passed the first
// sync point. Thus, peer GPU may write counter+1 while current GPU is busy
// waiting for counter. We use alternating counter array to avoid this
// possibility.
struct Signal {
  alignas(128) FlagType start[kMaxBlocks][kMaxCustomCollectiveRanks];
  alignas(128) FlagType end[kMaxBlocks][kMaxCustomCollectiveRanks];
  alignas(128) FlagType _flag[kMaxBlocks];  // incremental flags for each rank
};

struct __align__(16) RankData {
  const void* ptrs[kMaxCustomCollectiveRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxCustomCollectiveRanks];
};
```

```cuda
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  #else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
  #endif
  return flag;
}

// This function is meant to be used as the first synchronization in the all
// reduce kernel. Thus, it doesn't need to make any visibility guarantees for
// prior memory accesses. Note: volatile writes will not be reordered against
// other volatile writes.
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value
    // from peer.
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// This function is meant to be used as the second or the final
// synchronization barrier in the all reduce kernel. If it's the final
// synchronization barrier, we don't need to make any visibility guarantees
// for prior memory accesses.
template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();

  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}
```

The ROCm branch states the cost model explicitly:

```cuda
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                            flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}
```

And the reduce itself:

```cuda
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}
```

## Relevance
`Signal` is sized `[36][16]` per phase — the flag matrix is per-block × per-rank, i.e. a 16-rank one-shot pays 16 p2p flag writes per block per barrier, and vLLM's own comment prices a barrier at "Latency = 1 p2p write" only because all 16 writes are issued simultaneously by 16 distinct threads. If our 5.35 ms SuperKernel is barrier-bound rather than bandwidth-bound, this is the structure to compare against: two barriers per reduce, one p2p write-round-trip each.

---

### TensorRT-LLM `customAllReduceKernels.h` — strategy enum, Lamport thresholds, rank limits

https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/customAllReduceKernels.h
(NVIDIA, Copyright 2022-2024, `main` as of 2026-10-20)

```cpp
constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
// Use max modules to avoid overflow and ABA problem when the block num changes for barrier_flag
// Not a perfect solution, but it has large chance that it is correct
constexpr size_t MAX_ALL_REDUCE_MODULES = std::numeric_limits<uint32_t>::max() / 6 * 6;
constexpr size_t MAX_RANKS_PER_NODE = 16;
constexpr size_t DEFAULT_BLOCK_SIZE = 512;

namespace reduce_fusion::details
{
static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;
static constexpr int kMaxCtaSize = 1024;
static constexpr int kClusterMaxSize = 8;
static constexpr int kLamportTokenNumThreshold = 16;
static constexpr int kLamportHiddenSizeThreshold = 256;
}; // namespace reduce_fusion::details

// Warning: python definition is in tensorrt_llm/functional.py
// they must be kept in sync
enum class AllReduceStrategyType : int8_t
{
    NCCL = 0,
    MIN_LATENCY = 1,
    UB = 2,
    AUTO = 3,
    ONESHOT = 4,
    TWOSHOT = 5,
    LOWPRECISION = 6,
    MNNVL = 7,
    NCCL_SYMMETRIC = 8,
};

enum class AllReduceStrategyConfig : int8_t
{
    USE_MEMCPY = 1 << 0,
    PUSH_MODE = 1 << 1,
};

enum class AllReduceFusionOp : int8_t
{
    NONE = 0,
    RESIDUAL_RMS_NORM = 1,
    LAST_PROCESS_FOR_UB = 2,
    RESIDUAL_RMS_PREPOST_NORM = 3,
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4,
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5,
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6,
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7,
    MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM = 8,
    RMS_NORM = 9,
};
```

```cpp
struct AllReduceFusionParams
{
    ...
    // gemm bias
    void const* bias_buffer;
    // residuial add
    void const* residual_buffer;
    // rms norm
    int hidden_size;                             // equal to normalized_shape
    void const* weight_buffer;                   // norm elem-wise affine gamma
    void const* weight_buffer_pre_residual_norm; // for gemma norm before residual
    float eps;
    // new residual
    void* intermediate_buffer;
    void* lamport_peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE * 3];
};

struct AllReduceParams
{
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    size_t ranks_per_node;
    size_t local_rank;
    uint32_t* barrier_flag_ptr;
    uint32_t* barrier_flag_counter_ptr;
    uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
    uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE * 2];
    void* local_output_buffer_ptr;
    void const* local_input_buffer_ptr;

    AllReduceFusionParams fusion_params;

    static AllReduceParams deserialize(int64_t* buffer, size_t tpSize, size_t tpRank, tensorrt_llm::DataType dataType,
        int token_num, int hidden_size, AllReduceFusionOp op);
};

bool configurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t n_ranks, tensorrt_llm::DataType type);

void customAllReduce(kernels::AllReduceParams& params, tensorrt_llm::DataType dataType, AllReduceStrategyType strat,
    AllReduceStrategyConfig config, AllReduceFusionOp fusionOp, cudaStream_t stream);

void residualRmsNorm(
    kernels::AllReduceParams& params, tensorrt_llm::DataType dataType, cudaStream_t stream, AllReduceFusionOp fusionOp);

void lamportInitialize(void* buffer, size_t size, tensorrt_llm::DataType dataType, cudaStream_t stream);

namespace reduce_fusion
{
bool is_lamport_supported(tensorrt_llm::DataType dataType, int token_num, int hidden_size);
}
```

## Relevance
`PUSH_MODE` is a first-class config flag alongside `USE_MEMCPY`, and `MIN_LATENCY` is a *separate* strategy from `ONESHOT` — the Lamport (flag-free, value-as-signal) path gated at `kLamportTokenNumThreshold = 16` tokens and `kLamportHiddenSizeThreshold = 256` hidden. Our batch=1 decode is exactly 1 token, deep inside the Lamport-eligible region. Also `MAX_RANKS_PER_NODE = 16` matches our 16 dies, and the reduce has 10 fusion ops available including `RESIDUAL_RMS_NORM` — i.e. NVIDIA fuses the post-reduce epilogue into the collective rather than launching it separately.

---

### TensorRT-LLM perf cookbook — measured Lamport one-shot vs NCCL-symmetric per-call latency

https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-regression-cookbook/references/communication/allreduce-host-overhead-small-model-tp.md
(NVIDIA TensorRT-LLM in-repo regression cookbook; case `case-allreduce-host-overhead-small-model-tp`, nvbug 6262973, commit `c0e6d795c8d4`, PR #16902 merged 2026-08-11)

```text
- **Failed attempts:** PR **#15157** (bot-filed by `tensorrt-cicd`,
  2026-06-09) — still **OPEN** but **superseded by the merged #16902**
  (untouched since 2026-06-22, no human review at all, only a CodeRabbit
  comment; the code it wanted to revert has since been restructured into
  C++). It attacked the same host overhead by *reverting* rather than
  relocating: (1) flip `_NCCL_SYMMETRIC_ZERO_COPY` back from `"1"` to `"0"`
  in `tensorrt_llm/_torch/distributed/ops.py`; (2) drop
  `inputs_pre_hook_register_nccl_symmetric_memory_window` from
  `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` so the AutoTuner
  benchmarks each tactic on the real production input instead of a
  pre-copied window buffer; (3) change the cache-miss fallback from
  `NCCL_SYMMETRIC.value` to `NCCL.value`. Its own diagnosis is worth
  keeping — the post-fix profile still showed "322 ncclSymk calls @ 473us
  = 152ms" against "Lamport one-shot (4347 calls @ 7.3us = 31.6ms)", i.e.
  the `inputs_pre_hook` "hides NCCL_SYMMETRIC's in-kernel
  cudaMemcpyAsync cost" so the tuner picks it for shapes where
  ONESHOT/TWOSHOT would win — and it reported `total_token_throughput`
  `1.785e+04` bad → `1.863e+04` after fix against a `1.926e+04` good, i.e.
  it recovered only part of the gap. Two files (11/−25 and 6/−3 lines) are
  the whole diff. Lesson: reverting
  the zero-copy default was tried and not taken — the accepted fix moved
  the tactic dispatch to C++ and left the default alone.
```

```text
- **Root cause:** host-side, not kernel-side. Two contributions, both
  public: (1) the zero-copy default makes the GEMM output window-resident
  for NCCL_SYMMETRIC (PR #15157), so registered-NCCL-buffer bookkeeping
  crosses the C++/Python boundary per collective, and the AutoTuner's
  `inputs_pre_hook` pre-copies the input into the NCCL window before each
  tactic benchmark, "hiding NCCL_SYMMETRIC's in-kernel cudaMemcpyAsync
  cost" so the tuner picks it where ONESHOT/TWOSHOT would win (PR #15157);
  and (2) a pre-existing tax — under `AllReduceStrategy.AUTO` the per-call
  bucket lookup and tactic dispatch ran in Python
  (`torch.ops.trtllm.tunable_allreduce`) on every AllReduce, which PR
  #16902 moves to C++ "avoiding Python work during inference". Both only
  matter where the model is small enough that the host is the critical
  path; PR #16902 states it "benefits small, CPU-overhead-sensitive models".
```

```text
- **Fix mechanism:** PR #16902 moves the hot-path AllReduce autotuner bucket
  lookup and tactic dispatch out of Python into C++. It adds
  `autotuned_allreduce` plus a thread-safe native tactic cache in
  `cpp/tensorrt_llm/thop/allreduceOp.cpp` (14 fixed token-bucket cutoffs;
  key = tensor group, fusion-op id, NCCL-window vs non-window mode, input
  scalar dtype, static input shape excluding `size(0)`), with
  `register_allreduce_tactic`, `validate_allreduce_tuning_buckets` and
  `clear_allreduce_tactic_cache` as native control ops.
```

## Relevance
The one hard per-call number in the TensorRT-LLM tree: **Lamport one-shot allreduce = 7.3 us/call** (4347 calls, 31.6 ms total) versus **NCCL symmetric = 473 us/call**. 7.3 us is the order of magnitude a tuned one-shot allreduce should cost. Our 5.35 ms per round, spread over the ~60 TP reductions a GLM-5.2 decode round needs, is ~90 us per reduce — more than 12x the Lamport figure. That gap is the budget.

---

### TensorRT-LLM perf cookbook — AUTO keyed on token count alone picked one-shot where NCCL won

https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-regression-cookbook/references/communication/allreduce-strategy-lut.md
(NVIDIA TensorRT-LLM in-repo regression cookbook; case `case-allreduce-strategy-lut`, nvbugs 5540494 / 5531839 / 5517023, commit `fd4311e6a396`, PR #7870)

```text
# AllReduce AUTO keyed only on token count, so it picked a one-shot kernel where NCCL was faster

- **Symptom:** two directions, both in the PR. Regression: "perf regression due to
  using a one-shot kernel instead of NCCL on A100/H100" — AUTO chose a custom
  kernel that lost to plain NCCL on those SMs. Upside on the fixed path: "3-4 %
  perf gain in concurrencies of 256 and 512" on Deepseek-R1.
- **Root cause:** AUTO decided from **token count alone** — a single threshold
  (default 128, overridable) plus a `world_size <= 2` special case — and below the
  threshold always returned `MIN_LATENCY`, i.e. the one-shot custom kernel. The
  inputs it ignored are exactly the ones that determine the crossover:
  `hidden_size` (message bytes per token), the **fusion op** attached to the
  allreduce (which changes how much work the custom kernel is amortising), and the
  **SM architecture** (the custom kernels' advantage over NCCL is
  generation-dependent, and on A100/H100 it can be negative). A one-dimensional
  heuristic over a five-dimensional crossover surface is right in the region it was
  tuned on and wrong everywhere else — and because it is *deterministic*, it is
  wrong reproducibly, which reads as a product regression rather than a tuning gap.
- **Fix mechanism:** a generated lookup table,
  `AllReduceBestStrategyTable[sm][tp][fusion_op][hidden_size][num_tokens]`,
  populated from measurement for SM90 and SM100, with **out-of-range ⇒ NCCL** as
  the safe default. Explicit user requests for `ONESHOT` / `TWOSHOT` are still
  honoured (AUTO is the only path that changes). The kernels themselves also move
  from stronger orderings to `relaxed` stores plus a single
  `fence.release.sys`. What the PR **removes** matters as much: a `TORCH_CHECK`,
  three `TLLM_LOG_WARNING` fallback paths, and the env overrides
  `OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY` and
  `ALLREDUCE_AUTO_HEURISTIC_MIN_LATENCY_THRESHOLD_TOKEN_NUM` — so a runbook or
  script that sets either variable is silently a no-op after this commit.
- **Detection signal:** the log line was renamed, which dates any log precisely:
  `grep -n "AllReduceOp runtime strategy for rank" <log>` is post-fix,
  `grep -n "AllReducePlugin strategy for rank" <log>` is pre-fix.
```

## Relevance
Two transferable facts. First, "the kernels themselves also move from stronger orderings to `relaxed` stores plus a single `fence.release.sys`" — the memory-ordering strength of the barrier is itself a measurable lever on a tiny-message one-shot. Second, NVIDIA's own conclusion is that a custom one-shot is *not* unconditionally better than the vendor collective; the decision surface is `[sm][tp][fusion_op][hidden_size][num_tokens]`. Our situation — one collective that is 100% serial while HCCL is free — means the comparison on A3 is not even about kernel time but about overlap, which this table does not model.

---

### TensorRT-LLM perf casebook — MNNVL two-shot, Lamport flag sync, multicast loads

https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/references/communication/mnnvl-twoshot-allreduce.md
(NVIDIA TensorRT-LLM in-repo optimization casebook; commit `6e1aee6fd68a`, PR #5934)

```text
# Optimize the collective kernel itself (two-shot / Lamport-sync MNNVL AllReduce)

- **Commits:** `6e1aee6fd68a` [fix] Performance Optimization for MNNVL TwoShot Kernel (#5934).
- **Applies when:** communication-bound + signals: TP AllReduce on a multi-node NVLink (MNNVL) domain; medium/large messages where two-shot (reduce-scatter + allgather) beats one-shot; `allreduce_strategy=MNNVL` or `TWOSHOT` already selected and the AR kernel (not the network) is the bottleneck.
- **Mechanism:** tunes the MNNVL two-shot AllReduce CUDA kernel rather than swapping algorithms. Lamport-style flag sync with triple-buffered flags (`buffer_flags % 3`), grid reworked so only a subset of CTAs do the lamport sync; cross-CTA completion via `red.async.release.global.gpu.add.u32` atomics + spin on `offset_access_ptr < gridDim.x*gridDim.y`; data via vectorized `ld.volatile.global.v4.f32`/`v2.f32` over NVLink-multicast buffers (`mcastDeviceMemory`), assuming 8B-atomic writes and `num_token*token_dim` divisible by ELTS_PER_LOAD (4 BF16 / 2 FP32).
- **Generalizes to:** "make the collective kernel faster instead of changing the algorithm" — buffer-flag/Lamport sync, grid-rightsizing, vectorized multicast loads carry to one-shot AllReduce, allgather/reduce-scatter, custom NVLS/multimem collectives; adapt by matching alignment/divisibility assumptions and the flag lifecycle.
- **Apply via:** `allreduce_strategy=MNNVL` (or `TWOSHOT`) in `_torch/model_config.py` (`AllReduceStrategyType`: NCCL=0, MIN_LATENCY=1, UB=2, AUTO=3, ONESHOT=4, TWOSHOT=5, LOWPRECISION=6, MNNVL=7, NCCL_SYMMETRIC=8).
- **Eligibility:** "num_token*token_dim divisible by ELTS_PER_LOAD (4 for BF16, 2 for FP32), as of #5934"
```

And the sibling autotune case:

https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/references/communication/shape-aware-allreduce-autotune.md
(commits `22257457824a` PR #7870, `d272f1a9bcbc` PR #8531)

```text
- **Mechanism:** make the algorithm a function of shape. #7870 adds two `AUTO` heuristics in `customAllReduceUtils.h`: an LP threshold (`SelectStrategyLP`: per-`(SM_major, TP_size)` `(NCCL_num_token_threshold, TWO_SHOT_numel_threshold)` → TWOSHOT if `message_size ≥ two_shot_numel_threshold`, else ONESHOT, else NCCL) and a benchmark-derived LUT (`selectStrategyLookUpTable`, indexed `[sm][tp][fusionOp][log2(hidden)-7][log2(tokens)]`) generated offline by `allreduce_heuristic_code_gen.py`. Also fixes that ONESHOT/TWOSHOT couldn't be overridden, fixes a TWOSHOT perf bug, cleans dispatch. #8531 wires it into the runtime AutoTuner via `AllReduceRunner(TunableRunner)` + `trtllm::tunable_allreduce`, with `AutoTuner.choose_one` picking per-profile.
- **Expected effect:** recovers per-shape best AllReduce time, removes the one-shot-vs-NCCL regression. **PR #7870 reports ~3–4% end-to-end gain for DeepSeek-R1 at concurrency 256 and 512** (fixes a known one-shot regression). Re-measure on your shapes.
- **Knobs:** `allreduce_strategy`, `TLLM_DISABLE_ALLREDUCE_AUTOTUNE`, `OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY`
```

## Relevance
Triple-buffered Lamport flags (`buffer_flags % 3`) and "grid reworked so only a subset of CTAs do the lamport sync" are two concrete, portable reductions in barrier cost. The second is directly applicable: if only some AI cores participate in the cross-die sync, the barrier's fan-in shrinks from (cores × 16) to (sync-cores × 16).

---

## Part 2 — Ascend: symmetric memory, device-side put, compute-comm fusion

### ACLSHMEM / `cann/shmem` — Ascend symmetric-memory communication library (the Ascend answer to NVSHMEM)

https://gitcode.com/cann/shmem
(Huawei / CANN SIG `shmem`; Copyright (c) 2025 Huawei Technologies Co., Ltd.; v1.7.0 released October 2026; docs at https://shmem-doc.pages.dev/)

```text
# SHMEM
Symmetric Memory-based Ascend Distributed Memory Communication Acceleration Library

## What's New

🚀 [October 2026] SHMEM v1.7.0 is now available.

   - Added QP-specific put/get APIs for Ascend 950PR&950DT products RDMA/UDMA, expanded aggregate submission support, and enabled concurrent UDMA communication across multiple links.
   - Added QP-specific put/get APIs for A2/A3 SDMA to support multi-QP communication.
   - Improved installation and runtime diagnostics, with new MegaMoE and UDMA AllReduce examples and multi-QP performance tests.

📌 [July 2026] SHMEM v1.6.0: expanded Ascend 950PR&950DT products RDMA, UDMA/MTE, and SIMT capabilities, and improved Python wheel installation and `shmem-config` diagnostics.

📌 [April 2026] SHMEM v1.3.0: added AI Core direct-driven capabilities and 40+ communication APIs, with expanded DFX tooling.

🔥 [December 2025] Initial launch of the SHMEM project.
```

```text
## 1. Project Introduction
SHMEM is a multi-server, multi-device memory communication library designed for the Ascend platform. By abstracting host-side and device-side APIs, SHMEM enables fast cross-device memory access and data synchronization. Its core benefits include:
- Support for AI Core direct-driven MTE and xDMA, enabling D2D, D2H, H2D, D2rH, and rH2D communication paths
- Simplified inter-device communication logic in distributed workloads, lowering operator development complexity
- Deep integration with the CANN ecosystem, accelerating deployment of computing-communication fusion operators

## 2. Core Functions

**1. Dual-side API system**

- Host side: initialization, memory heap management, communicator (team) creation, and global synchronization
- Device side: remote memory access (RMA), device-level synchronization, and team operations

The API design follows the Ascend operator development paradigm and supports host-device collaboration.

**2. High-performance communication optimization**

- Built-in MTE and xDMA engines for direct remote memory read/write, minimizing latency
- MPI interoperability, supporting collective communication primitives such as AllGather and AllReduce
- Optimized data transfer paths for Ascend hardware features to improve multi-device collaboration efficiency

**4. Communication path coverage**

The following figure shows the full-link communication paths supported by SHMEM (using Ascend 910A3 as an example), covering different transmission engines on the host and device sides:

- **MTE Engine**: chip-level memory transfer engine, supporting D2D, D2H, H2D, D2rH, and rH2D paths
- **xDMA Engine**: high-speed direct memory access engine, supporting efficient intra-host and inter-host data transfer
```

Code structure (verbatim), showing the device-side data planes:

```text
shmem/                                  # Project root directory
├── include/                            # External header files
│   ├── shmem.h                         # All SHMEM external APIs
│   ├── device/                         # Device-side header files
│   │   ├── gm2gm/                      # Data plane APIs gm2gm driven by AI Core
│   │   │   └── engine/                 # Low-level APIs gm2gm directly driven by AI Core
│   │   ├── team/                       # Device-side team management
│   │   └── ub2gm/                      # Data plane APIs ub2gm driven by AI Core
│   │       └── engine/                 # Low-level APIs ub2gm directly driven by AI Core
│   ├── host/                           # Host-side header files
│   │   ├── data_plane/                 # Host-side data plane APIs
│   │   ├── init/                       # Host-side initialization APIs
│   │   ├── mem/                        # Host-side memory management APIs
│   │   ├── team/                       # Host-side team management APIs
│   │   └── utils/                      # Tools and general auxiliary code
│   └── host_device/                    # Shared directory
├── src/                                # Source code implementation
│   ├── device/                         # Device-side implementation
│   │   ├── gm2gm/                      # Data plane APIs gm2gm directly driven by AI Core
│   │   │   └── engine/                 # Low-level APIs gm2gm directly driven by AI Core
│   │   ├── team/                       # Device-side team management
│   │   └── ub2gm/                      # Data plane APIs ub2gm driven by AI Core
│   │       └── mte/                    # Low-level APIs ub2gm directly driven by AI Core
│   ├── host/                           # Host-side implementation
│   │   ├── sync/                       # Synchronization primitives (barrier/p2p/order)
│   │   ├── transport/                  # Transport layer implementation (such as RDMA, SDMA, and UDMA)
```

```text
## 4. Typical Use Cases

**1. Computing-communication fusion operator development**: Develop custom operators that merge compute and communication (such as Matmul + AllReduce) by leveraging device-side direct memory access APIs. This reduces inter-device data copies and improves operator execution efficiency.

**3. Low-latency inter-device communication**: Employ RDMA-optimized device-side APIs to transfer data among devices in milliseconds, meeting the real-time requirements of latency-sensitive AI inference scenarios.
```

TLS is on by default and must be disabled *before* init:

```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```

```text
**Q4: Why is an encryption failure still reported after TLS is disabled?**

A: Call `aclshmemx_set_conf_store_tls` before `aclshmemx_init_attr`. The TLS configuration cannot be modified after initialization.
```

Install and build:

```bash
python3 -m pip install cann-shmem \
  --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/
shmem-config --version   # Query the installed version
shmem-config --diagnose  # Check native library loading and package integrity
```

```bash
git clone https://gitcode.com/cann/shmem.git
cd shmem
bash scripts/build.sh             # A2/A3 platforms
# bash scripts/build.sh -soc_type Ascend950  # Ascend 950PR&950DT products platform
source install/set_env.sh
```

The umbrella header `include/shmem.h`, which enumerates the whole device-side surface:

```c
#ifndef SHMEM_API_H
#define SHMEM_API_H

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "device/gm2gm/shmem_device_amo.h"
#include "device/gm2gm/shmem_device_cc.h"
#include "device/gm2gm/shmem_device_mo.h"
#include "device/gm2gm/shmem_device_p2p_sync.h"
#include "device/gm2gm/shmem_device_rma.h"
#include "device/gm2gm/shmem_device_so.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_sdma.h"
#include "device/gm2gm/engine/shmem_device_udma.h"
#include "device/ub2gm/shmem_device_rma.h"
#include "device/ub2gm/engine/shmem_device_mte.h"
#include "device/shmem_def.h"
#include "device/team/shmem_device_team.h"

// simt
#if defined(USE_SIMT)
#include "device_simt/gm2gm/shmem_device_simt_rma.h"
#include "device_simt/gm2gm/shmem_device_simt_amo.h"
#include "device_simt/gm2gm/engine/shmem_device_simt_mte.h"
#include "device_simt/ub2gm/shmem_device_simt_rma.h"
#include "device_simt/ub2gm/engine/shmem_device_simt_mte.h"
#include "device_simt/team/shmem_device_simt_team.h"
#endif
#endif

#include "host/shmem_host_def.h"
#include "host/mem/shmem_host_heap.h"
#include "host/init/shmem_host_init.h"
#include "host/utils/shmem_host_exception.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_so.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_cc.h"
#include "host/team/shmem_host_team.h"
#include "host/utils/shmem_log.h"

#endif // SHMEM_API_H
```

## Relevance
This is the Ascend equivalent of `torch.distributed._symmetric_memory` + NVSHMEM device-side put, and it exists in the open on CANN. `device/gm2gm/engine/shmem_device_mte.h` and `shmem_device_sdma.h` are AI-Core-driven direct remote writes — i.e. exactly the push primitive our SuperKernel needs, with a maintained `p2p_sync` and device-side `team` barrier next to it. `ub2gm/` matters even more for batch=1: it writes from Unified Buffer straight to a peer's global memory, skipping the local GM round trip.

---

### ACLSHMEM device-side RMA — `aclshmem_*_put_nbi` / `get_nbi` exact signatures (UB ⇄ peer GM)

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_rma.h
(Huawei / CANN `cann/shmem`, master as of 2026-10-20; fetched from `raw.gitcode.com/cann/shmem/raw/master/include/device/gm2gm/shmem_device_rma.h`)

Supported element types, as the generator macro:

```c
/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |double     | double    |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 * |bfloat16   | bfloat16  |
 */
#define ACLSHMEM_TYPE_FUNC(FUNC) \
    FUNC(half, half);            \
    FUNC(float, float);          \
    FUNC(double, double);        \
    FUNC(int8, int8_t);          \
    FUNC(int16, int16_t);        \
    FUNC(int32, int32_t);        \
    FUNC(int64, int64_t);        \
    FUNC(uint8, uint8_t);        \
    FUNC(uint16, uint16_t);      \
    FUNC(uint32, uint32_t);      \
    FUNC(uint64, uint64_t);      \
    FUNC(char, char);            \
    FUNC(bfloat16, bfloat16_t)
```

The push primitive — UB straight to a peer PE's symmetric memory, non-blocking:

```c
/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, uint32_t elem_size, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local UB of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                   \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, uint32_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI);
```

AscendC tensor overload, and the non-contiguous (strided) forms:

```c
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, uint32_t elem_size, int32_t pe)
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int32_t pe)

 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, const non_contiguous_copy_param &copy_params, int32_t pe)
 * @par Function Description
 * Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
```

The pull primitive (peer GM → local UB), for a read-based one-shot:

```c
/**
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local UB.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local UB of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                               \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int pe)
```

## Relevance
`aclshmem_bfloat16_put_nbi(__gm__ bfloat16_t *dst, __ubuf__ bfloat16_t *src, uint32_t elem_size, int32_t pe)` is the exact one-sided push our SuperKernel needs, callable from inside an AscendC kernel, with the peer selected by `pe` at runtime (no host round trip, no graph-capture pointer problem). The `__ubuf__` source means the partial sum never has to land in local HBM before being pushed.

---

### ACLSHMEM `shmem_device_mte.h` — `aclshmem_ptr`, MTE put/get with explicit UB staging, device-side atomics

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/engine/shmem_device_mte.h
(Huawei / CANN `cann/shmem`, master as of 2026-10-20)

```c
/**
 * @brief Translate a local symmetric address to the corresponding symmetric address on the specified PE.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] Target PE number.
 * @return The corresponding symmetric address on the specified PE.
 * @note The supported access method for the returned address depends on the transport and topology.
 */
ACLSHMEM_DEVICE __gm__ void* aclshmem_ptr(__gm__ void* ptr, int pe);
#define shmem_ptr aclshmem_ptr
```

```c
/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 * ...
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, const non_contiguous_copy_param& copy_params,
    int pe, uint32_t sync_id);

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);

#define shmem_mte_put_mem_nbi aclshmemx_mte_put_nbi

/**
 * @brief Asynchronous interface. Clear instruction pipes and flush data cache to GM.
 */
ACLSHMEM_DEVICE void aclshmemx_mte_quiet();
```

Device-side remote atomics (the natural reduce-into-peer primitive):

```c
/**
 * @brief Atomic fetch operation. Returns the value at the source address on the specified PE.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch(__gm__ T* src, int32_t pe);

/**
 * @brief Atomic set operation. Sets the value at the destination address on the specified PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_set(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic compare and swap operation. Conditionally updates the value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe);

/**
 * @brief Atomic swap operation. Swaps the value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_swap(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic increment operation. Increments the value at the destination address by 1.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Atomic add operation. Adds the value to the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic fetch increment operation. Increments the value at the destination address by 1 and returns the old
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Atomic fetch add operation. Adds the value to the destination address and returns the old value.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe);
```

## Relevance
`aclshmem_ptr(ptr, pe)` is the Ascend equivalent of the IPC peer-pointer table vLLM has to patch after CUDA-graph capture — it resolves on device, so a captured ACLGraph does not need the peer addresses baked into kernel args. `aclshmemx_mte_atomic_fetch_inc` is enough to build a counter barrier with one remote atomic instead of a 16-wide flag matrix. `aclshmemx_mte_quiet()` is the only completion fence for NBI puts, so it is the thing to count in a latency budget.

---

### ACLSHMEM `shmem_device_cc.h` — grid-wide device-side barrier without returning to host

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_cc.h
(Huawei / CANN `cann/shmem`, master as of 2026-10-20)

The programming contract, verbatim, is the single most important paragraph for a persistent-kernel design:

```c
/*
 * Collective programming contract:
 * - The all-core barrier and sync APIs require a MIX kernel; all participating local AIC and AIV
 *   cores must enter. Vector-only sync APIs involve only AIV cores. The _all forms select
 *   ACLSHMEM_TEAM_WORLD without changing the local core participation scope.
 * - Every PE in a team must call its collectives in the same order. Do not overlap collective
 *   calls on the same team. ACLSHMEM_TEAM_INVALID (-1) is a no-op for team-taking APIs;
 *   other invalid handles are not covered by this rule.
 * - Ordinary symmetric-memory stores from participating cores are synchronized on supported
 *   HCCS/UB paths. The Cube scalar unit is outside this visibility guarantee.
 * - Device collectives cover Device-issued work; Host-issued work needs the corresponding
 *   Host-side completion and ordering.
 */
```

```c
/**
 * @fn ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config)
 * @brief Set runtime ffts address. Call this at MIX Kernel entry point (if the kernel contains barrier calls).
 *
 * @param config              [config] ffts config, acquired by util_get_ffts_config()
 */
ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config);
#define shmemx_set_ffts_config util_set_ffts_config

/**
 * @brief Complete prior communication and synchronize the PEs in a team.
 * @note All local cores in the declared participation scope must enter, and all team PEs
 *       must call collectives in the same order. On return, prior communication covered
 *       by this Device API is complete and its updates are visible to the destination PE.
 *       The source buffer of a completed transfer may then be reused. QP-specific transfers
 *       require their matching QP-specific quiet.
 *
 * @param[in] team Team to synchronize.
 * @return 0 after the collective (also for ACLSHMEM_TEAM_INVALID); Device execution errors
 *         are not reported through this return value.
 */
ACLSHMEM_DEVICE int aclshmem_barrier(aclshmem_team_t team);
#define shmem_barrier aclshmem_barrier

/**
 * @brief Perform aclshmem_barrier over ACLSHMEM_TEAM_WORLD with the same local core scope.
 */
ACLSHMEM_DEVICE void aclshmem_barrier_all(void);
#define shmem_barrier_all aclshmem_barrier_all

/**
 * @brief Collective synchronization over a team in which only vector cores participate. This routine does not
 * ensure
 *        completion of remote memory updates issued through ACLSHMEM communication APIs. Cube cores may call
 * the API
 *        but take no effect.
 *
 * @param team              [in] team to synchronize
 */
ACLSHMEM_DEVICE void aclshmemx_sync_vec(aclshmem_team_t team);

/**
 * @brief aclshmemx_sync_vec over ACLSHMEM_TEAM_WORLD.
 */
ACLSHMEM_DEVICE void aclshmemx_sync_vec_all(void);

/**
 * @brief Wait for all team PEs to reach this synchronization point.
 * @note On supported HCCS/UB paths, ordinary symmetric-memory stores from participating
 *       cores are synchronized within the applicable visibility scope. This call does not
 *       complete outstanding NBI communication. Call the matching quiet before sync when
 *       communication completion or source-buffer reuse is required, or use barrier instead.
 *       All cores in the declared participation scope must enter in the same order.
 *
 * @param[in] team Team to synchronize.
 */
ACLSHMEM_DEVICE void aclshmem_sync(aclshmem_team_t team);

ACLSHMEM_DEVICE int aclshmem_team_sync(aclshmem_team_t team);

ACLSHMEM_DEVICE void aclshmem_sync_all(void);
```

Deprecations worth noting so the wrong name is not used:

```c
[[deprecated("aclshmemx_barrier_vec is deprecated, please use aclshmemx_sync_vec instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t team);

[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmemx_sync_vec_all instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec(void);
```

## Relevance
This is the direct answer to "how can a persistent kernel do a cross-device reduce without returning to host". Three facts bear on our launch-floor numbers: (1) `aclshmem_barrier` / `aclshmem_sync` require a **MIX** kernel with every AIC and AIV core entering — and MIX_AIV/MIX_AIC are our two most expensive launch classes at 12.04/13.58 us, so an all-core barrier forces the expensive launch class; (2) `aclshmemx_sync_vec` is vector-only and explicitly does *not* complete remote updates, which is the cheap barrier if the data movement is already ordered by something else — AI_VECTOR_CORE launches at 0.70 us; (3) "The Cube scalar unit is outside this visibility guarantee" is a correctness trap for fusing a reduce into a cube-heavy GEMM epilogue.

---

### ACLSHMEM `shmem_device_p2p_sync.h` — signal/wait, the Lamport-style primitive on Ascend

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_p2p_sync.h
(Huawei / CANN `cann/shmem`, master as of 2026-10-20)

```c
/**
 * @brief The aclshmemx_signal_op operation updates sig_addr with signal using operation sig_op on the specified PE.
 *        This operation can be used together with aclshmem_signal_wait_until for efficient
 *        point-to-point synchronization.
 *        On Ascend950, RDMA is used as the communication engine, supporting signal_set with guaranteed atomicity.
 *
 * @param sig_addr              [in] Symmetric address of the signal word to be updated.
 * @param signal                [in] The value used to update sig_addr.
 * @param sig_op                [in] Operation used to update sig_addr with signal. Supported operations:
 *                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                    [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);
#define shmemx_signal_op aclshmemx_signal_op

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                                   ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @return Return the contents of the signal data object, sig_addr, at the calling PE that satisfies the wait condition.
 */
ACLSHMEM_DEVICE int32_t aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val);
#define shmem_signal_wait_until aclshmem_signal_wait_until
```

The generated per-type wait/test family (all `ACLSHMEM_DEVICE`, over the 13 RMA types):

```c
ACLSHMEM_DEVICE void   aclshmem_##NAME##_wait_until(__gm__ TYPE *ivar, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE void   aclshmem_##NAME##_wait(__gm__ TYPE *ivar, TYPE cmp_value)
ACLSHMEM_DEVICE void   aclshmem_##NAME##_wait_until_all(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_any(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_some(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE void   aclshmem_##NAME##_wait_until_all_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_any_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_some_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE int    aclshmem_##NAME##_test(__gm__ TYPE *ivars, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_any(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_some(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE int    aclshmem_##NAME##_test_all_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_any_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_some_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
```

## Relevance
`aclshmem_*_wait_until_all(ivars, nelems=16, ..., ACLSHMEM_CMP_EQ, flag)` is a single-call 16-way arrival wait, and `aclshmemx_signal_op(..., ACLSHMEM_SIGNAL_ADD, pe)` is a remote add. Together they replace vLLM's `[36][16]` flag matrix with one counter word per rank — the cheapest barrier shape available, and the `_test*` variants let the kernel poll without blocking so the barrier can be overlapped with the tail of the local GEMM instead of being serial.

---

### CATCCOS — CANN Templates for Compute-Communication Overlap Subroutines

https://gitcode.com/cann/catccos
(Huawei / CANN; README fetched 2026-10-20)

````text
# CATCCOS

## 📌 简介

CATCCOS(**CA**NN **T**emplates for **C**ompute-**C**ommunication **O**verlap **S**ubroutines)，中文名为昇腾计算-通信融合算子模板库，是一个聚焦于提供高性能计算通信融合类算子基础模板的代码库。

通过抽象分层的方式将计算-通信算子代码模板化。简化通算融合算子开发，解决易用性问题。内存语义实现计算通信细粒度并行，最大化掩盖度；根据计算通信特征，结合硬件架构深度优化，提供极致性能。

本代码仓为CATCCOS代码仓。结合昇腾生态力量，共同设计研发算子模板，并提供典型通算融合算子的高性能实现代码样例。

## 📁 目录结构说明

```bash
catccos
├── 3rdparty    # 依赖的catlass工程文件
├── docs        # 文档（索引见 docs/README.md，算子见 docs/operators.md）
├── examples    # kernel使用样例
├── include     # 模板头文件
├── pre-commit  # CI 代码检查与自动格式化的配置文件
├── tests       # 单元测试与动态 tiling 测试
├── tools       # AscendTimer 打点计时工具
└── utils       # 算子注册、日志、共享内存等通用辅助代码
```

## 💻 软硬件配套说明

- 硬件平台:
  - **CPU**: `aarch64`/`x86_64`
  - **NPU**: `Atlas A2 训练系列产品`/`Atlas 800I A2 推理产品`/`A200I A2 Box 异构组件`/`Atlas 350 加速卡`

- CANN版本:

| CANN包类别 | 版本要求                    |
| ---------- | --------------------------- |
| 社区版     | 8.5.0.alpha002 及之后版本 |
| 商用版     | 8.5.0及之后版本           |

## 🚀 快速上手

以`matmul_allreduce`算子样例为例，快速上手CATCCOS算子开发：

```bash
  # 用于统一配置 CANN、SHMEM、CATLASS 相关环境变量（A2 默认）
  source ./examples/utils/setup.sh

  # Ascend950 算子需指定 soc 类型（首次编译 SHMEM 时生效）
  source ./examples/utils/setup.sh -soc_type Ascend950
```

```bash
cd examples/matmul_allreduce
bash scripts/build.sh
```

```bash
# bash scripts/run.sh <device_list>
# device_list为选择的设备编号，以启动两卡执行任务为例
bash scripts/run.sh 0,1
```
````

## Relevance
CATCCOS is the Ascend analogue of Flux / TileLink: a template library whose stated goal is "内存语义实现计算通信细粒度并行，最大化掩盖度" — memory-semantics fine-grained compute/comm parallelism to maximise overlap — and it ships a `matmul_allreduce` example built on top of SHMEM. This is the supported route to fusing the reduction into the producing kernel instead of launching it serially, which is precisely the 5.35 ms we need to hide. Note it depends on `catlass`, not on HCCL.

---

### vllm-ascend Super Kernel — operator-binary fusion at ACLGraph capture

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/graph_mode.md
(vLLM Ascend plugin, `main` as of 2026-10-20)

````text
### Super Kernel optimization

[Super Kernel](https://www.hiascend.com/document/detail/en/Pytorch/latest/devguide/TorchAir/docs/en/npugraph_ex/advanced/superkernel.md) is an **optional** operator-binary fusion optimization. Unlike source-level fusion, it works on compiled kernel binaries: eligible subgraphs are identified, their child kernels are combined into a larger kernel, and synchronization is inserted according to graph dependencies. Compared with launching operators individually, this can reduce task scheduling waits, launch overhead, and operator-head overhead.

In vLLM Ascend, Super Kernel optimization is applied during ACLGraph capture. It depends on both static kernel and Npugraph_ex:

```text
Super Kernel -> static kernel -> Npugraph_ex
```

!!! note

    Static kernel is disabled by default, so Super Kernel is also disabled under the default configuration. When `enable_static_kernel` is explicitly set to `true` and `enable_super_kernel` is omitted, Super Kernel follows static kernel and is enabled. Set `enable_super_kernel` explicitly to override this inherited behavior. Super Kernel cannot be enabled when static kernel is disabled.

    Not every operator is eligible for Super Kernel fusion. An unsupported operator may split the fusion range, so the actual coverage and performance benefit depend on the model and operator sequence. Changing the Super Kernel setting also changes the static-kernel compilation configuration and may cause the static-kernel cache to be rebuilt.
````

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true}}'
```

````text
#### Verifying Super Kernel is active

When Super Kernel optimization is applied during ACLGraph capture, vLLM Ascend emits:

```text
Super kernel optimization is enabled for ACL graph capture.
```

This message confirms that the optimization step was invoked. To verify the resulting fusion coverage and performance benefit, collect an **Ascend Profiling** trace and inspect `kernel_details.csv`. Compare the kernel/task entries and latency with Super Kernel disabled; the profiler also reports the launch core count for generated Super Kernels in the `Block Dim` field.
````

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/configuration/additional_config.md

```text
| `enable_npugraph_ex`               | bool | `True` | Whether to enable npugraph_ex backend.                                                 |
| `enable_static_kernel` | bool | `False` | Whether to enable static kernel. Suitable for scenarios where shape changes are minimal and some time is available for static kernel compilation. |
| `enable_super_kernel` | bool | Inherits `enable_static_kernel` | Whether to enable Super Kernel optimization. When omitted, it follows `enable_static_kernel`; set it explicitly to override that behavior. Super Kernel requires static kernel. |
| `fuse_norm_quant`  | bool | `True` | Whether to enable fuse_norm_quant pass. |
| `fuse_qknorm_rope` | bool | `True` | Whether to enable fuse_qknorm_rope pass. If Triton is not in the environment, set it to False. |
| `fuse_muls_add` | bool | `True` | Whether to enable fuse_muls_add pass.|
```

## Relevance
"reduce task scheduling waits, launch overhead, and operator-head overhead" is the official framing of what Super Kernel buys — and at ~3008 kernels a round with a 0.70–13.58 us per-call launch floor, launch overhead alone is in the tens of milliseconds. Note the hard dependency chain `Super Kernel -> static kernel -> Npugraph_ex`, and the warning that an unsupported operator *splits the fusion range* — one ineligible op in the attention-to-MLP stretch costs the whole fusion.

---

### vllm-ascend's own GLM-5.2 A3 16-die super-kernel configuration (closest public analogue to our target)

https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_128k-static-super-kernels.yaml
(vLLM Ascend plugin; header says "Optional configuration based on PR #17871; intentionally not registered in weekly_config.yaml")

```yaml
# Optional configuration based on PR #17871; intentionally not registered in weekly_config.yaml.
test_name: "multi-node-GLM-5.2-W8A8C8-MTP-A3_128k-static-super-kernels"
model: "Eco-Tech/GLM-5.2-w8a8c8"
num_nodes: 2
npu_per_node: 16
dp_load_balancing: internal
env_common: &env_common
  LOCAL_WORLD_SIZE: "16"
  HCCL_OP_EXPANSION_MODE: "AIV"
  OMP_PROC_BIND: "false"
  VLLM_USE_MODELSCOPE: "true"
  OMP_NUM_THREADS: "1"
  HCCL_BUFFSIZE: "400"
  PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
  VLLM_ASCEND_ENABLE_FLASHCOMM1: "1"
  VLLM_ENGINE_READY_TIMEOUT_S: "3000"
  SERVER_PORT: 8077

# TODO: need to identify why TP and mtp+1 divisibility rules break on dual-node case

deployment:
  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve Eco-Tech/GLM-5.2-w8a8c8
      --host 0.0.0.0
      --port $SERVER_PORT
      --data-parallel-size 8
      --data-parallel-size-local 4
      --data-parallel-address $LOCAL_IP
      --enable-expert-parallel
      --data-parallel-rpc-port 12981
      --tensor-parallel-size 4
      --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
      --seed 1024
      --tool-call-parser glm47
      --reasoning-parser glm45
      --enable-auto-tool-choice
      --max-num-seqs 6
      --max-model-len 133120
      --max-num-batched-tokens 4096
      --trust-remote-code
      --gpu-memory-utilization 0.92
      --quantization ascend
      --enable-chunked-prefill
      --enable-prefix-caching
      --async-scheduling
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
      --kv-cache-dtype int8
      --attention_config.indexer_kv_dtype int8
      --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "ascend_compilation_config": {"fuse_muls_add": true, "enable_npugraph_ex": true, "enable_static_kernel": true, "enable_super_kernel": true}, "multistream_overlap_shared_expert": true,"enable_fused_mc2":1,"enable_mlapo":true}'
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

## Relevance
Huawei's own reference GLM-5.2 configuration on A3 uses `HCCL_OP_EXPANSION_MODE: "AIV"` (collective expansion executed on vector cores, not AICPU), `HCCL_BUFFSIZE: "400"`, `enable_fused_mc2: 1`, `multistream_overlap_shared_expert: true`, and `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`. It also pairs super kernels with MTP speculative decoding at `num_speculative_tokens: 3` and `enforce_eager: true` for the drafter. Note it chooses `--tensor-parallel-size 4` with DP8/EP rather than TP16 — the public reference does not run TP16 for GLM-5.2.

---

### vllm-ascend comm-fusion and finegrained-TP knobs

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/configuration/additional_config.md

```text
| `mc2_comm_alg`                      | str  | `""`    | set dispatch/combine op's `comm_alg` param, only supports `""/"fullmesh"/"hierarchy"/"fullmesh_v2"`. `"hierarchy"` is only supported by A2/A3, and `"fullmesh_v2"` is only supported by A3 now. |
| `enable_mc2_hierarchy_comm`         | bool | `False` | Enable dispatch/combine op inter-node communication by ROCE. This param will be deprecated and be replaced by mc2_comm_alg = "hierarchy" |
| `enable_prefill_mc2`                | bool | `False` | Whether to reserve mc2_token_capacity for prefill batches. When enabled, `max_num_batched_tokens` is used to calculate the mc2_token_capacity instead of the decode-only capacity. In this scenario, the recommended maximum value of `max_num_batched_tokens` is `tp_size * 512`. This is a temporary switch; once MC2 operators are complete for all scenarios, this switch will be removed and MC2 will be enabled by default. |
| `enable_fused_mc2`                  | int  | `0`     | Fused MC2 configuration. `0` disables the fused path, `1` selects dispatch-FFN-combine, and `2` selects CANN MegaMoe when the model and parallel configuration support it. On A5, SiTU models require a CANN MegaMoe wrapper exposing `activation` and `activation_params`; both SiTU parameters are bound from the MoE configuration during initialization. The legacy `VLLM_ASCEND_ENABLE_FUSED_MC2` environment variable is no longer supported. |
| `enable_flashcomm1`                 | bool | `False` | Whether to enable SP MoE. See [Sequence Parallelism](../feature_guide/sequence_parallelism.md). |
| `mega_moe_max_tokens`               | int  | `65536` | Reference per-rank token capacity after dispatch in the fused MC2/MegaMoe path. It is passed as `dispatch_ffn_combine`'s `max_output_size` and CANN MegaMoe buffer's `max_recv_token_num`. |
| `combine_quant_mode`                | int  | `0`     | Fused MC2 configuration. This configuration will be passed as the `comm_quant_mode` argument for the `torch_npu.npu_moe_distribute_combine_v2` operator. |
```

**finegrained_tp_config**

```text
| `oproj_tensor_parallel_size`     | int  | `0` | The custom tensor parallel size of `o_proj`.     |
| `mlp_tensor_parallel_size`       | int  | `0` | The custom tensor parallel size of `mlp`.       |
| `lmhead_tensor_parallel_size`    | int  | `0` | The custom tensor parallel size of `lm_head`.    |
| `embedding_tensor_parallel_size` | int  | `0` | The custom tensor parallel size of `embedding`. |
```

```python
{
    "finegrained_tp_config": {
        "oproj_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8,
        "lmhead_tensor_parallel_size": 8,
        "embedding_tensor_parallel_size": 8,
    },
    "enable_kv_nz": False,
    "multistream_overlap_shared_expert": True,
```

## Relevance
`finegrained_tp_config` is the structural lever that reduces the *number* of 16-way reductions: shrinking `oproj_tensor_parallel_size` / `mlp_tensor_parallel_size` below the global TP turns a 16-way allreduce into an 8-way one plus an allgather, and at batch=1 the reduce is latency- not bandwidth-bound, so halving the fan-in halves the barrier fan-in too.

---

### `cann/asc-comm` — `AscendC::Hcomm` and `AscendC::Ain`: AIV-direct-driven kernel-side one-sided comm (the Ascend NVSHMEM-device-API equivalent)

https://gitcode.com/cann/asc-comm
(Huawei / CANN, "asc-comm项目首次上线" 2026/07; Copyright (c) 2026 Huawei Technologies Co., Ltd.; README fetched 2026-10-20)

```text
# asc-comm

面向昇腾AI处理器通信场景，提供Hcomm与Ain通信API、AIV直驱实现、样例和验证用例

### 🚀 当前能力

- 提供AICore侧Hcomm点对点通信接口，覆盖普通`Init`、`ReadNbi`、`WriteNbi`、`WriteWithNotifyNbi`、`AtomicFAA`、`AtomicCAS`、`Commit`、`Drain`，以及Ascend 950 UB_CTP路径的`MakeBatchHandle`、批量读写、`BatchCommit`和批量`Drain`。
- 提供AIV直驱Hcomm RoCE和UB_CTP/URMA相关实现，主实现位于`src/aicore/hcomm/`。
- 提供AICore侧Ain单边通信接口，覆盖`Put`、`PutValue`、`Get`、`Signal`、`ReadSignal`、`WaitSignal`、`Flush`、`FlushAsync`、`Wait`，以及`AinBarrierSession`集合通信同步原语，主实现位于`src/aicore/ain/`。
- 提供Hcomm UT工程，覆盖`ascend950pr_9599_AIV`的RoCE/URMA普通接口和UB_CTP批量接口，以及`ascend910B1_AIC`基础接口用例。
- 提供SIMT URMA `WriteNbi`、`WriteValueNbi`、`ReadNbi`、`WriteWithNotifyNbi`、`AtomicFAA`、`AtomicCAS`和`Drain`，以及`simt_urma`功能样例和`simt_urma_perftest`性能样例。

## 🚀概述

asc-comm是面向昇腾AI处理器通信场景的开源仓，当前用于承载AICore侧公开API、AIV直驱设备侧实现、API文档、样例和验证能力。

当前公开能力包括`AscendC::Hcomm`点对点通信和`AscendC::Ain`单边通信，面向算子Kernel侧通信数据路径。Hcomm侧使用方通过`AscendC::Hcomm`模板选择通信协议：普通接口通过`ChannelHandle`逐条提交通信任务；Ascend 950 UB_CTP路径还可以通过BatchHandle添加多个通信任务，并通过`BatchCommit`统一提交。两种流程分别通过对应的`Drain`重载管理完成等待。Ain侧使用方通过`AscendC::Ain`模板基于对称窗口（Symmetric Window）发起`Put`/`Get`/`Signal`等单边操作，通过`Flush`或`FlushAsync`+`Wait`管理完成等待。
```

Protocol capability table (verbatim) — this is the A3-vs-950 dividing line:

```text
| 协议 | 能力说明 |
| --- | --- |
| `COMM_PROTOCOL_ROCE` | RoCE点对点通信路径，支持普通`ReadNbi`、`WriteNbi`、`Commit`、`Drain`，不支持`WriteWithNotifyNbi`和BatchHandle接口。 |
| `COMM_PROTOCOL_UB_CTP` | UBC CTP/URMA点对点通信路径，支持普通`ReadNbi`、`WriteNbi`、`WriteWithNotifyNbi`、`AtomicFAA`、`AtomicCAS`、`Commit`、`Drain`；Ascend 950还支持BatchHandle批量接口。 |
```

````text
### 如何使用Hcomm接口

Hcomm Kernel侧使用时包含如下头文件：

```cpp
#include "hcomm/hcomm.h"
```

普通接口调用流程如下：

1. 创建`AscendC::Hcomm`对象，并选择通信协议。
2. 调用`Init`初始化临时工作区。
3. 通过`ReadNbi`、`WriteNbi`、`WriteWithNotifyNbi`、`AtomicFAA`或`AtomicCAS`提交通信任务。
4. 如果提交任务时设置`commit = false`，调用`Commit`显式提交通信任务。
5. 调用`Drain`等待通道上的通信任务完成。
````

````text
### 如何使用Ain接口

Ain Kernel侧使用时包含如下头文件：

```cpp
#include "ain/ain.h"
```

基本调用流程如下：

1. 创建`AscendC::Ain`对象，绑定通信上下文索引。
2. 通过`Put`/`PutValue`/`Get`发起单边读写，或通过`Signal`发起远端原子信号操作。
3. 如果提交任务时设置`AIN_COMMIT_DELAYED`，提交将被延迟，直到后续`AIN_COMMIT_IMMED`任务触发敲门铃；否则立即提交。
4. 完成等待可通过`Flush`或`FlushAsync`+`Wait`管理。
5. 通过`ReadSignal`/`WaitSignal`读取或等待本地信号。
6. 如需集合同步，通过`AinBarrierSession`的`Sync`完成同步。

提交模式说明：

| 模式 | 行为说明 |
| --- | --- |
| `AIN_COMMIT_IMMED` | 组装通信任务后立即敲响门铃，提交给底层引擎执行。 |
| `AIN_COMMIT_DELAYED` | 仅组装通信任务，不敲门铃；延迟到后续`AIN_COMMIT_IMMED`任务触发提交。 |
````

## Relevance
`AIN_COMMIT_DELAYED` is the doorbell-batching lever: 16 peer puts assembled with no doorbell, then one `AIN_COMMIT_IMMED` rings once. On the UB_CTP path `BatchCommit` does the same at the Hcomm layer. For a 16-die one-shot push this collapses 16 submission events into one — but note the table says BatchHandle is **Ascend 950 only** and `WriteWithNotifyNbi` is **not supported on `COMM_PROTOCOL_ROCE`**, which is the A3-relevant protocol. On A3 the available push is plain `WriteNbi` + `Commit` + `Drain`.

---

### `cann/asc-comm` `include/aicore/hcomm/hcomm.h` — exact kernel-side primitive signatures

https://gitcode.com/cann/asc-comm/blob/master/include/aicore/hcomm/hcomm.h
(Copyright (c) 2026 Huawei Technologies Co., Ltd.)

```cpp
namespace AscendC {

/*!
 * @class Hcomm
 * @brief This class mainly provides a series of point-to-point communication primitive interfaces,
 *        benchmarking against Huawei's point-to-point communication C++ interface,
 *        including WriteNbi、ReadNbi and so on.
 *        The typical usage of this class is as follows:
 *          1) Create Hcomm object.
 *          2) Initialize communication channel.
 *          3) Launch comm tasks asynchronously through the corresponding interface,
 *             and the server starts assembling and launching comm tasks as soon as it listens.
 *          4) If commit is false when launching task, call the Commit interface to notify the execution of the
 *             corresponding comm task.
 *          5) Call the Drain interface (blocking) to wait for the server to complete the corresponding comm task.
 * @tparam commProtocol: The communication protocol to use. COMM_PROTOCOL_UB_CTP (implemented through URMA) is used
 *                       by default.
 */
template <CommProtocol commProtocol = COMM_PROTOCOL_UB_CTP>
class Hcomm {
public:
    /*!
     * @brief Initialize Hcomm workspace.
     * @param [in] buff: The UB buffer provided by caller.
     * @param [in] len: The buffer length in bytes.
     * @return 0 indicates success and -1 indicates failure.
     * @note URMA uses buff as its temporary workspace after 32-byte alignment.
     */
    __aicore__ inline int32_t Init(__ubuf__ uint8_t* buff, uint32_t len);
```

```cpp
    /*!
     * @brief The task launching interface of the Write point-to-point communication operator.
     *        (task content: Write data of length len from src to dst through the specified channel.)
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE3 supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The destination address of the data.
     * @param [in] src: The source address of the data.
     * @param [in] len: The length of the data to write, using byte as the basic unit. Must be less than 256 MB
     *                  (256 * 1024 * 1024 bytes).
     * @return 0 indicates success and -1 indicates failure.
     * @note Must be called after channel initialization.
     */
    template <
        bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len);
```

The remote-reduce push — note the platform restriction in the last `@note`:

```cpp
    /*!
     * @brief The task launching interface of the write-with-reduce point-to-point communication operator.
     *        The source data is reduced into the destination data on the remote side.
     * @tparam T: The element data type. int8_t, int16_t, int32_t, uint32_t, half, float and bfloat16_t are supported.
     * @tparam reduceOp: The reduction operation.
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The remote destination address of the reduction.
     * @param [in] src: The local source address of the reduction.
     * @param [in] count: The number of elements to reduce.
     * @return 0 indicates success and -1 indicates failure.
     * @note Only the UB_CTP/URMA path supports this interface.
     */
    template <
        typename T, HcommUrmaReduceOp reduceOp, bool commit = true, pipe_t commitPipe = PIPE_S,
        pipe_t reqPipe = PIPE_MTE3, auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteReduceNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t count);
```

```cpp
    /*!
     * @brief The task launching interface of the Write-with-notify point-to-point communication operator.
     * ...
     * @param [in] notifyAddr: The remote notify address.
     * @param [in] notifyVal: The remote notify value.
     */
    template <
        bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteWithNotifyNbi(
        ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len, GM_ADDR notifyAddr, uint64_t notifyVal);

    /*!
     * @brief The task launching interface of the Fetch-and-add point-to-point communication operator.
     * @tparam T: The data type of the atomic operation. Only int32_t, uint32_t, int64_t, uint64_t is supported.
     * ...
     * @param [out] fetchAddr: The address to store the old value before atomic add.
     * @param [in] addVal: The remote add value.
     */
    template <
        typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t AtomicFAA(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr, T addVal);

    /*!
     * @brief Informed that tasks submitted on channel can be executed.
     * @tparam pipe: The pipe type to use for commit, PIPE_S supported as default.
     */
    template <pipe_t pipe = PIPE_S>
    __aicore__ inline int32_t Commit(ChannelHandle channel);

    /*!
     * @brief Submit all tasks in the current batch of a batch handle.
     */
    template <typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t BatchCommit(T& batchHandle);

    /*!
     * @brief Block Aicore and drain comm tasks submitted on channel until finish processing.
     * @tparam pipe: The pipe type to use for drain, PIPE_MTE3 supported as default.
     * @param [in] channel: The handle of the communication channel.
     * @return 0 indicates success. A non-zero value indicates failure. For COMM_PROTOCOL_UB_CTP, the underlying
     *         CQ polling error code is returned directly.
     */
    template <pipe_t pipe = PIPE_MTE3>
    __aicore__ inline int32_t Drain(ChannelHandle channel);
```

Cross-AI-Core channel locking (Ascend 950 only):

```cpp
    /*!
     * @brief Acquire the cross-AI-Core lock of a channel.
     * @note This interface is supported only by COMM_PROTOCOL_UB_CTP on Ascend 950. It blocks until the lock is
     * ...
     * @brief Flush the channel state and release its cross-AI-Core lock.
     * @note This interface is supported only by COMM_PROTOCOL_UB_CTP on Ascend 950. It must be called by the AI Core
```

## Relevance
`WriteNbi` has an explicit `commit` template parameter, and the commit/req pipes are template-selectable (`PIPE_S` / `PIPE_MTE3`). That makes the submission cost of a 16-way push statically controllable from inside the kernel: issue 16 `WriteNbi<false>`, then one `Commit`, then one `Drain`. `WriteReduceNbi` would do the whole allreduce in one primitive (remote reduce-into-destination) — but it is UB_CTP/URMA only, i.e. **not available on our A3**; on A3 the reduce has to stay local after a plain push.

---

### `cann/asc-comm` `include/aicore/ain/ain.h` — `Put` / `PutValue` / `Signal` / `WaitSignal` / `AinBarrierSession`

https://gitcode.com/cann/asc-comm/blob/master/include/aicore/ain/ain.h
(Copyright (c) 2026 Huawei Technologies Co., Ltd.)

```cpp
namespace AscendC {
/*!
 * @brief Get a pointer to a remote peer's memory location within a symmetric window.
 * @param [in] peer: Peer rank id within the team.
 * @param [in] window: Symmetric window handle.
 * @param [in] offset: Byte offset into the window.
 * @return An opaque memory handle pointing to the remote peer's location.
 */
AIN_DEVICE HcommMemHandle GetPeerPointer(uint32_t peer, HcclCommSymWindow window, size_t offset);

/*!
 * @class Ain
 * @brief This class provides device-side one-sided communication primitives (put/get) layered on top
 *        of the Hcomm point-to-point engine. It resolves the per-peer communication channel from the
 *        device communication table, translates symmetric-window handles into remote/local addresses,
 *        and submits tasks immediately or defers submission until a subsequent immediate-commit task.
 *        Pending tasks can be awaited via Flush() or FlushAsync()/Wait().
 */
template <unsigned CommEngineMask = AIN_MASK_DEFAULT>
class Ain {
public:
    AIN_DEVICE Ain(uint32_t contextIndex = 0);
    AIN_DEVICE ~Ain();

    /*!
     * @brief Drain every peer channel in the team, blocking until all pending tasks complete.
     */
    AIN_DEVICE void Flush(HcommTeamHandle team);

    /*!
     * @brief Get the channel handle of a given peer without blocking, for deferred wait.
     * @note Each rank currently has only one channel; channel index 0 is used.
     */
    AIN_DEVICE void FlushAsync(HcommTeamHandle team, uint32_t peer, ChannelHandle* outChannelHandle);

    /*!
     * @brief Block until all tasks submitted on the given channel are complete.
     */
    AIN_DEVICE void Wait(ChannelHandle& channelHandle);

    /*!
     * @brief Issue a one-sided write (put) from a local window to a remote peer's window.
     * @tparam RemoteAction: Optional remote-side action tag, AinRemoteNone by default.
     * @tparam DescriptorUbuf: UB workspace descriptor type, AinDescriptorUbuf by default.
     * @tparam CommitFlags: AIN_COMMIT_IMMED rings the doorbell immediately;
     *                      AIN_COMMIT_DELAYED defers submission until a subsequent AIN_COMMIT_IMMED task.
     * @tparam Config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] remoteAction: Remote signal action fired after write; AinRemoteNone by default.
     */
    template <
        typename RemoteAction = AinRemoteNone, typename DescriptorUbuf = AinDescriptorUbuf,
        AinCommitFlags CommitFlags = AIN_COMMIT_IMMED, auto const& Config = URMA_DEFAULT_CFG>
    AIN_DEVICE void Put(
        HcommTeamHandle team, uint32_t peer, HcclCommSymWindow dstWin, uint64_t dstOffset, HcclCommSymWindow srcWin,
        uint64_t srcOffset, uint64_t bytes, RemoteAction remoteAction = RemoteAction{},
        const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Issue an inline one-sided write (put) of a typed value to a remote peer's window.
     * @tparam RemoteAction: Signal action type, for example AinSignalInc, AinSignalAdd.
     * @tparam Config: URMA WQE control config, only used by URMA.
     *         Default: strongly ordered + fence + CQE + inline enabled.
     */
    template <
        typename T, typename RemoteAction = AinRemoteNone, typename DescriptorUbuf = AinDescriptorUbuf,
        AinCommitFlags CommitFlags = AIN_COMMIT_IMMED, auto const& Config = URMA_INLINE_CFG>
    AIN_DEVICE void PutValue(
        HcommTeamHandle team, uint32_t peer, HcclCommSymWindow dstWin, uint64_t dstOffset, T value,
        RemoteAction remoteAction = RemoteAction{}, const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Issue a one-sided read (get) from a remote peer's window into a local window.
     */
    template <
        typename DescriptorUbuf = AinDescriptorUbuf, AinCommitFlags CommitFlags = AIN_COMMIT_IMMED,
        auto const& Config = URMA_DEFAULT_CFG>
    AIN_DEVICE void Get(
        HcommTeamHandle team, uint32_t peer, HcclCommSymWindow dstWin, uint64_t dstOffset, HcclCommSymWindow srcWin,
        uint64_t srcOffset, uint64_t bytes, const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Atomically update a remote peer's signal via Fetch-and-Add.
     * @param [in] action: Signal action; AinSignalInc adds 1, AinSignalAdd adds a custom value.
     */
    template <
        typename RemoteAction, typename DescriptorUbuf = AinDescriptorUbuf,
        AinCommitFlags CommitFlags = AIN_COMMIT_IMMED, auto const& Config = URMA_DEFAULT_CFG>
    AIN_DEVICE void Signal(
        HcommTeamHandle team, uint32_t peer, RemoteAction action, const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Read the local signal value, masked to the lower `bits` bits.
     * @param [in] bits: Number of low-order bits to read (1-64).
     * @param [in] order: Memory order, defaults to AIN_MEMORY_ORDER_RELAX.
     */
    AIN_DEVICE uint64_t ReadSignal(
        HcommTeamHandle team, HcclCommSymWindow signalWindow, size_t signalOffset, uint32_t bits = 64,
        AinMemoryOrder order = AIN_MEMORY_ORDER_RELAX) const;

    /*!
     * @brief Block until the local signal reaches at least `least` (masked to `bits` bits).
     */
    AIN_DEVICE void WaitSignal(
        HcommTeamHandle team, HcclCommSymWindow signalWindow, size_t signalOffset, uint64_t least, uint32_t bits = 64,
        AinMemoryOrder order = AIN_MEMORY_ORDER_RELAX) const;

private:
    static constexpr CommProtocol commProtocol = MaskToCommProtocol(CommEngineMask);
    uint32_t contextIndex_ = 0;
    Hcomm<commProtocol> hcomm_;
};
```

```cpp
template <unsigned CommEngineMask = AIN_MASK_DEFAULT>
class AinBarrierSession {
public:
    /*!
     * @brief Construct a barrier session over the given team at the given barrier resource index.
     * @param [in] index: Barrier resource index; each index isolates an independent barrier.
     */
    AIN_DEVICE AinBarrierSession(Ain<CommEngineMask>* ain, HcommTeamHandle team, uint32_t index);

    /*!
     * @brief Synchronize with all peers in the team, blocking until completion.
     * @param [in] order: Memory order, defaults to AIN_MEMORY_ORDER_RELAX.
     * @note Implements a ring barrier: Signal every other rank, then WaitSignal from each.
     */
    template <typename DescriptorUbuf = AinDescriptorUbuf>
    AIN_DEVICE void Sync(AinMemoryOrder order, const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Synchronize with all peers in the team with a timeout.
     * @return 0 on success, -1 on timeout.
     */
    template <typename DescriptorUbuf = AinDescriptorUbuf>
    AIN_DEVICE int32_t
    Sync(AinMemoryOrder order, uint64_t timeoutCycles, const DescriptorUbuf& ubuf = DescriptorUbuf{});

private:
    AinBarrierSessionImpl<CommEngineMask> barrierImpl_;
};
```

## Relevance
`AinBarrierSession::Sync` is documented as "Implements a ring barrier: Signal every other rank, then WaitSignal from each" — i.e. exactly vLLM's `barrier_at_start` shape, but as a supported CANN device API with a timeout variant and `AinMemoryOrder` selectable down to `AIN_MEMORY_ORDER_RELAX`. `Put` with `RemoteAction = AinSignalInc` fuses the data push and the arrival signal into one submitted task, which removes one of the two barrier round trips from a one-shot. `GetPeerPointer` plus `index`-isolated barrier resources means several independent reductions can be in flight with separate barriers — relevant when a round needs ~60 reductions and we want them pipelined rather than serialized.

---

### `torch_npu.npu_mm_all_reduce_base` (MC2 MatmulAllReduce) — signature, constraints, and the explicit "do not use in decode" rule

https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_mm_all_reduce_base.md
(Huawei Ascend op-plugin, tag 7.3.0)

```text
# torch_npu.npu_mm_all_reduce_base

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |

## 功能说明

- API功能：TP切分场景下，实现mm和all_reduce的融合，融合算子内部实现计算和通信流水并行。

- 计算公式：
    $$
    output = allreduce\left(x1 \mathbin{@} \left((x2 + antiquantOffset) * antiquantScale\right) + bias + x3\right)
    $$

> [!NOTE]
> 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 函数原型

```
torch_npu.npu_mm_all_reduce_base(x1, x2, hcom, *, reduce_op='sum', bias=None, antiquant_scale=None, antiquant_offset=None, x3=None, dequant_scale=None, pertoken_scale=None, comm_quant_scale_1=None, comm_quant_scale_2=None, comm_turn=0, antiquant_group_size=0) -> Tensor
```
```

```text
- **hcom** (`str`)：必选参数。通信域handle名，通过`get_hccl_comm_name`接口获取。
- **reduce_op** (`str`)：可选参数。reduce操作类型，当前版本仅支持`sum`，默认值：`sum`。
- **comm_quant_scale_1** (`Tensor`)：可选参数。alltoall通信前后的量化、去量化系数。支持`float16`、`bfloat16`，支持$ND$格式。`x2`为$[k, n]$时shape为$[1, n]$或$[n]$，用户需保证每张卡上数据保持一致且正确。
- **comm_quant_scale_2** (`Tensor`)：可选参数。allgather通信前后的量化、去量化系数。支持`float16`、`bfloat16`，支持$ND$格式。`x2`为$[k, n]$时shape为$[1, n]$或$[n]$，用户需保证每张卡上数据保持一致且正确。
- **comm_turn** (`int`)：可选参数。表示rank间通信切分粒度，默认值：`0`，表示默认的切分方式。当前版本仅支持输入`0`。
```

```text
## 约束说明

- 该接口支持推理场景下使用。
- 增量场景不使能该融合算子，全量场景使能该融合算子。
- 该接口支持图模式。
- 输入`x1`可为2维或者3维、`x2`必须是2维，分别为$(b, s, k)/(m, k)$, $(k, n)$，$k$轴满足mm算子入参要求，$k$轴相等。`bias`当前仅支持一维，且维度大小与`output`的最后一维大小相同。`x3`的shape与`output`的shape相同。
- `x1`不支持输入转置后的tensor，`x2`转置后输入，需要满足shape的第一维大小与`x1`的最后一维相同，满足matmul的计算条件。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 数据类型支持`bfloat16`。
    - `x1`、`x2`不支持为空tensor。
    - 支持1、2、4、8卡，并且仅支持hccs链路all mesh组网。
    - 非量化场景下，$m、k、n$的取值范围均为`[1, 2147483647]`。
    - `comm_quant_scale_1`，`comm_quant_scale_2`的shape应保持一致，dtype与输出的dtype保持一致，且只在全量化场景支持。
- Atlas A2 训练系列产品：一个模型中的通算融合算子（AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce），仅支持相同通信域。
- 在长序列场景，随着$b/s$或者$m$的增大，可能出现内存不足或者计算超时。
```

## Relevance
Three hard blockers for our case, stated by Huawei: (1) "增量场景不使能该融合算子，全量场景使能该融合算子" — MC2 MatmulAllReduce is explicitly *not* to be enabled in the incremental/decode scenario, which is exactly batch=1 TPOT; (2) "支持1、2、4、8卡" — 1/2/4/8 dies only, so TP16 is out of spec; (3) the three MC2 fusion ops in one model must share one communication domain. So the supported MC2 path does not reach our configuration and a custom kernel on top of `asc-comm`/`shmem` is the remaining route.

---

### HCCL / vllm-ascend environment variables that move the collective off the AI CPU

https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/optimization_and_tuning.html
(vLLM Ascend plugin documentation, "Optimization and Tuning"; also mirrored in zh-cn)

Quoted from the page's variable list (the values below are the page's own recommendations):

```text
HCCL_OP_EXPANSION_MODE  — Sets communication scheduling to AI vector cores over RoCE instead of AI CPU.  Recommended: "AIV"
HCCL_INTRA_ROCE_ENABLE  — Uses RDMA instead of SDMA links between two 8P nodes for mesh interconnect.
HCCL_RDMA_TC            — Sets the traffic class of the RDMA NIC.
HCCL_RDMA_SL            — Sets the service level of the RDMA NIC.
HCCL_BUFFSIZE           — Sets the cache size used to share data between two NPUs.
TASK_QUEUE_ENABLE       — Optimizes the operator dispatch queue. It can raise peak memory use and may hurt performance when memory is tight.  Recommended: 2
CPU_AFFINITY_CONF       — Improves performance for CPU-bottlenecked models and keeps NPU-bottlenecked models at similar performance.  Recommended: 1
PYTORCH_NPU_ALLOC_CONF  — expandable_segments:True releases communication-stream memory early so the compute stream can reuse it.
```

And the Chinese edition of the same page, on `HCCL_OP_EXPANSION_MODE`:

> 在该模式下，通信改由 AI 向量核心经 RoCE 发起调度，不再由 AI CPU 负责下发。

Huawei's own HCCL_BUFFSIZE guidance, quoted from the CANN environment-variable reference:

> 对于集合通信算子，当数据量超过 HCCL_BUFFSIZE 的取值时，性能可能会下降。建议将 HCCL_BUFFSIZE 的值设置为大于数据量。

## Relevance
`HCCL_OP_EXPANSION_MODE=AIV` moves collective *scheduling* from the AI CPU to vector cores — which is exactly why HCCL collectives on A3 can overlap with compute for free in our measurement while our own kernel does not. Our 5.35 ms SuperKernel is serial because it occupies the same cores as compute; HCCL's AIV expansion runs the collective's control flow on cores the compute kernel is not using. Any replacement must preserve that property, not just lower the kernel time.

---

## Part 3 — Barrier-free one-shot: the strongest measured results in the literature

### "Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives"

https://arxiv.org/abs/2607.16100 · https://arxiv.org/html/2607.16100v1
Siyuan Shen, Anton Korzh, John Bachan, Tiancheng Chen, Arnav Goel, Ludwig Schneider, Pouya Kousha, Zhenhao He, Sylvain Jeaugey, Kamil Iskra, Nishank Chandawala, Jeff R. Hammond, Torsten Hoefler. Submitted 17 Jul 2026.

Abstract, verbatim:

> GPU collective communication is typically optimized for bandwidth, yet many emerging workloads are increasingly limited by latency. Long-context decode-heavy large language model (LLM) inference is a prime example, where serving large models requires multiple GPUs, and many small collectives lie directly on the critical path of token generation. Therefore, even μs of overhead can impact performance and cost. In this work, we study how to approach the hardware Speed-of-Light (SoL) lower bound for GPU collectives within a scale-up network. We identify key principles for near-optimal designs, including barrier-free synchronization and efficient use of symmetric memory and multicast. Building on NCCL's device-side API, we develop low-latency interfaces for constructing custom collective kernels and use them to implement new symmetric collectives in NCCL. Microbenchmarks show substantial latency reductions for small and medium messages, reducing overhead to within 7% of the absolute SoL lower bound. When integrated into real applications, these kernels improve inter-token latency and throughput in LLM inference and accelerate cuSOLVERMp, demonstrating benefits for both AI inference and traditional HPC workloads.

The headline measurement:

> Figure 1 illustrates this effect for the long-context, small-batch decode setting that we target. On 4 GB200 GPUs, the NCCL low-latency kernel we introduce in this work reduces average latency for small messages from 11.0 μs for NCCL ring to 2.37 μs, yielding an 8.7% ITL reduction for the inference workload of Llama-3.1-70B. Using CoreWeave's on-demand price of $42/hour for 4 GB200 [8] and converting output throughput into cost per 1M output tokens, the measured data implies that each μs removed from AllReduce latency reduces cost by about 0.9%.

Push vs pull, stated as a rule:

> In a one-shot AllReduce, the entire reduction is completed in a single communication phase, where each GPU fetches data from all peers, performs the reduction locally, and writes the result to the output. In pull mode, GPUs read remote data via loads, while in push mode they write data to remote buffers before reducing locally. Push is generally faster, requiring only half a GPU-to-GPU RTT versus a full RTT for remote loads, but needs additional buffering for incoming data. Consequently, push-based one-shot designs are often preferred for latency-sensitive collectives.

> For N ranks reducing M bytes of data, one-shot incurs O(N·M) total communication volume, whereas two-shot reduces this to O(M).

The barrier-cost measurement — the key number for our 5.35 ms:

> After examining several one-shot and two-shot AllReduce implementations in state-of-the-art frameworks and libraries [6, 12, 3, 30], we observe that, regardless of push or pull communication, these designs typically rely on explicit memory barriers to synchronize peers and signal data readiness at the thread-block level. Using NCCL's ncclLsaBarrierSession, which implements a memory barrier for load-store accessible (LSA) devices, we measure the overhead of such synchronization under relaxed memory ordering and report the results in Fig. 3. Although NCCL's implementation may not be fully optimized, alternative designs follow the same fundamental pattern: a flag is propagated to all peers, and each GPU waits until it observes the corresponding signals from every other participant. Therefore, it is representative of the inherent cost in such approaches.

> The measured barrier latency shows that each barrier will incur more than 1 μs of overhead. In many AllReduce kernels, two such barriers are required. As illustrated in Fig. 1, when a small-message AllReduce completes in roughly 5 μs on four GPUs, two barrier calls alone will account for about 40% of the total latency. As emphasized earlier, when targeting near SoL performance, every μs matters. Thus, eliminating these barriers entirely can yield substantial performance gains.

The three barrier-elimination mechanisms, verbatim:

> IV-1 LL
>
> The first technique we introduce is LL, short for low latency, which originates from the LL protocol in NCCL [11] and is also used in libraries such as NVSHMEM [12] and MSCCL++ [30]. Conventional synchronization signals data arrival using explicit flags and enforced ordering between data and signals. LL removes the signaling step by packing the 8-byte flag with the 8-byte data and transmitting them atomically with 16-byte atomic stores, allowing the receiver to determine data readiness by checking the flag directly. This design halves the effective payload bandwidth and doubles scratch buffer usage, making LL mostly suitable for very small messages.
>
> IV-2 Sentinel
>
> Instead of embedding a signal in the transmitted data, the receiving scratch buffer can be initialized with a sentinel value that is unlikely to appear in valid computations, such as the floating-point value -NaN. Data is then written directly to the buffer, and the receiver polls until the value changes from the sentinel, indicating that valid data has arrived. Compared to LL, this approach preserves full effective bandwidth and uses less scratch space, making it more efficient for moderately larger messages. However, it has a few drawbacks. First, unlike LL, which updates its flag each iteration, the sentinel method requires the buffer to be reset before reuse, complicating buffer management. Second, transmitted values must never match the sentinel because a matching value would prevent the receiver from detecting data arrival. Users must therefore exclude such values to ensure correctness.
>
> IV-3 Bidirectional Communication & Double Buffering
>
> Although LL and sentinel synchronization eliminate explicit memory barriers for single exchanges, they are insufficient when messages require multiple iterations due to limited buffer space. In such cases, inputs are partitioned and processed in chunks. Conventional approaches insert barriers between iterations to prevent buffer overwrites. We can eliminate them by using bidirectional communication and double buffering.
>
> Fig. 4 demonstrates the mechanism. Consider two ranks whose scratch space cannot hold all input data simultaneously. In step ①, both ranks begin with the scratch buffer set to Buffer 0. In step ②, they exchange their first chunks C_r^0. Because Rank 0 processes data faster, it completes the reduction R and writes the result to its output while Rank 1 is still processing. In step ③, Rank 0 advances to the next iteration and switches to Buffer 1 to broadcast chunk C_0^1. It then waits for C_1^1 before proceeding, ensuring that data in Buffer 0 is not overwritten before Rank 1 finishes reading it.
>
> This bidirectional exchange, combined with double buffering, ensures that a rank cannot overwrite a peer's buffer for the next iteration until it has received data from that peer in the current one. This assumes that each rank communicates with a given peer at most once per iteration, avoiding multiple stores to the same remote address. In effect, each receive from a peer serves as an implicit permission for the next send, analogous to credit-based flow control. This mechanism allows multiple reduction iterations without costly global memory barriers.

Table I (algorithm trade-offs), reconstructed exactly as published — columns are AllReduce Algorithm / Comm. Volume per GPU / Latency, # Synchronizations / Scratch Space per Iter / Deterministic:

```text
One-shot (LL)             2(N-1)M            1    2ND
One-shot (Sentinel)       (N-1)M             1    ND
Two-shot (LL)             4(N-1)M/N          2    2D
Two-shot (Sentinel)       2(N-1)M/N          2    D
Two-shot (LL128 Atomic)   ≈2(N-1)M/N         2    ≈D/N
```

The SoL derivation and the measured constants:

> In this section, we describe how the SoL lower bound can be estimated. We define the SoL as the minimal data movement required to complete the AllReduce while ignoring all other overheads, such as instruction scheduling and computation. Since small-message latency is determined by the transfer of the smallest unit handled by the memory system, we focus on the movement of a single 128-byte cache line.

> Combining the components above, the SoL latency of an AllReduce can be expressed as
> L_SoL = 2 L_L2_RTT + L_remote_store.
> Under the SoL assumption, sending data to N peers incurs the same latency as sending to a single peer. Therefore, L_SoL represents an absolute lower bound that is independent of the number of ranks involved.

> On two GB200, we measured the L_L2_RTT and L_remote_store as 0.306 µs and 0.792 µs, respectively. Therefore, the SoL latency of an AllReduce is computed to be 1.404 µs.

Scratch-buffer sizing (measured):

> When the buffer is small (< 8 MiB), two-shot kernels are slower than one-shot kernels. Each iteration of the two-shot design requires two synchronization steps, which dominate latency. Although fewer iterations are needed overall, the additional synchronization cost outweighs this benefit. As the buffer size increases, the performance of two-shot kernels improves rapidly and then plateaus once the entire message fits within a single iteration. This indicates that, ideally, achieving the best performance for two-shot kernels requires a scratch buffer large enough to process the data in one iteration.
>
> In one-shot kernels, every rank writes to all peers simultaneously, which means that increasing the buffer size increases the amount of data sent per iteration, which raises instantaneous bandwidth pressure on the NVLink fabric. Hence, beyond a certain point, increasing the scratch size provides little benefit and can even slightly degrade performance.
>
> Based on these results, we select 4 MiB of scratch buffer for one-shot kernels, as they show limited benefit from larger buffers. For two-shot and LL128 atomic kernels, we choose 64 MiB as the default.

Results versus rank count — the part that matters for TP16:

> Our LLBuffer-based one-shot AllReduce achieves the lowest latency for small messages across all GPU counts. As shown in Fig. 11, our kernels incur only about 7% overhead over the SoL bound at 2 GPUs, while competing implementations remain noticeably farther away. Even at 64 GPUs, multicast one-shot variants stay within roughly 70% overhead of the SoL bound. While NCCL AGxLL and MSCCL++ also use LL-style synchronization, our implementation benefits from targeted optimizations including aggressive compile-time unrolling and parallel polling and reduction across ranks, which reduce detection and accumulation latency. We also observe a crossover between LL and sentinel synchronization. LL performs slightly better at very small sizes, whereas sentinel becomes preferable as message size and rank count increase, avoiding the flag overhead of LL.

> After extensively benchmarking the kernels, we modify NCCL to select the best-performing low-latency algorithm based on the collected empirical results. For example, at 4 ranks, messages below 1 MiB use one-shot kernels, while those between 1 and 2 MiB use two-shot kernels.

> Across all models, low-latency kernels consistently improve performance. The best configuration reduces ITL by 7–13% on 4 GPUs and 9–11% on 8 GPUs, with similar gains in throughput. Results hold across diverse model architectures, including dense (Llama), mixture-of-experts (DeepSeek), and hybrid attention (Qwen3-Next). NCCL LL alone performs well because decode steps involve small AllReduce operations, where one-shot LLBuffer kernels approach the SoL bound.

The API design note (why thread-level, and when barriers are unavoidable):

> The low latency APIs presented here are intentionally fine grained. Unlike NVSHMEM, which provides APIs at the thread, warp, and block levels, our design exposes only thread level primitives. This choice maximizes user control and allows implementations to achieve the lowest possible latency. Moreover, the APIs favor bidirectional communication patterns to achieve safety and optimal performance. As an example, algorithms with independent per-iteration communication, such as one-shot and two-shot AllReduce, can operate without barriers because each step uses disjoint buffers and each rank both sends and receives within the same iteration. In contrast, algorithms with inter-step dependencies and without bidirectional communication, such as ring AllReduce, still require explicit memory barriers. In a ring schedule, each rank must receive a chunk before reducing and forwarding it, as advancing early risks overwriting unconsumed data.

> Using the proposed APIs, we implement one-shot, two-shot, and two-shot LL128 atomic AllReduce, along with other collectives. Taking AllReduce as an example, algorithms can be selected via NCCL_SYM_KERNEL: AllReduce_LLBuffer for one-shot, AllReduce_LLBuffer_Twoshot for two-shot, and AllReduce_LL128_Atomic for the LL128 atomic variant. The synchronization mode is controlled by NCCL_SYM_LLBUFFER_SYNC, allowing users to switch between LL and sentinel mechanisms for the same algorithm. Additionally, two-shot AllReduce requires symmetric (LSA) output buffers, while one-shot only relies on symmetric scratch buffers initialized by NCCL and has no such requirements.

Limitations of the LL128-atomic variant:

> The algorithm has several limitations. It requires NVLink to guarantee cache-line–level atomic addition and supports only single- and half-precision types due to the availability of vectorized atomics in CUDA [40, 41]. It is also limited to addition, as the embedded flag relies on commutativity, and CUDA does not provide vectorized atomic multiplication. In practice, this is not too restrictive since addition dominates most target workloads, such as LLM inference. Finally, the algorithm is non-deterministic because floating-point atomic ordering is not guaranteed. A one-shot variant is possible, but different ranks may observe different results, which violates AllReduce semantics [16].

## Relevance
This paper is the closest published analogue of our problem: latency-bound small allreduce on the decode critical path, scale-up domain, batch=1-ish. Three things transfer directly. (1) A barrier costs >1 us and two barriers are ~40% of a 5 us allreduce — so our SuperKernel's cost is almost certainly synchronization, not data. (2) The fix is barrier *elimination*, not barrier optimization: LL (flag packed into the 16-byte atomic store), sentinel (-NaN pre-fill, poll for change), and bidirectional double buffering as credit-based flow control. The ACLSHMEM/`asc-comm` primitives above support all three — `PutValue` with inline config is the LL analogue, `WaitSignal` with `AIN_MEMORY_ORDER_RELAX` is the poll. (3) The scaling evidence is cautionary for TP16: 7% over SoL at 2 GPUs becomes ~70% over SoL at 64 GPUs even with multicast, so at 16 dies the achievable floor is several times the 2-rank floor, and a hierarchical split (see `finegrained_tp_config`) deserves measurement.

---

### PyTorch `torch.distributed._symmetric_memory` — operator maturity table, `one_shot_all_reduce`, and the Triton one-shot kernel

https://docs.pytorch.org/docs/main/distributed/symmetric_memory.html
(PyTorch documentation; "Created On: Oct 24, 2025 | Last Updated On: Oct 15, 2026")

```text
Note

torch.distributed._symmetric_memory is currently in alpha state and under
development. Its Python APIs and every operator in torch.ops.symm_mem may
change without notice. A dispatcher-visible operator is not necessarily a
user-facing API. The maturity table below identifies which operators are
intended for direct use and which are implementation details.
```

The maturity table, verbatim (Operations / Maturity / Scope or requirements / When to use them):

```text
empty, rendezvous, get, and get_mem_pool
  Alpha | General | Allocate, establish, access, and reuse symmetric memory.

fused_all_gather_matmul and fused_matmul_reduce_scatter
  Alpha | General | Overlap an unscaled matrix multiplication with a tensor-parallel collective. Ordinary accelerator tensors are accepted; the operators manage a symmetric workspace. These are also targets of the compiler's tensor-parallel fusion pass.

fused_all_gather_scaled_matmul and fused_scaled_matmul_reduce_scatter
  Experimental | Hardware-specific | FP8 or other scaled matrix multiplication when the accelerator supports aten._scaled_mm. Their interfaces closely follow compiler lowering requirements.

one_shot_all_reduce, one_shot_all_reduce_out, and two_shot_all_reduce_
  Alpha | Hardware-specific | Low-latency, direct-access all-reduce on a symmetric tensor.

multimem_all_reduce_ and multimem_all_gather_out
  Alpha | Hardware-specific | NVIDIA systems with multicast and multimem support, such as NVLink SHARP.

multimem_one_shot_all_reduce, multimem_one_shot_all_reduce_out, and multimem_one_shot_reduce_out
  Experimental | Hardware-specific | Compiler-oriented multimem variants. Their reduction order is not fixed, so all ranks are not guaranteed to receive bitwise-identical results.

get_remote_tensors
  Alpha | Hardware-specific | Create local tensor views of peer allocations when every peer is directly addressable. The views require explicit cross-rank synchronization.

nvshmem_*, all_to_all_vdev*, tile_reduce, and multi_root_tile_reduce
  Alpha | Backend-specific | NVSHMEM or rocSHMEM transport and device-driven collectives. Availability differs by operation and SHMEM implementation.

put_signal and wait_signal
  Experimental | Backend-specific | One-sided signaling; currently implemented only for the NCCL backend.

_low_contention_*, _async_input_mm, _rendezvous, and _barrier
  Internal | Compiler-facing | Compiler and dispatcher implementation details.

stream_write_value32_, memset32_, memcpy_to_multicast_, and raw nccl_* operators
  Internal | Backend implementation | Kernel composition or backend dispatch. Use a documented Python wrapper instead.
```

The motivation section, verbatim — it is the design rationale for exactly what we are building:

> With rapidly evolving parallelization techniques, existing frameworks and libraries often struggle to keep up, and developers increasingly rely on custom implementations directly scheduling communications and computations. In recent years we've witnessed a shift from primarily relying on one-dimensional data-parallelism techniques to multi-dimensional parallelism ones. The latter have different latency requirements for different types of communications and thus require fine-grained overlapping of compute and communications.
>
> To minimize compute interference, they also require the use of copy engines and network interface cards (NICs) to drive communication. [...] This increase in variety indicates the need for finer-grained communication primitives than are offered today by high-level collective APIs, ones that would enable developers to implement specific algorithms tailored for their use cases, such as low-latency collectives, fine-grained compute-communications overlap, or custom fusions.

> PyTorch Symmetric Memory unlocks three new capabilities:
>
> Customized communication patterns: Increased flexibility in kernel writing allows developers to write custom kernels that implement their custom computations and communications, directly tailored to the need of the application. [...]
>
> In-kernel compute-comm fusion: Device-initiated communication capability allows developers to write kernels with both computation and communication instructions, allowing for the fusion of computation and data movement in the smallest possible granularity.
>
> Low-latency remote access: Network transport protocols like RDMA enhance the performance of symmetric memory in networked environments by enabling direct, high-speed, and low-latency communication between processors and memory. [...]

API usage, verbatim:

```python
import torch.distributed._symmetric_memory as symm_mem

t = symm_mem.empty(128, device=torch.device("cuda", rank))
hdl = symm_mem.rendezvous(t, group)
```

> The symm_mem.empty function creates a tensor that is backed by a symmetric memory allocation. The rendezvous function establishes a rendezvous with peers in the group, and returns a handle to the symmetric memory allocation. The handle provides method to access information related to the symmetric memory allocation, such as pointers to symmetric buffer on peer ranks, multicast pointer (if supported), and signal pads.
>
> The empty and rendezvous functions must be called in the same order on all ranks in the group.

```python
# Most SymmMem ops are under the torch.ops.symm_mem namespace
torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group)
```

```text
Access to buffer pointers, multimem pointer, and signal pads is provided via:

hdl.buffer_ptrs
hdl.multicast_ptr
hdl.signal_pad_ptrs
```

> PyTorch Symmetric Memory provides CUDA Graph-compatible synchronization primitives that operate on the signal pad accompanying each symmetric memory allocation.

The reference one-shot allreduce kernel in Triton, verbatim:

```python
@triton.jit
def one_shot_all_reduce_kernel(
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequenceMemAccess=True
    )

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.bfloat16)

        for i in tl.static_range(world_size):
            buffer_rank = buf_tuple[i]
            x = tl.load(buffer_rank + offsets, mask=mask)
            acc += x

        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )
```

> Synchronizations at the beginning and the end of the kernel above guarantee that all the processes see consistent data. [...] Visit https://github.com/meta-pytorch/kraken/blob/main/kraken to see additional utilities and examples of using symmetric memory to implement common patterns in Triton.

One-sided get:

```python
src = symm_mem.empty(1024, device=device)
hdl = symm_mem.rendezvous(src, group)

if dist.get_rank(group) == 0:
    dst = torch.empty((512,), device=device)
    # Copy the last 512 elements of the peer's allocation into dst.
    symm_mem.get(dst, hdl, peer=1, offset=512)
```

Memory-pool route, which avoids rewriting allocations:

```python
    mempool = symm_mem.get_mem_pool(device)
    ...
    torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name)
```

NCCL symmetric kernels:

```text
NCCL Symmetric Kernels

Note

Requires NCCL 2.27 or later and a single NVLink domain (every rank reachable
over direct NVLink).

NCCL 2.27+ added a family of device kernels — "SymK" internally — written
specifically for symmetric, window-registered buffers. Because each rank knows
every peer's buffer address up front, these kernels skip the generic
proxy/ring machinery and instead use LL (low-latency), multimem/NVLS, and TMA
variants. NCCL picks one per call from message size, so the same
dist.all_reduce gets a latency-optimized kernel for small messages and a
bandwidth-optimized one for large ones.
```

```python
# `symm=True` registers each segment with `ncclCommWindowRegister` using
# `NCCL_WIN_COLL_SYMMETRIC`, which is what makes the symmetric kernels
# eligible. The default `symm=False` performs ordinary user-buffer
```

## Relevance
The Triton kernel is a 30-line reference one-shot: sync, load from all `world_size` peer pointers, accumulate, store, sync. That is a template we can mirror in AscendC over `aclshmem_ptr` / `GetPeerPointer`. Note PyTorch's structural split matching our problem: `one_shot_all_reduce` for the small/latency case and `fused_matmul_reduce_scatter` for the fused case, as two different operators — upstream does not try to make one kernel do both. Also note "CUDA Graph-compatible synchronization primitives", which is the property we need for ACLGraph capture of the reduce.

---

### FLUX (ByteDance) — fusing the collective into the GEMM kernel

https://arxiv.org/abs/2406.06858 · https://arxiv.org/html/2406.06858v1 · code: https://github.com/bytedance/flux (mirror https://gitee.com/ByteDance/flux)
Li-Wen Chang, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Yinmin Zhong, Xuanrun Zhang, Zuquan Song, Chengji Yao, Ziheng Jiang, Haibin Lin, Xin Jin, Xin Liu (ByteDance / Peking University).

Abstract, verbatim:

> Large deep learning models have demonstrated strong ability to solve many tasks across a wide range of applications. Those large models typically require training and inference to be distributed. Tensor parallelism is a common technique partitioning computation of an operation or layer across devices to overcome the memory capacity limitation of a single processor, and/or to accelerate computation to meet a certain latency requirement. However, this kind of parallelism introduces additional communication that might contribute a significant portion of overall runtime. Thus limits scalability of this technique within a group of devices with high speed interconnects, such as GPUs with NVLinks in a node. This paper proposes a novel method, Flux, to significantly hide communication latencies with dependent computations for GPUs. Flux over-decomposes communication and computation operations into much finer-grained operations and further fuses them into a larger kernel to effectively hide communication without compromising kernel efficiency. Flux can potentially overlap up to 96% of communication given a fused kernel. Overall, it can achieve up to 1.24x speedups for training over Megatron-LM on a cluster of 128 GPUs with various GPU generations and interconnects, and up to 1.66x and 1.30x speedups for prefill and decoding inference over vLLM on a cluster with 8 GPUs with various GPU generations and interconnects.

Mechanism, as reported:

> Each dependent computation and communication tile is mapped into a thread block, with optimizations such as kernel fusion and tile coordinate swizzling.
> ReduceScatter communication is fused into the epilogue of the GEMM kernel.
> The paper also notes that fusing the AlltoAll step into the GEMM epilogue is typically enough to overlap communication, while the reduction fusion gives only marginal gains.
> Existing overlap approaches have trouble with precise control over execution timing of operations on GPUs when using streams, and splitting GEMMs into smaller kernels often underutilizes the GPU's stream processors.
> It is built on NVIDIA CUTLASS, and reports results on A100 and H800 GPUs across PCIe and NVLink interconnects.

## Relevance
"1.30x speedups for decoding inference over vLLM" is the only published decode-phase number for GEMM-epilogue collective fusion, and it is on 8 GPUs. The important caveat in their own findings is that at the MoE AlltoAll the *communication* fusion carries the win while the *reduction* fusion is marginal — consistent with our measurement that the reduction, not the transfer, is the serial cost. Note also the stated reason streams do not work: "existing overlap approaches have trouble with precise control over execution timing of operations on GPUs when using streams" — which matches the Ascend situation where HCCL on a separate stream overlaps but our own kernel on the compute stream does not.

---

### TileLink (MLSys 2025) — generating overlapped compute-communication kernels from tile-centric primitives

https://proceedings.mlsys.org/paper_files/paper/2025/hash/c6ee784cbe46d854843e4c883a3321ef-Abstract-Conference.html · arXiv:2503.20313 · https://hgpu.org/?p=29841
ByteDance Seed; MLSys 2025.

As reported:

> Fusing communication kernels with compute kernels demands significant expertise and is error-prone.
> The system has a frontend and a backend. The frontend separates the design space for communication from that for computation and linking these two parts via tile-centric primitives. The backend lowers those primitives into low-level communication instructions and combines the parts into overlapped execution.
> Each side can be tuned on its own for tile size, tile order, and hardware resource. For example, communication can run on a copy engine while computation runs on SMs.
> Results: a 1.17× to 20.76× speedup to non-overlapping baseline and performance comparable to state-of-the-art overlapping libraries on GPUs.

## Relevance
"communication can run on a copy engine while computation runs on SMs" is the structural principle behind why HCCL is free for us and our SuperKernel is not. On Ascend the analogue of the copy engine is SDMA/xDMA (reachable through `aclshmemx_mte_put_nbi` / `aclshmemx_sdma_*`) and the analogue of "computation on SMs" is AI Core. A reduce that issues SDMA descriptors and then polls, rather than one that occupies vector cores for the whole transfer, is the shape that can overlap.

---

## Part 4 — HCCL's own open-source AIV one-shot AllReduce SuperKernel (Ascend, directly comparable)

### `HCCL_OP_EXPANSION_MODE` — the exact accepted values and what each sets

https://gitcode.com/cann/hccl/blob/master/src/common/alg_env_config.cc
(Huawei / CANN `cann/hccl`, master as of 2026-10-20; Copyright (c) 2025 Huawei Technologies Co., Ltd.)

```cpp
HcclResult ParseOpExpansion()
{
    const std::string& opExpansionModeEnv = GetEnv("HCCL_OP_EXPANSION_MODE");
    g_algEnvConfig.aicpuUnfold = false;
    g_algEnvConfig.aivMode = false;
    g_algEnvConfig.aivOnlyMode = false;
    g_algEnvConfig.ccuMSMode = false;
    g_algEnvConfig.ccuSchedMode = false;

    if (opExpansionModeEnv == "CCU_MS") {
        g_algEnvConfig.ccuMSMode = true;
        HCCL_INFO("HCCL_OP_EXPANSION_MODE set by environment to [CCU_MS]");
        return HCCL_SUCCESS;
    }

    if (opExpansionModeEnv == "CCU_SCHED") {
        g_algEnvConfig.ccuSchedMode = true;
        HCCL_INFO("HCCL_OP_EXPANSION_MODE set by environment to [CCU_SCHED]");
        return HCCL_SUCCESS;
    }

    HcclDevType deviceType;
    CHK_RET(HcclGetDeviceType(deviceType));
    // 910_93默认打开AICPU展开
    if (deviceType == HcclDevType::DEV_TYPE_910_93) {
        g_algEnvConfig.aicpuUnfold = true;
    }

    if (opExpansionModeEnv == "EmptyString") {
        HCCL_INFO(
            "HCCL_OP_EXPANSION_MODE is not set, aicpuUnfold is [%u], aivMode is [%u]", g_algEnvConfig.aicpuUnfold,
            g_algEnvConfig.aivMode);
        return HCCL_SUCCESS;
    }

    if (opExpansionModeEnv == "AI_CPU" || opExpansionModeEnv == "AICPU_TS") {
        if (deviceType == HcclDevType::DEV_TYPE_910) {
            HCCL_WARNING("910 does not support AICPU unfolding.");
        } else {
            g_algEnvConfig.aicpuUnfold = true;
        }

        // A5及后续代际默认开启aicpu task cache
        if (EqualToOrGreaterThan950(deviceType)) {
            g_algEnvConfig.aicpuCacheEnable = 1;
        }
    } else if (opExpansionModeEnv == "AIV") {
        g_algEnvConfig.aivMode = true;
    } else if (opExpansionModeEnv == "HOST") {
        g_algEnvConfig.aivMode = false;
        g_algEnvConfig.aicpuUnfold = false;
    } else if (opExpansionModeEnv == "HOST_TS") {
        if (deviceType == HcclDevType::DEV_TYPE_910B) {
            g_algEnvConfig.enableFfts = false;
        } else {
            HCCL_WARNING("deviceType[%u] does not support HOST_TS", deviceType);
        }
    } else if (opExpansionModeEnv == "AICPU_CacheDisable") {
        if (deviceType == HcclDevType::DEV_TYPE_910) {
            HCCL_WARNING("910 does not support AICPU unfolding.");
        } else {
            g_algEnvConfig.aicpuUnfold = true;
            g_algEnvConfig.aicpuCacheEnable = 0; // Disable aicpu cache
        }
    } else {
        HCCL_ERROR(
            "HCCL_OP_EXPANSION_MODE is set to [%s], which is incorrect. Please check", opExpansionModeEnv.c_str());
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "environmental variable HCCL_OP_EXPANSION_MODE is [%s], aicpuUnfold[%u], aicpuCacheEnable[%u], "
        "aivMode[%u], enableFfts[%u]",
        opExpansionModeEnv.c_str(), g_algEnvConfig.aicpuUnfold, g_algEnvConfig.aicpuCacheEnable, g_algEnvConfig.aivMode,
        g_algEnvConfig.enableFfts);
    return HCCL_SUCCESS;
}
```

And the config struct it writes into, from https://gitcode.com/cann/hccl/blob/master/src/common/alg_env_config.h :

```cpp
    u32 intraRoceSwitch;             // server内的通信方式 与intraPcieSwitch组合使用，默认为0
    int32_t inconsistentCheckSwitch; // 参数一致性校验开关，默认为0
    u8 hcclDeterministic;
    bool aicpuUnfold;
    uint8_t aicpuCacheEnable;
    bool aivMode;
    bool aivOnlyMode;
    bool ccuMSMode;
    bool ccuSchedMode;
    bool enableFfts;
    bool execTimeOutSet;
    double execTimeout;
    bool multipleDimensionSplitRatioSet;
    double multipleDimensionSplitRatio;
    bool hcclRetryConfig[HCCL_RETRY_ENABLE_LEVEL_NUM];
    bool taskExceptionEnable;
    std::map<HcclCMDType, std::vector<HcclAlgoType>> hcclAlgoConfig;
```

```cpp
        hcclDeterministic = static_cast<u8>(
            DeterministicEnableLevel::
                DETERMINISTIC_DISABLE); // 确定性配置 0：不支持；1：支持确定性不支持规约保序；2：支持确定性&规约保序
```

## Relevance
`DEV_TYPE_910_93` (the A3 class) defaults to AICPU unfolding. Setting `AIV` turns `aivMode` on and leaves `aicpuUnfold` false in that branch — i.e. the two are alternatives, not additive, and `AIV_ONLY` is a separate `aivOnlyMode` flag. Note the undocumented accepted values the error message does not list: `AICPU_TS`, `HOST_TS`, `AICPU_CacheDisable`, `CCU_MS`, `CCU_SCHED`.

---

### HCCL A3 operator support matrix — AIV mode is small-data/inference-only, and excludes zero-copy and retry by design

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/hcclug/hcclug_000134.html
(static mirror used for extraction: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/hcclug/hcclug_000134.html — CANN Community Edition 9.0.0, page title "Atlas A3 训练系列产品/Atlas A3 推理系列产品")

Definitions, verbatim:

```text
本节提供Atlas A3 训练系列产品/Atlas A3 推理系列产品的通信算子支持情况。
单算子零拷贝：为了降低内存拷贝开销，使得HCCL可以直接对业务传入的内存进行操作，提升通信性能。
通信算子重执行：网络故障导致通信闪断时，HCCL会尝试重新执行此通信算子，提升通信稳定性。
确定性计算：归约类通信算子在相同的硬件和输入下，多次执行将产生相同的输出。
下面按照通信算子的展开模式进行通信算子支持情况的呈现，未列出的展开模式代表不支持。
本节表格中“√”代表支持，“☓”代表不支持，“NA”代表不涉及。
未列出的算子与网络运行模式代表不支持。
```

The AIV table, verbatim (columns: 算子 | 网络运行模式 | 单算子零拷贝 | 确定性计算 | 重执行 | 节点内通信 | 超节点内通信 | 超节点间通信):

```text
AIV
|算子 | 网络运行模式 | 单算子零拷贝 | 确定性计算 | 重执行 | 节点内通信 | 超节点内通信 | 超节点间通信|
|Broadcast | 单算子模式 | ☓ | NA | ☓ | √ | ☓ | ☓|
|图模式Ascend IR | ☓ | NA | ☓ | ☓ | ☓ | ☓|
|图捕获模式aclgraph | ☓ | NA | ☓ | ☓ | ☓ | ☓|
|AllGather | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
|图模式Ascend IR | ☓ | NA | ☓ | √ | √ | ☓|
|图捕获模式aclgraph | ☓ | NA | ☓ | √ | √ | ☓|
|AllReduce | 单算子模式 | ☓ | √ | ☓ | √ | √ | ☓|
|图模式Ascend IR | ☓ | √ | ☓ | √ | √ | ☓|
|图捕获模式aclgraph | ☓ | √ | ☓ | √ | √ | ☓|
|ReduceScatter | 单算子模式 | ☓ | √ | ☓ | √ | √ | ☓|
|图模式Ascend IR | ☓ | √ | ☓ | √ | √ | ☓|
|图捕获模式aclgraph | ☓ | √ | ☓ | √ | √ | ☓|
|AlltoAll | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
|AlltoAllV | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
|AlltoAllVC | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
|ReduceScatterV | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
```

The footnote, which is the key statement:

```text
AIV模式在小数据量通信时性能较优，主要用于推理场景，此模式下：
单算子零拷贝会引入执行时内存协商，导致通信时延变大，所以当前AIV模式下不支持单算子零拷贝。
重执行特性会增加执行耗时，所以AIV模式不支持算子重执行。
仅支持超节点内通信，不支持超节点间通信。
```

By contrast, the AI CPU table on the same page shows `AllReduce / 单算子模式 / 单算子零拷贝 = √ / 确定性计算 = √ / 重执行 = √`, and all four scopes (节点内/超节点内/超节点间) supported:

```text
AI CPU
|AllReduce | 单算子模式 | √ | √ | √ | √ | √ | √|
|图模式Ascend IR | NA | √ | √ | √ | √ | √|
|图捕获模式aclgraph | NA | √ | √ | √ | √ | √|
```

## Relevance
Huawei's own position: AIV expansion is the small-data, inference-oriented mode, supported for AllReduce in `图捕获模式aclgraph` with deterministic computation, intra-supernode only. Our TP16 on one A3 machine is 超节点内, so the mode is in scope. The two things AIV deliberately gives up — 单算子零拷贝 and 重执行 — are both acceptable for us, and the stated reason for dropping zero-copy is instructive: "单算子零拷贝会引入执行时内存协商，导致通信时延变大" — negotiation at execution time raises latency, so a *pre-registered* symmetric buffer beats a zero-copy handshake at batch=1.

---

### HCCL AllReduce algorithm auto-selection — the exact AIV one-shot/two-shot crossover constants

https://gitcode.com/cann/hccl/blob/master/src/ops/all_reduce/selector/all_reduce_auto_selector.cc
(Huawei / CANN `cann/hccl`, master as of 2026-10-20)

```cpp
namespace ops_hccl {
constexpr u64 RS_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 AR_ONESHOT_1D_MAX_DATA_SIZE = 16 * 1024;
constexpr u64 AR_M2M_1D_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_MAX_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_CROSS_SMALL_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_64DATATYPE_DATA_SIZE = 8 * 1024 * 1024;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u32 MAX_RANK_NUM_FOR_REDUCE_MS_ALGO = 8;
constexpr u64 AR_FLATTEN_MAX_DATA_SIZE = 512 * 1024;
constexpr u64 AR_CCU_CLOS_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_SEQUENCE_DATA_SIZE = 4ULL * 1024 * 1024 * 1024;
constexpr u64 OMNI_PCIE_AR_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_SCHED_DATA_SIZE = 64 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_MS_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_LARGE_SIZE = 128 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_16P_LARGE_SIZE = 256 * 1024 * 1024;
constexpr u32 OMNI_UBX_AR_RANK_SIZE = 16;
constexpr u64 AR_AIV_SMALL_DATA_SIZE_IN_BOARD = 128 * 1024;
constexpr u64 AR_AIV_BOARD_SIZE = 8;
constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;
constexpr u64 AR_2P_DETOUR_DATA_SIZE = 8 * 1024 * 1024;
constexpr u32 AR_MORE_64P_SEQ_MAX_DATA_SIZE = 32 * 1024 * 1024;
constexpr u32 AR_CCU_MAX_RANK_SIZE = 64;
constexpr u64 AR_CCU_SEQ_MAX_DATA_SIZE = 64 * 1024 * 1024;
```

The AIV selection, verbatim:

```cpp
    if (topoInfo->level0Topo != Level0Shape::MESH_1D) {
        selectAlgName = "AivAllReduceSoleMeshTwoShot";
    } else if (topoInfo->userRankSize <= AR_AIV_BOARD_SIZE) {
        // 板内8p场景，按照时延拐点选择算法
        if (dataSize < AR_AIV_SMALL_DATA_SIZE_IN_BOARD) {
            selectAlgName = "AivAllReduceSoleMeshOneShot";
        } else {
            selectAlgName = "AivAllReduceSoleMeshTwoShot";
        }
    } else {
        if (IsSmallData(dataSize)) {
            selectAlgName = "AivAllReduceSoleMeshOneShot";
        } else {
            selectAlgName = "AivAllReduceSoleMeshTwoShot";
        }
    }
```

The guards that reject AIV, verbatim:

```cpp
    // 保序模式不支持AIV，需要回退到AICPU
    CHK_PRT_RET(
        IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize),
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG,
            "[AllReduceAutoSelector][Algo][AllReduceAutoSelector] DETERMINISTIC_STRICT mode is not supported yet for "
            "AIV mode."),
        SelectorStatus::NOT_MATCH);

    // aiv 模式不支持 PROD
    CHK_PRT_RET(
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
        || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[Algo][AllReduceAutoSelector] aiv mode does not support UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->userRankSize > MAX_RANK_SIZE) { ... return SelectorStatus::NOT_MATCH; }

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_PRT_RET(
        HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS, ...);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY
        && dataSize >= AIV_MAX_PER_RANK_DATA_SIZE * topoInfo->userRankSize) {
        ... return SelectorStatus::NOT_MATCH;
    }
    if (dataSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) {
        ... return SelectorStatus::NOT_MATCH;
    }
```

## Relevance
Huawei tuned the AIV one-shot/two-shot crossover for a 1D mesh at **128 KB for ≤8 ranks** ("按照时延拐点选择算法" — chosen at the latency knee), versus vLLM's 256 KB at ≤8 GPUs. Above 8 ranks it falls back to a generic `IsSmallData`. `OMNI_UBX_AR_RANK_SIZE = 16` and `OMNI_UBX_AR_16P_LARGE_SIZE = 256 MB` show 16-rank paths exist but are sized for large messages. Also note `HcclGetHcclBuffer` is a public-ish accessor for the registered CCL buffer — the same symmetric scratch our own kernel would need, and `HCCL_BUFFSIZE` sizes it.

---

### HCCL AIV one-shot AllReduce kernel — complete source, including the `SuperKernel` entry point

https://gitcode.com/cann/hccl/blob/master/src/ops/all_reduce/algorithm/template/aiv/kernel/aiv_all_reduce_mesh_1d_oneshot.h
(Huawei / CANN `cann/hccl`, master as of 2026-10-20; Copyright (c) 2025 Huawei Technologies Co., Ltd.)

```cpp
#include "aiv_communication_base_v2.h"

using namespace AscendC;

template <typename T>

class AivAllReduceMesh1DOneShot : public AivCommBase {
    constexpr static uint64_t CORE_NUMS_PER_STAGE = 16; // 每个阶段提供的最大核数
    constexpr static uint64_t STAGE_NUM = 2;            // 生产者 消费者
    constexpr static uint64_t TAG_FLAG_SIZE = 8;
    constexpr static uint64_t coreNumPerRank = 1;

public:
    __aicore__ inline AivAllReduceMesh1DOneShot() {}

    __aicore__ inline void Producer()
    {
        CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)input_, len_);
        pipe_barrier(PIPE_ALL);
        Record(targetRank, rank_, curTag_);
    }

    __aicore__ inline void Consumer()
    {
        // 初值：rank0 数据（myGmIn_ 槽 0）就位后覆盖到 output_
        WaitFlag(rank_, 0, curTag_);
        CpGM2GM((__gm__ T*)output_, (__gm__ T*)myGmIn_, len_);
        pipe_barrier(PIPE_ALL);
        // 树形折叠（与 Aicpu 保序 RunLocalReduce 同序）：slot(dst) += slot(src)，dst==0 累加进 output_，dst>=1 累加进
        // myGmIn_ 槽 零存储 flag 等待：源槽仅当 src >= k 时等待（[1,k) 已在作为目的槽被折入时等待过）；
        // 目的槽仅首次被折入时等待（新目的槽只出现在第 0/1 轮）
        uint32_t curBlocks = static_cast<uint32_t>(rankSize_);
        uint32_t m0 = GetLargestPowerOf2LessThan(curBlocks);
        uint32_t firstRoundDstNum = curBlocks - m0;
        uint32_t k = (firstRoundDstNum > (m0 >> 1)) ? firstRoundDstNum : (m0 >> 1);
        uint32_t round = 0;
        while (curBlocks > 1) {
            uint32_t M = GetLargestPowerOf2LessThan(curBlocks);
            if (M == 0) {
                break;
            }
            for (uint32_t src = M; src < curBlocks; ++src) {
                uint32_t dst = src % M;
                if (src >= k) {
                    WaitFlag(rank_, src, curTag_);
                }
                if (round == 0 || (round == 1 && dst >= firstRoundDstNum)) {
                    WaitFlag(rank_, dst, curTag_);
                }
                uint64_t srcOff = reinterpret_cast<uint64_t>(myGmIn_) + src * len_ * sizeof(T);
                if (dst == 0) {
                    CpGM2GM((__gm__ T*)output_, (__gm__ T*)srcOff, len_, reduceOp_);
                } else {
                    CpGM2GM(
                        (__gm__ T*)(reinterpret_cast<uint64_t>(myGmIn_) + dst * len_ * sizeof(T)), (__gm__ T*)srcOff,
                        len_, reduceOp_);
                }
                pipe_barrier(PIPE_ALL);
            }
            curBlocks = M;
            ++round;
        }
    }

    __aicore__ inline void FlagClear()
    {
        uint64_t flag_offset = 0; //(blockIdx_ % coreNumPerRank) * TAG_FLAG_SIZE;
        Record(rank_, flag_offset, 0);
    }

    // aiv core数目大于ranksize+1
    __aicore__ inline void ProcessCoreLargeCase(uint64_t curCount, uint32_t sliceId, uint64_t stride)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        this->curCount = curCount / coreNumPerRank;
        coreNumPerStage = coreNumPerRank * rankSize_;

        if (blockIdx_ < coreNumPerStage) {
            targetRank = blockIdx_;
            uint32_t preSyncOffset = rankSize_ * 2;
            Record(targetRank, preSyncOffset + rank_, curTag_);
            WaitFlag(rank_, preSyncOffset + targetRank, curTag_);
            uint64_t outerOffset = rank_ * this->curCount * sizeof(T);
            outputOffset = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + outerOffset;
            Producer();
            SyncAll<true>();
        } else if (blockIdx_ < coreNumPerStage + coreNumPerRank) {
            SyncAll<true>();
            Consumer();
        } else {
            SyncAll<true>();
        }
    }

    // aiv core数目小于ranksize
    __aicore__ inline void ProcessCoreSmallCase(uint64_t curCount, uint32_t sliceId, uint64_t stride)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        this->curCount = curCount;

        // SK空闲核（blockIdx >= numBlocks_）跳过生产者搬运，避免与忙核重复写数据和flag
        if (!IsIdleCore()) {
            uint32_t preSyncOffset = rankSize_ * 2;
            for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
                targetRank = blockIdx_ + i * numBlocks_;
                Record(targetRank, preSyncOffset + rank_, curTag_);
            }
            pipe_barrier(PIPE_ALL);
            for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
                targetRank = blockIdx_ + i * numBlocks_;
                WaitFlag(rank_, preSyncOffset + targetRank, curTag_);
            }
            for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
                targetRank = blockIdx_ + i * numBlocks_;
                uint64_t outerOffset = rank_ * this->curCount * sizeof(T);
                outputOffset = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + outerOffset;
                Producer();
            }
        }
        SyncAll<true>();
        if (blockIdx_ == numBlocks_ - 1) {
            Consumer();
        }
    }

    uint32_t coreNumPerStage;
    uint32_t targetRank;
    GM_ADDR peerMemThisCore;
    uint64_t outputOffset;
    uint64_t curCount;
    uint64_t dataBufferSize;
};

template <typename T>
__aicore__ inline void AivAllReduceV2Mesh1DOneShot(KERNEL_ARGS_DEF)
{
    AivAllReduceMesh1DOneShot<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    if (rankSize + 1 <= op.numBlocks_) {
        op.ProcessCoreLargeCase(len, sliceId, inputSliceStride);
    } else {
        op.ProcessCoreSmallCase(len, sliceId, inputSliceStride);
    }
    op.BarrierAll();
}

template <typename T>
__aicore__ inline void AivAllReduceV2Mesh1DOneShotSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    AivAllReduceMesh1DOneShot<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / op.rankSize_ / sizeof(T);
    uint64_t countLeft = op.len_;

    int32_t loopTag = op.tag_;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        if (op.rankSize_ + 1 <= op.numBlocks_) {
            op.ProcessCoreLargeCase(curCount, loopTag, 1);
        } else {
            op.ProcessCoreSmallCase(curCount, loopTag, 1);
        }
        op.BarrierAll();

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

__aicore__ inline void sk_ar_mesh_1d_oneshot(SUPERKERNEL_ARGS_DEF)
{
#ifdef HCCL_DTYPE_INT8
    AivAllReduceV2Mesh1DOneShotSuperKernel<int8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT16
    AivAllReduceV2Mesh1DOneShotSuperKernel<int16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT32
    AivAllReduceV2Mesh1DOneShotSuperKernel<int32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP16
    AivAllReduceV2Mesh1DOneShotSuperKernel<half>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP32
    AivAllReduceV2Mesh1DOneShotSuperKernel<float>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_BFP16
    AivAllReduceV2Mesh1DOneShotSuperKernel<bfloat16_t>(SUPERKERNEL_ARGS_CALL);
#else
#endif
}
```

## Relevance
This is HCCL's own one-shot push allreduce, in AscendC, with a `SuperKernel` entry — the direct reference implementation for what we built. Structural points worth comparing against ours: (a) the shape is push-then-tree-fold, *not* push-then-flat-sum — the consumer does `GetLargestPowerOf2LessThan` rounds of `slot(dst) += slot(src)` with `CpGM2GM(..., reduceOp_)`, matched in order to the AICPU order-preserving reduce for bit-identical results; (b) exactly one core does the Consumer (`blockIdx_ == numBlocks_ - 1`) in the small-core case, which serializes the reduction on one AIV; (c) the pre-sync uses a *per-target-rank* flag slot at `preSyncOffset = rankSize_ * 2` so the arrival wait is one flag per peer, no 36×16 matrix; (d) `pipe_barrier(PIPE_ALL)` appears after every copy — a full pipe barrier per peer is a real cost at 16 peers.

---

### HCCL AIV constants and the CCL buffer layout — `aiv_defines.h`

https://gitcode.com/cann/hccl/blob/master/src/ops/op_common/algorithm/template/aiv/aiv_defines.h
(Huawei / CANN `cann/hccl`, master as of 2026-10-20)

```cpp
// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
constexpr uint32_t MAX_RANK_SIZE = 1024; // server内最大卡数
constexpr uint32_t MAX_RANK_SIZE_V = 256;
constexpr uint32_t BR_CTRL_CORE_LIMIT_RANK_SIZE = 64;
constexpr uint64_t BUFFER_OUT_ADDR_OFFSET = 16 * 1024;
constexpr uint64_t LOCAL_FLAG_BUF_LEN = 2560;
constexpr uint64_t AIV_TAG_MOVE_RIGHT_BITS = 16;
constexpr uint64_t LOW_16_BITS = 0xFFFF;
constexpr uint64_t DATA_LIMIT = 512 * 1024;
constexpr uint32_t PING_PONG = 2;

constexpr uint64_t AIV_PING_PONG_FACTOR_TWO = 2;
constexpr uint32_t NUM_BLOCKS_FOUR_PER_RANK_A3 = 4;
constexpr uint32_t MAX_NUM_BLOCKS = 48;

constexpr uint64_t FLAG_SIZE = 128;
constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE_4 = UB_FLAG_SIZE * 4;
constexpr uint64_t UB_FLAG_SIZE_8 = UB_FLAG_SIZE * 8;
constexpr uint64_t UB_MAX_DATA_SIZE = 190 * 1024;
constexpr uint64_t UB_DB_DATA_BATCH_SIZE = UB_MAX_DATA_SIZE / 2;
constexpr uint32_t MaxBufferSize = 200 * 1024 * 1024;

constexpr uint64_t ATOMIC_FLAG_SIZE = 512;
constexpr uint64_t FLAG_ONE_OFFSET = 0;
constexpr uint64_t FLAG_TWO_OFFSET = FLAG_SIZE;
constexpr uint64_t FLAG_THREE_OFFSET = FLAG_SIZE * 2;
constexpr uint64_t FLAG_FOUR_OFFSET = FLAG_SIZE * 3;
constexpr uint64_t FLAG_FIVE_OFFSET = FLAG_SIZE * 4;

constexpr uint64_t DOUBLE = 2;
constexpr uint64_t FLAG_BUF_NUM = 3;
constexpr uint64_t TILING_NUM = 4;
constexpr uint64_t CHUNK_SIZE = 2048;

constexpr int32_t TAG_INIT_VALUE = 1;
constexpr int32_t TAG_RESET_COUNT = 1000;
constexpr uint32_t AIV_FLAG_CLEAR_OFFSET = 512 * 1024;
// 相对于GM_OUT，前同步、尾同步使用的同步标记区的偏移，也是普通标记区的大小
constexpr uint32_t FLAG1_OFFSET = 1 * 1024 * 1024;
constexpr uint32_t FLAG2_OFFSET = 5 * 1024 * 1024;
constexpr uint32_t BASE_FLAG_OFFSET = 9 * 1024 * 1024;
constexpr uint32_t AIV_FLAG_EMPTY_OFFSET = 10 * 1024 * 1024;
constexpr uint32_t GM_OUT_PING_OFFSET = 18 * 1024 * 1024;
constexpr uint32_t GM_OUT_PONG_OFFSET = 34 * 1024 * 1024;
constexpr uint64_t PINGPONG_TOTAL_DATA_LIMIT = (GM_OUT_PONG_OFFSET - GM_OUT_PING_OFFSET) / PING_PONG;

#if defined(HCCL_AIV_ACLGRAPH_SK)
// SK_BIND cap 掩码：blockdim_scale_up，表示算子已自行支持核数向上扩展
constexpr uint64_t SK_CAP_BLOCKDIM_SCALE_UP = 16;
#endif

/**
 * ccl buffers        GM_OUT               Tag(大小4)             flag1         flag2             BarrierBase Clear
 * data1              data2 0       |         16K          |         512K          |      1M      |       5M      | 9M
 * |               10M           |   18M             |    34M BUFFER_OUT_ADDR_OFFSET | AIV_FLAG_CLEAR_OFFSET |
 * FLAG1_OFFSET|  FLAG2_OFFSET |  BASE_FLAG_OFFSET |  AIV_FLAG_EMPTY_OFFSET |GM_OUT_PING_OFFSET | GM_OUT_PONG_OFFSET
 */
```

The SuperKernel argument struct and macros:

```cpp
using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    GM_ADDR buffersIn = nullptr; // 注册的CCLIN地址，所有卡可访问
    uint64_t rank;
    uint64_t rankSize;
    uint64_t len;
    uint64_t dataType;
    uint64_t unitSize;
    uint64_t reduceOp;
    uint64_t numBlocks;
    uint64_t tag; // 第几次调用，定时重置成1
    uint64_t clearEnable;
    uint64_t inputSliceStride;
    uint64_t outputSliceStride;
    uint64_t repeatNum;
    uint64_t inputRepeatStride;
    uint64_t outputRepeatStride;
    uint64_t input;
    uint64_t output;
    uint64_t cclBufferSize;
};

enum class AivNotifyType { ACK, DataSignal, Done };

#define SUPERKERNEL_LITE_ARGS_DEF uint64_t args_offset

#define SUPERKERNEL_LITE_ARGS_EXTRACT                \
    GM_ADDR* param_base = (GM_ADDR*)get_para_base(); \
    GM_ADDR hiddenInput = param_base[args_offset++]; \
    GM_ADDR input = param_base[args_offset++];       \
    GM_ADDR output = param_base[args_offset++]

#define SUPERKERNEL_ARGS_DEF GM_ADDR hiddenInput, GM_ADDR input, GM_ADDR output

#define SUPERKERNEL_ARGS_CALL hiddenInput, input, output

#define SUPERKERNEL_CLASS_INIT hiddenInput, input, output

#define EXPORT_AIV_META_INFO(kernel_name)                               \
    static const struct FunLevelKType kernel_name##_kernel_type_section \
        __attribute__((used, section(".ascend.meta." #kernel_name)))    \
        = {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}
```

## Relevance
`HCCL_AIV_ACLGRAPH_SK` and `SK_CAP_BLOCKDIM_SCALE_UP = 16` confirm HCCL has a supported SuperKernel path intended for ACLGraph capture, with a capability bit meaning "the operator handles core-count scale-up itself". `SUPERKERNEL_ARGS_DEF` is only three pointers (`hiddenInput, input, output`) with everything else read from `get_para_base()` — so a captured SuperKernel's per-call mutable state is three addresses, which is the mechanism that makes graph capture of a collective workable. `MAX_NUM_BLOCKS = 48` and `NUM_BLOCKS_FOUR_PER_RANK_A3 = 4` give the core-budget shape on A3; `UB_MAX_DATA_SIZE = 190 * 1024` with double buffering gives 95 KB per batch, and `FLAG_SIZE = 128` is a whole cache line per flag.

---

### HCCL AIV `Record` / `WaitFlag` / `BarrierAll` — the actual Ascend barrier implementation

https://gitcode.com/cann/hccl/blob/master/src/ops/op_common/algorithm/template/aiv/aiv_communication_base_v2.h
(Huawei / CANN `cann/hccl`, master as of 2026-10-20)

```cpp
__aicore__ inline void AivCommBase::Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    d2hGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(GetGmOut(targetRank) + flag_offset * FLAG_SIZE));
    localTagTensor.SetValue(0, curTag);
    pipe_barrier(PIPE_ALL);
    DataCopyUB2GM(d2hGlobal, localTagTensor, FLAG_SIZE / sizeof(int32_t));
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void AivCommBase::WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    d2hGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(GetGmOut(targetRank) + flag_offset * FLAG_SIZE));
    while (true) {
        DataCopyGM2UB(localTagTensor, d2hGlobal, FLAG_SIZE / sizeof(int32_t));
        pipe_barrier(PIPE_ALL);
        if (localTagTensor.GetValue(0) == curTag) {
            break;
        }
    }
}
```

```cpp
__aicore__ inline void AivCommBase::BarrierAll()
{
    SyncAll<true>();
    // SK空闲核必须参与本SyncAll（补齐满核同步），之后的flag收发不参与
    if (IsIdleCore()) {
        return;
    }

    // 每个核分配多个rank
    uint32_t perCoreRankNum = rankSize_ / numBlocks_;
    uint32_t remainRankNum = rankSize_ % numBlocks_;
    uint32_t curCoreRankNum = blockIdx_ < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
    uint32_t startRank
        = blockIdx_ < remainRankNum ? (perCoreRankNum + 1) * blockIdx_ : perCoreRankNum * blockIdx_ + remainRankNum;
    uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset_ + rank_ * FLAG_SIZE;
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        Record(rank, flag_offset / FLAG_SIZE, 1);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset_ + rank * FLAG_SIZE;
        WaitFlag(rank_, flag_offset / FLAG_SIZE, 1);
        Record(rank_, flag_offset / FLAG_SIZE, 0);
    }
}
```

```cpp
__aicore__ inline void AivCommBase::Barrier(uint32_t step)
{
    // 用10个flag
    uint32_t flagOffset = AIV_FLAG_EMPTY_OFFSET - gmOutOffset_ - (step % 2 + 1) * FLAG_SIZE * rankSize_;
    __gm__ int32_t* ctrlFlagsGM;
    if (blockIdx_ == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_;
            ctrlFlagsGM = (__gm__ int32_t*)(GetGmOut(targetRank) + flagOffset + rank_ * FLAG_SIZE);
            SetSignalValue(ctrlFlagsGM, localSetTensor, 1);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_;
            ctrlFlagsGM = (__gm__ int32_t*)(myGmOut_ + flagOffset + targetRank * FLAG_SIZE);
            WaitSignalValue(ctrlFlagsGM, localCheckTensor, 1);
        }
```

```cpp
    // SuperKernel满核启动时，blockIdx超出算子自身核数(numBlocks_)的核为空闲核
    __aicore__ inline bool IsIdleCore() const { return blockIdx_ >= static_cast<uint32_t>(numBlocks_); }

    __aicore__ inline void BarrierForFirstOP()
    {
        BarrierForFirstOPInner(0);
        SyncAll<true>();
        ClearGM();
        SyncAll<true>();
        BarrierForFirstOPInner(1);
        SyncAll<true>();
    }
```

```cpp
__aicore__ inline void AivCommBase::ClearGM()
{
    // 无论pingpong，清零区域始终从FLAG1_OFFSET开始
    // myGmOut_ = base + gmOutOffset_，需减去gmOutOffset_再加FLAG1_OFFSET
    // SK空闲核不参与清零（忙核的均分已覆盖整个清零区域）
    if (IsIdleCore()) {
        return;
    }
    GM_ADDR flagBase = myGmOut_ - gmOutOffset_ + FLAG1_OFFSET;
    uint32_t emptyOffset = AIV_FLAG_EMPTY_OFFSET - FLAG1_OFFSET;
    uint32_t blockCount = (BASE_FLAG_OFFSET - FLAG1_OFFSET) / numBlocks_;
    uint32_t blockOffset = blockCount * blockIdx_;
    CpGM2GM(flagBase + blockOffset, flagBase + blockOffset + emptyOffset, blockCount);
}
```

## Relevance
`Record` and `WaitFlag` are each wrapped in `pipe_barrier(PIPE_ALL)`, and `WaitFlag` re-issues a full `DataCopyGM2UB` of a 128-byte `FLAG_SIZE` plus a `pipe_barrier(PIPE_ALL)` on *every* poll iteration. That polling loop is the Ascend analogue of the >1 us barrier the SoL paper measures, and it is strictly more expensive than a scalar `ld.volatile` spin because each iteration drains all pipes. `BarrierAll` splits ranks across cores (`perCoreRankNum = rankSize_ / numBlocks_`), so at TP16 with few blocks each core does several Record/WaitFlag pairs serially. These three facts together are the concrete, Ascend-specific place to attack a 5.35 ms serial reduce: replace the GM2UB-poll-plus-full-pipe-barrier loop with a cheaper wait, and stop paying `pipe_barrier(PIPE_ALL)` per peer.

---

## Part 5 — Measured small-message numbers and remaining tuning knobs

### MC² 算子性能调优案例 — Huawei's own measured MatmulAllReduce fusion gain and the AllReduce latency fit

https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0043.html
(static mirror used for extraction: https://www.hiascend.com/doc_center/source/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0043.html — CANN commercial 8.5.0, "MC²算子性能调优案例")

```text
MC2通算融合算子的性能收益主要来自于通信、计算的并行执行，即将输入数据切分为多个子块，子块的计算和通信任务形成两条流水线，通过两条流水线上任务的并行执行，实现流水掩盖，从而提升算子性能。如下图所示，MC2算子先做Matmul计算、后通信的场景，输入矩阵沿M轴被切分为两块，第二块数据的Matmul计算和第一块数据的通信可以并行执行，从而达到计算和通信时间相互掩盖的目的。
```

```text
MC2算子性能收益公式为：
融合前算子串行耗时 = 融合前计算算子耗时 + 融合前通信算子耗时
MC2算子收益 = (融合前算子串行耗时 - 融合后MC2算子耗时) / 融合前算子串行耗时
```

```text
因素一：计算和通信的执行时间差异
若计算和通信任务的执行时间相差不大，则融合后，MC2算子的计算和通信并行执行，实现流水掩盖效果，性能收益较大。
若计算和通信任务的执行时间差异较大，则融合后，MC2算子内计算和通信并行执行，能够掩盖的时间较少，算子整体执行耗时与未切分串行时的算子执行耗时接近，此时无法获得较大性能收益。

因素二：数据切分导致的计算或通信的执行时间膨胀
当对输入数据进行切分后，原本的整块数据被切分成若干小数据块，对若干小数据块分别做Matmul计算或者执行通信任务，此时相比切分前，计算或者通信任务的执行时间可能发生膨胀（即执行时间变长）。该膨胀产生的原因包括：切分后的数据块过小导致计算或通信的效率降低、切分的数据块过多导致增加额外的调度开销、并行执行后计算和通信对L2 Cache或device存储内存的访问冲突等。
```

```text
未发生膨胀：
数据切分前，Matmul执行时间为200us，将Matmul的输入均匀切分为两块，假设切分后，每块数据的Matmul执行时间都是100us，通过计算的并行执行，下图实际性能收益为100us。

发生一般程度的膨胀：
数据切分前，Matmul执行时间为200us，将Matmul的输入均匀切分为两块，假设切分后，每块数据的Matmul执行时间都是150us，通过计算的并行执行，下图实际性能收益为50us。

发生严重程度的膨胀：
数据切分前，Matmul执行时间为200us，将Matmul的输入均匀切分为两块，假设切分后，每块数据的Matmul执行时间都是200us，通过计算的并行执行，下图实际性能收益为劣化50us。
```

The M-axis-only constraint, verbatim:

```text
只切分M轴。因为通信任务调用的Hccl API要求分块数据内存连续，若按N轴切分，则每行数据都被切断，导致通信数据的内存不连续，不满足通信要求；若按M轴切分，则每行数据都是内存连续的，满足通信要求。
若A表示长块、B表示短块，只能切出A或B连续排布的形式，例如AAAB、BAAA等情况。
```

The measured AllReduce cost model (8 cards, A2, half, N=8192):

```text
数据量x = m * N * sizeof(dataType)，单位是Bytes。该拟合公式表示为：
x小于8MB：t = -0.9698202 * x * x + 27.0622573 * x + 14.769，单位是us。
x大于等于8MB：t = 13.58491263 * x + 61.508333，单位是us。
```

```text
注意：通信和计算并行执行时，可能出现抢占内存带宽的情况，导致执行时间增加，一般按经验在拟合公式中乘以1.15的系数，用户可以根据实测情况调整该系数。
```

The worked example with hard numbers:

```text
本MatmulAllReduce案例中，给定的输入矩阵Shape为M=4096，K=3072，N=8192，数据类型为half，分核数为8，通过融合前msProf工具采集，得到该输入的Matmul计算执行时间为803us，AllReduce通信的执行时间为1071us，总耗时1874us，属于通信bound场景。
...
最终得到将原始输入矩阵切分为5个数据块，长度分别为：{512，896，896，896，896}。
```

```cpp
MatmulAllReduceCustomTilingData *tiling = context->GetTilingData<MatmulAllReduceCustomTilingData>();
tiling->param.rankDim = 8;
tiling->param.tileM = 512; // 短块大小
tiling->param.tileNum = 1; // 短块个数
tiling->param.tailM = 896; // 长块大小
tiling->param.tailNum = 4; // 长块个数
tiling->param.rankM = 4096;
tiling->param.rankN = 8192;
tiling->param.rankK = 4096;
tiling->param.isTransposeA = 0;
tiling->param.isTransposeB = 0;
tiling->param.cToFloatLen = 0;
tiling->param.nd2NzWorkLen = true;
tiling->param.dataType = static_cast<uint8_t>(HCCL_DATA_TYPE_MAP.at(aType));
```

```text
按该切分策略测试，融合后该算子的执行时间为1262us，则融合算子的性能收益为(1874 - 1262) / 1874 = 32.7%。
```

```text
针对切分膨胀做调整
上文提到，切分数据会引起计算或通信执行时间的膨胀，使实测结果与理论值有偏差。比如切块数量较多时，执行时间的膨胀对性能影响较大，可能导致性能收益变小或者出现性能劣化，因此最终需要根据上述理论切分策略，结合实测，对切分策略做调整。
```

## Relevance
The intercept of the small-message branch of Huawei's own fit is `14.769 us` at x→0 — i.e. an 8-card AllReduce on A2 has a measured fixed cost around 15 us even for zero bytes. That is the per-call floor to beat, and at TP16 on A3 it will be larger. Also note the fusion mechanism only works when compute and comm times are comparable ("计算和通信的执行时间差异"); at batch=1 the GEMM per layer is tiny relative to the collective, so the M-axis pipelining this doc describes has almost nothing to pipeline — which is consistent with Huawei's own rule that MC2 is disabled in 增量 (decode).

---

### NVSHMEM measured intra-node small-message latencies (H200 / NVLink-4)

https://arxiv.org/abs/2606.05951 · https://arxiv.org/html/2606.05951v1
Yijun Ma, Siyuan Shen, Tiancheng Chen, Akhil Langer, Jiri Kraus, Benjamin Glick, Craig Belusar, Jeff Hammond, Torsten Hoefler. Submitted 4 Jun 2026.

Testbed, verbatim:

> All measurements in this section were collected on a CoreWeave H200 cluster. Each node contains eight NVIDIA H200 SXM5 GPUs with 144 GB HBM3e per GPU, connected intra-node by NVLink-4. The cluster notes report 900 GB/s bidirectional NVLink bandwidth per GPU and NVLink-SHARP (NVLS) multicast support. Inter-node communication uses ConnectX-7 InfiniBand with eight NICs per node.

One-sided RMA, bandwidth and the push/pull asymmetry:

> Intra-node bulk put reaches 313 GB/s, while get peaks at 141 GB/s. Scalar p reaches 172 GB/s because many GPU threads issue independent remote stores that can be buffered and overlapped. In contrast, scalar g stays below 9 GB/s because for each remote load, the thread needs the returned value before the operation completes, limiting the number of outstanding operations and preventing deep pipelining. These results remain below the 450 GB/s NVLink reference due to NVSHMEM address translation, synchronization, work partitioning, and protocol overheads. Across nodes with tuned IBGDA settings, bulk put and get both approach the single-rail IB reference at 48.0 GB/s and 48.2 GB/s, respectively. Scalar p improves substantially and peaks at 15.6 GB/s, while scalar g remains lower at 1.28 GB/s.

The small-message latency numbers:

> The latency insets show why small messages are not bandwidth efficient. Intra-node operations take about 1.8–2.5 μs for bulk put/get and 1.3–2.2 μs for scalar p/g. With IBGDA, inter-node put/get take about 9.4–9.5 μs at 256 B and remain near 9.7 μs at 64 KiB, while scalar p/g take about 7.5 μs and 25.3 μs at 256 B. Thus, NVSHMEM is most effective for bulk or aggregated write-style RMA, while scalar operations should be treated as latency/control primitives and batched when bandwidth matters.

Device-initiated (single-CTA) vs host-side (multi-CTA) collectives:

> Figure 5 shows that multi-CTA execution is crucial for NVSHMEM collective performance. Intra-node, the host-side on-stream path reaches 264 GB/s in algorithm bandwidth, outperforming the forced NCCL ring baseline and approaching NCCL's NVLS path at 276 GB/s, while the device-side block-scoped path peaks at only 30 GB/s because it uses a single CTA. For small messages, however, the device path remains latency-competitive, taking about 3.8–7.1 μs up to 64 KiB, compared with roughly 4.7–8.9 μs for NCCL ring and 5.6–5.9 μs for NCCL NVLS. Across nodes, both NVSHMEM variants remain below 0.20 GB/s, while NCCL reaches 180 GB/s with ring and 252 GB/s with NVLS Tree.

DeepEP's low-latency inference path:

> DeepEP's low-latency path targets inference, where batch sizes are small and end-to-end layer latency matters more than peak bandwidth. Accordingly, it removes the intra-node NVLink forwarding stage used by the high-throughput path and relies on RDMA for inter-node delivery.

> During dispatch, the sender warps place tokens into per-peer RDMA ring buffers on the symmetric NVSHMEM heap, and the sender-coordinator warp periodically batches these writes into larger RDMA transfers through nvshmemi_ibgda_put_nbi_warp, updating the remote tail with nvshmemi_ibgda_amo_nonfetch_add.

## Relevance
Two concrete transfers. First, the push/pull asymmetry is not small: scalar put 172 GB/s against scalar get <9 GB/s, because a load blocks on its return value and cannot pipeline. If any part of our reduce reads peer memory rather than being pushed, that is a ~20x structural penalty on issue rate. Second, a *fully device-initiated single-CTA* collective is latency-competitive for small messages (3.8–7.1 us to 64 KiB) but caps at 30 GB/s — exactly the trade a persistent-kernel in-graph reduce makes, and the reason it is the right choice only at batch=1.

---

### vLLM `QuickAllReduce` (AMD) — quantized-in-kernel allreduce and its per-(dtype, world_size) size table

https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/quick_all_reduce.py
https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html (AMD ROCm blog, "QuickReduce: Up to 3x Faster All-reduce for vLLM and SGLang")
https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-3/README.html (AMD ROCm blog, "QuickReduce INT3 Quantization and Benchmarking on MI355")

The size-threshold table, verbatim from `quick_all_reduce.py`:

```python
MB = 1024 * KB
...
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
...
        (torch.float16, 2): [1 * MB, 2 * MB, 2 * MB, 1 * MB, 1 * MB],
        (torch.float16, 4): [1 * MB, 16 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.float16, 8): [16 * MB, 4 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.bfloat16, 2): [2 * MB, 8 * MB, 8 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 4): [8 * MB, 64 * MB, 64 * MB, 16 * MB, 16 * MB],
        (torch.bfloat16, 8): [
            16 * MB,
            2048 * MB,
            2048 * MB,
            2048 * MB,
            2048 * MB,
        ],
```

```python
        # INT3 is only enabled for TP2 (world_size == 2).
        ...
        if self.qr_quant_level == QuickReduceRegime.INT3 and self.world_size != 2:
            ...
                "only supported for TP2 (world_size == 2), but world_size "
```

```python
        # VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB is specified in MB
        qr_max_size = envs.VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB
        ...
                    "You should not set a max_size smaller than 1MB, which can "
        ...
            qr_max_size = qr_max_size * MB
        ...
        self._ptr = ops.init_custom_qr(self.rank, self.world_size, qr_max_size)
        ...
                "Custom quick allreduce: min size override = %d MB",
                qr_min_size // MB,
        ...
                "Custom quick allreduce: quantization codec threshold = %d KB",
```

As published in the docs and blog:

> Compression and decompression happen inside the thread-level data exchanges of the kernel, not through quantizing all the data up front.
> Q3 (INT3) is restricted to TP2 (world_size == 2) due to poor performance on larger world sizes.
> QuickReduce originally supported only fp16 inputs. We extended it to support bfloat16 (bf16) as well.
> On RocM, bfloat16 kernels are slower than fp16 due to slower match operations.
> Set the environment variable `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` to a level such as FP, INT8, INT6, or INT4. The value NONE disables it.
> Only the ROCm MI300 series is supported for quick allreduce at this time. It is a complement to custom allreduce and has stricter initialization requirements.

## Relevance
This is the one production allreduce that quantizes *inside* the kernel's thread-level exchanges rather than quantizing the tensor first. For a W4A8C8 model where the activations crossing the TP boundary are already low precision, the same idea applies directly: push int8 instead of bf16 and halve the pushed bytes, accepting the same accuracy argument our quantization already accepts. Note the hard caveat that INT3 was restricted to world_size==2 for performance, i.e. aggressive codecs do not survive wide fan-in.

---

### CoCoNet / Centauri / COMET / DistributedGEMM — the rest of the fused-collective family

https://proceedings.mlsys.org/paper_files/paper/2025/hash/e27ea0cd50b798ff8942caf9203f0992-Abstract-Conference.html (COMET, MLSys 2025) · arXiv:2502.19811 · https://github.com/bytedance/flux
https://seed.bytedance.com/en/blog/comet-has-been-deployed-in-large-scale-clusters-saving-millions-of-gpu-hours-moe-communication-optimization-technology-comet-is-now-open-source
Shulai Zhang, Ningxin Zheng, Haibin Lin et al. (ByteDance Seed).

As published:

> Large MoE models spend a lot of time on inter-device communication. One figure cited is 47% of total execution time. Earlier overlap methods split work coarsely, which hurts compute efficiency.
> The system uses data dependency analysis and task rescheduling to overlap communication and computation at a fine grain. It also adjusts workload assignment adaptively.
> Results: a 1.96× speedup for a single MoE layer and a 1.71× average speedup end to end.
> ByteDance says COMET runs in production on clusters with tens of thousands of GPUs and has saved millions of GPU hours.

## Relevance
COMET's 47% figure is for MoE AllToAll in training, not decode TP allreduce; its mechanism (dependency analysis + task rescheduling inside the kernel) is the same family as Flux/TileLink. Recorded here for completeness of the approach family rather than as a transferable number — none of COMET, CoCoNet or Centauri publishes a batch=1 decode allreduce latency.

---

## Part 6 — Barrier elimination, measured: SiFAR and the SGLang/vLLM push protocol

### SiFAR: Synchronization-Free All-Reduce for Low-Latency LLM Inference (the closest published match to our problem)

https://arxiv.org/abs/2607.08973 · https://arxiv.org/html/2607.08973v1
Georgia Institute of Technology, Atlanta, Georgia, USA; NVIDIA, Santa Clara, California, USA.
Keywords: Low-Latency LLM Inference, All-Reduce, In-Switch Reduction.

Abstract, verbatim:

> The rise of reasoning models and agentic systems has made LLM token-generation latency a key bottleneck. Unlike chatbots, where latency gains saturate at human reading speed, these systems generate intermediate reasoning tokens that are not consumed by humans. As a result, per-token latency directly determines end-to-end response time. To reduce latency, inference engines operate with minimal batching, making token generation bandwidth-bound. Tensor Parallelism addresses this bottleneck by sharding model weights across GPUs and loading them in parallel. However, scaling to more GPUs introduces All-Reduce overheads that grow with GPU count. We observe that removing All-Reduce improves token throughput by 43% for Llama-3.1-8B on 8 H200 GPUs.
>
> To address this overhead, we propose Synchronization-Free All-Reduce (SiFAR), which reduces synchronization overheads incurred by All-Reduce during low-latency inference. Existing oneshot and twoshot algorithms incur high synchronization overheads due to barriers before and after communication, accounting for 32–62% of the latency for small payloads we observe in low-batch serving. To tackle this, first, we find that the bottom barrier in oneshot enforces a WAW dependency, and eliminate it by co-designing communication and model execution to enable dual buffering. However, oneshot scales poorly with GPU count. Twoshot performs better at higher TP degrees, but incurs an unavoidable bottom barrier. To overcome this, we leverage in-switch reduction in modern switches, primarily used for reduce-scatter. We propose redundant pull, where each GPU reduces the entire all-reduce payload at the switch. This improves oneshot scalability while retaining its no bottom barrier advantage. Finally, to reduce top barrier overhead, we observe that each token generation step issues multiple All-Reduce operations, keeping GPUs tightly synchronized after the first All-Reduce. We therefore propose speculative reduction, which initiates data transfer without waiting for the top barrier and ensures correctness via a lightweight validation mechanism. SiFAR reduces All-Reduce latency by up to 52% and improves end-to-end throughput by 18.6% for Llama-3.1-8B and 13.1% for Qwen3.5-397B-17B at TP=8.

The motivating measurement — megakernel + TP scaling:

> To quantify this trend, we measure time per output token (TPOT) for Llama 3.1 8B on H200 GPUs using a Megakernel implementation (Spector et al., 2025). As we scale from TP=1 to TP=8, TPOT reduces from 2.73 ms to 1.43 ms, and the fraction of time in All-Reduce increases from 0% to 30% (corresponding to a 43% throughput gain). Thus, All-Reduce becomes a growing contributor to latency and is poised to dominate as other bottlenecks diminish. Despite this, optimizing All-Reduce for low-latency inference remains largely unexplored.

> Recent system optimizations mitigate many of these non-communication bottlenecks. For example, Megakernels (Wu et al., 2025b; Spector et al., 2025) fuse the forward pass into a single kernel to eliminate launch overheads and improve HBM utilization, while Programmatic Dependent Launch (PDL) (NVIDIA, 2025b) enables weights prefetching across kernel boundaries. Thus, as non-communication overheads are mitigated, All-Reduce becomes the dominant bottleneck in low-latency inference.

> For Llama-3.1-8B, removing All-Reduce improves performance by 18% at TP=2 and 43% at TP=8. For Qwen3.5-397B-17B, removing All-Reduce improves performance by 16% at TP=4 and 21% at TP=8.

The barrier breakdown — the headline numbers:

> Figure 7. Breakdown of All-Reduce latency on H200 GPUs. Synchronization overhead accounts for 32-50% of oneshot and 49-62% of twoshot latency for small payloads across all TP configurations.

> We observe that synchronization overhead accounts for 32-50% of oneshot and 49-62% of twoshot latency for small payloads (8-32 KB) across all TP configurations. For example, at TP=4 with 8 KB payload, oneshot takes 4.66 μs with barriers but only 2.33 μs without, a 2× slowdown purely from synchronization. For twoshot at TP=8 with 8 KB payload, barriers account for 3.15 μs out of 5.11 μs total, representing 62% overhead. Twoshot exhibits higher synchronization overhead because its bottom barrier must ensure that broadcast writes are visible before the next operation, which requires a more expensive acquire-release barrier that enforces write ordering. These sizes of 4-8 KB represent the All-Reduce payloads for most leading models (Liu et al., 2024; Guo et al., 2025; Grattafiori et al., 2024; Yang et al., 2025; Agarwal et al., 2025; Team et al., 2025) at batch size 1. Thus, during autoregressive decoding, every All-Reduce operation encounters this synchronization bottleneck, spending more time waiting at barriers than transferring and reducing data.

Insight 1 — dual buffering removes the bottom (WAW) barrier:

> The bottom barrier in oneshot prevents Write-After-Write (WAW) hazards by ensuring all peers finish reading the payload buffer before the owning GPU overwrites it with new data. However, WAW is an avoidable dependency that can be eliminated through redundancy. Despite this, dual buffering is not widely used in current LLM serving systems because communication is handled by application-agnostic libraries like NCCL, which cannot make assumptions about buffer usage patterns and must include a bottom barrier to ensure correctness.

> The bottom barrier in oneshot ensures all peers have finished reading the payload buffer before the owning GPU overwrites it with new data, enforcing a write-after-write dependency. We eliminate this barrier using dual buffering. The inference engine alternates between two payload buffers across successive All-Reduce operations, ensuring that the payload is never overwritten before all peers finish reading. While dual buffering is well known, it is not widely used in current LLM serving systems because communication libraries like NCCL (NVIDIA Corporation, 2025a) are decoupled from model execution and cannot make assumptions about buffer reuse patterns. SiFAR overcomes this by co-designing communication with model execution, enabling dual buffering in systems like vLLM (Kwon et al., 2023) and Megakernels (Spector et al., 2025) without changes to their memory management (Figure 1 (b)).

The TP-scaling numbers for one-shot vs two-shot data transfer:

> However, removing the bottom barrier is insufficient. Oneshot scales poorly with GPU count, making it slower than twoshot beyond TP=2. At TP=8 with an 8 KB payload, oneshot takes 4.01 μs for data transfer compared to 1.96 μs for twoshot. While twoshot has lower data transfer latency, it requires a bottom barrier that cannot be eliminated because it enforces a read-after-write dependency after the broadcast phase of twoshot. Ideally, we want oneshot latency to be comparable to twoshot so that dual buffering is effective.

> In oneshot, each GPU sends and receives K×(N−1) bytes (where K is the payload size and N is the number of GPUs), scaling linearly with GPU count. In contrast, twoshot keeps data transfer constant by decomposing reduction into two steps: reduce-scatter and broadcast. The data transfer during twoshot can be further reduced using in-switch computation capabilities (NVIDIA Corporation, 2025b; Wang et al., 2025) of modern switches. With in-switch computation, across both steps of twoshot, each GPU sends K+K/N and receives K bytes.

Insight 2 — redundant pull via in-switch reduction:

> Figure 10. Redundant Pull: each GPU issues ld_reduce over the entire K bytes. The switch reduces data from all peers and returns the result, reducing data received per GPU from K×(N−1) to K.

> However, we find that ld_reduce can reduce oneshot latency beyond its conventional use in the reduce-scatter phase of twoshot. Reduce-scatter latency remains flat up to 64 KB and grows only slightly up to 256 KB. At TP=8, a 256 KB reduce-scatter means each GPU reduces 32 KB at the switch, which is comparable to the 4-32 KB payloads in low-batch inference.

Insight 3 — speculative reduction removes the top (RAW) barrier:

> The top barrier ensures all GPUs have produced valid partials before communication begins, enforcing a RAW dependency that cannot be simply eliminated. However, we observe that LLM decoding issues multiple All-Reduce operations per token. As a result, GPUs remain tightly synchronized after the first All-Reduce, as they execute identical operations across successive All-Reduces. Thus, the overhead of the top barrier is primarily due to flag exchange, not divergence between GPUs. This leads to our third optimization: speculate and verify. When a GPU enters an All-Reduce, it speculatively begins fetching data from peers, assuming their data is ready, thereby avoiding explicit flag exchange. To ensure correctness, each GPU writes a flag to a validation buffer that is reduced alongside the payload. If the result equals N×flag after reduction, all GPUs had valid data and speculation succeeded. Otherwise, SiFAR reruns the All-Reduce.

> Figure 12. Representative Nsight profiler output: the cost of the top barrier decomposes into divergence (dynamic overhead) and flag exchange (static overhead). Because kernels launch via CUDA Graphs, divergence is near-zero after the first All-Reduce, making flag exchange the dominant cost.

Implementation detail of the validation mechanism:

> The payload buffer is reduced using switch_reduce, which performs reduction in the switch. For the validation buffer, we instead use pull-and-local-reduce to compute valid_out, as it empirically outperforms using switch_reduce for this purpose. Each threadblock runs with one extra warp to process the validation buffer while the remaining warps handle the payload. Before beginning the All-Reduce, each GPU writes the flag value into its validation buffer. After reduction, we check if the result matches ngpus × flag. If validation fails, we re-execute until validation succeeds.

> To maintain correctness across multiple All-Reduce invocations, we increment the flag value after each All-Reduce, ensuring each All-Reduce has a unique validation result. We apply dual buffering to the validation buffer to prevent race conditions where a fast GPU might overwrite its flag before all peers have finished reading it. Since remote loads are cached in L1 (Fusco et al., 2024), upon mis-speculation we use the .cg modifier to load the validation buffer to bypass L1 and fetch fresh data. The payload switch_reduce does not require this, since data is fetched from the switch and not cached locally. The correctness of the verification logic relies on GPU monotonicity; see Section 8.1 for details.

> In Megakernels (Spector et al., 2025), the entire forward pass is fused into a single kernel, and successive operations within the kernel communicate completion via flags. SiFAR reuses these existing operator flags for validation by reading and reducing the flag from all remote GPUs, instead of explicitly writing to a separate validation buffer. This eliminates the overhead of writing to the validation buffer before each All-Reduce, further reducing latency.

Evaluation methodology:

> We evaluate SiFAR by integrating it into the Megakernel framework (Spector et al., 2025). Megakernels fuse the entire forward pass into a single CUDA kernel, eliminating kernel launch overheads and improving HBM utilization. Within the Megakernel, weights of the next operation are prefetched into shared memory whenever possible, including during All-Reduce, further reducing latency. We evaluate on Megakernels rather than serving systems like vLLM (Kwon et al., 2023) because Megakernels represent the state-of-the-art for low-latency inference, where non-communication overheads are already minimized, making the impact of All-Reduce optimization most visible. We extend the original Megakernel implementation to support Llama-3.1-8B and Qwen3.5-397B-17B with multi-GPU tensor parallelism. Our Megakernel baseline without SiFAR already achieves 1.3× lower TPOT than vLLM 0.18.1 for Llama-3.1-8B and 2.2× for Qwen3.5-397B-17B at TP=8, confirming that non-communication overheads are substantially reduced. We run both models on batch size 1 unless otherwise specified.

> Figure 15 compares standalone All-Reduce latency across five implementations: oneshot, twoshot, TRT-LLM, Lamport, and SiFAR for TP configurations of 2, 4, and 8 GPUs with payloads from 8 to 32 KB. The baseline for each configuration is the best of oneshot and twoshot, which is the default in vLLM and SGLang. We execute All-Reduce back-to-back within a Megakernel to avoid measuring kernel launch overheads. The measured latency includes any mis-speculation overheads incurred by SiFAR.

Mis-speculation cost — the critical risk number:

> For Llama-3.1-8B at TP=8, SiFAR achieves a 18.6% improvement. With perfect speculation, the improvement increases to 20.2%, meaning mis-speculation retries cost only 1.6% despite a 32.2% average mis-speculation rate (Figure 24). Removing barriers entirely yields 21.6%, so verification overhead accounts for an additional 1.4%. Together, mis-speculation and verification account for only 3.0% of lost improvement, showing that speculative reduction captures most of the available benefit.

> Figure 24. Mis-speculation rate for Llama-3.1-8B and Qwen3.5-397B-17B. While mis-speculation rate increases with TP degree, it results in at most 1.6% lost performance.

> Figure 25 shows the SiFAR latency for the no-retry case and the added latency from each retry. We measure this by forcing SiFAR to retry a fixed number of times. Even with retries, the average SiFAR latency stays well below the baseline (oneshot or twoshot) across all evaluated TP degrees and payload sizes. Each retry costs less than the first attempt, which pays the one-time startup cost of beginning the All-Reduce.

> Figure 21 reports TPOT percentiles at TP=8, normalized to the baseline p50. SiFAR lowers both median and tail latency: for Llama-3.1-8B it reduces p50 to 81.4% and p99 to 84.0%, and for Qwen3.5-397B-17B it reduces p50 to 86.3% and p99 to 87.8%. The speedup from SiFAR holds through p99.9, so mis-speculation retries do not inflate tail latency.

TP is the only parallelism that helps TPOT:

> Other parallelism strategies are less effective for reducing TPOT. Pipeline parallelism (Huang et al., 2019) does not increase memory bandwidth, as each stage still loads full model weights sequentially. Expert parallelism (Fedus et al., 2022) introduces load imbalance across experts, leading to uneven bandwidth utilization. As a result, neither approach is optimal for reducing TPOT. Thus, in this work, we focus on TP.

> TP shards weight matrices across GPUs, requiring each layer to perform two All-Reduce operations: one after attention and one after the feed-forward network.

## Relevance
This paper is our problem statement, measured on GPUs. Three of its findings map onto our 5.35 ms directly. (1) The payload size regime matches: "These sizes of 4-8 KB represent the All-Reduce payloads for most leading models at batch size 1" — our TP16 batch=1 activations are in the same few-KB range, where synchronization is 32–62% of the cost. (2) The two barriers have *different* dependency types and different fixes: the bottom barrier is WAW and removable by alternating two payload buffers (free, no correctness risk, requires only that the engine not reuse the buffer before the next reduce); the top barrier is RAW and only removable by speculate-and-verify, at a measured 32.2% mis-speculation rate that costs just 1.6% of the win because a retry is cheaper than the first attempt. (3) The specific enabling observation for speculation is ours too: "LLM decoding issues multiple All-Reduce operations per token. As a result, GPUs remain tightly synchronized after the first All-Reduce" — with ~60 reductions a round in our model, after the first one the dies are implicitly in lockstep, so the top barrier is paying for flag exchange rather than for real divergence. Note the one part that does **not** transfer: `redundant pull` depends on in-switch reduction (NVLink SHARP `ld_reduce`), which A3's HCCS fabric does not provide; on Ascend the equivalent lever is `WriteReduceNbi`, and that is UB_CTP/950-only (see Part 2).

---

### vLLM `push_all_reduce` — SGLang's barrier-free push protocol, with a measured batch=1 TP=8 decode TPOT

https://github.com/vllm-project/vllm/commit/e3b4fdaf5db1aed53675c8467bc4862ba53bbfff
Alexander Matveev <amatveev@redhat.com>. Ported from `sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh`, SGLang commit `edb1b3f`.

The full commit message, verbatim:

```text
perf: add push-based allreduce for small tensor reductions

Port SGLang's push-based 2-buffer allreduce protocol into vLLM as a new
communicator backend for small-message reductions. The push protocol
eliminates the two explicit cross-GPU NVLink barrier round-trips used by
the existing barrier-based CustomAllreduce, replacing them with a
sentinel-based data arrival detection mechanism and double-buffered epoch
alternation.

Key advantages over the barrier-based approach:
- Zero barriers: data arrival IS the synchronization (positive-zero sentinel)
- Single NVLink round-trip instead of two barrier exchanges + remote reads
- All SMs active (SM_count CTAs vs 2 CTAs) for higher NVLink bandwidth
- No cudaMemcpy to IPC staging buffer in eager mode
- PDL (griddepcontrol) support for kernel overlap on sm_90+

The new PushAllReduce is inserted in the CudaCommunicator dispatch chain
above the existing CustomAllreduce for messages below a size threshold
(~720 KB at TP=8). Larger messages continue to use the barrier-based
path. The existing CustomAllreduce code is not modified.

Measured results on DeepSeek-V4-Pro (61 layers, TP=8, 8x NVIDIA B200,
BS=1, decode with ISL=4, OSL=33024):
- Throughput: +2.14% (84.06 vs 82.30 tokens/s)
- TPOT: -2.09% (11.90 vs 12.15 ms/token)

Correctness verified via lm_eval gsm8k 5-shot with no regression
(exact_match delta within statistical noise).

The feature can be disabled at runtime via VLLM_DISABLE_PUSH_ALLREDUCE=1
to fall back to the barrier-based path.

Signed-off-by: Alexander Matveev <amatveev@redhat.com>
```

Files changed: `CMakeLists.txt`, `csrc/push_all_reduce.cu`, `csrc/push_all_reduce.cuh`, `csrc/push_all_reduce_kernel.cuh`, `csrc/torch_bindings.cpp`, `tests/distributed/_test_push_ar_worker.py`, `tests/distributed/test_push_all_reduce.py`, `vllm/_custom_ops.py`, `vllm/distributed/device_communicators/cuda_communicator.py`, `vllm/distributed/device_communicators/push_all_reduce.py`, `vllm/distributed/parallel_state.py`.

The header comment of the ported kernel, verbatim:

```cuda
// Push-based 2-buffer allreduce kernel, ported from SGLang's
// all_reduce_one_shot_push_kernel. Uses epoch-based double-buffered
// protocol with positive-zero sentinel for data arrival detection.
//
// Original source:
//   sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh
// Protocol reference: SGLang commit edb1b3f
//
// Changes from SGLang:
//   - All code placed in namespace vllm::push_ar
//   - SGL_DEVICE macros replaced with __device__ __forceinline__
//   - SGL_CUDA_ARCH replaced with __CUDA_ARCH__
//   - std::integral (C++20) replaced with explicit overloads (C++17)
//   - kMaxVecBytes hardcoded to 16 (push kernel uses 16-byte vectors)
//   - TVM/FFI dependencies removed; file is self-contained
```

The sentinel mechanism, verbatim:

```cuda
template <typename T>
struct fp_trait {};

template <>
struct fp_trait<bf16_t> {
  using type = uint16_t;
  [[maybe_unused]] static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]] static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<fp16_t> {
  using type = uint16_t;
  [[maybe_unused]] static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]] static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<float> {
  using type = uint32_t;
  [[maybe_unused]] static constexpr uint32_t pos_zero = 0x00000000u;
  [[maybe_unused]] static constexpr uint32_t neg_zero = 0x80000000u;
};

template <typename DType>
__device__ __forceinline__ void clear_pos_zero(DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<typename Trait::type*>(&val);
  if (*ptr == Trait::pos_zero) *ptr = Trait::neg_zero;
}

template <typename DType>
__device__ __forceinline__ bool is_pos_zero(const DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<const typename Trait::type*>(&val);
  return *ptr == Trait::pos_zero;
}

template <typename DType>
__device__ __forceinline__ DType get_pos_zero() {
  using Trait = fp_trait<DType>;
  const auto value = Trait::pos_zero;
  return *reinterpret_cast<const DType*>(&value);
}
```

The epoch controller and the data structures:

```cuda
static constexpr uint32_t kMaxNumGPU = 8;

struct PushController {
  using SignalType = uint32_t;
  static constexpr int64_t kNumStages = 2;  // double-buffered epochs

  PushController() : m_local_signal(nullptr) {}

  PushController(void* ptr) : m_local_signal(static_cast<SignalType*>(ptr)) {}

  __device__ __forceinline__ SignalType epoch() const {
    return m_local_signal[blockIdx.x];
  }

  __device__ __forceinline__ void exit() const {
    __syncthreads();
    if (threadIdx.x == 0) {
      exit_unsafe(blockIdx.x);
    }
  }

  __device__ __forceinline__ void exit_unsafe(uint32_t which) const {
    auto& signal = m_local_signal[which];
    signal = (signal + 1) % kNumStages;
  }

  SignalType* m_local_signal;
};

struct AllReducePushData {
  void* __restrict__ buffer[kMaxNumGPU];
  const void* input;
  void* output;
  uint32_t rank;
  uint32_t num_items;
  uint32_t buffer_bytes;
  uint32_t epoch_bytes;
};
```

The push half — note `clear_pos_zero` applied to every element before the remote store, so that a genuine `+0.0` in the data can never be mistaken for "not yet arrived":

```cuda
template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void push_impl(DType* (&push_buf)[kNumGPU],
                                          const void* data,
                                          uint32_t num_items) {
  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using Storage = AlignedVector<packed_t<DType>, kVecSize>;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage vec;
    vec.load(data, offset);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      clear_pos_zero(vec[j].x);
      clear_pos_zero(vec[j].y);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      st_global_volatile_16B(vec, push_buf[i], offset);
    }
  }
}
```

The poll half — the whole barrier is replaced by polling the *data* for the sentinel, and the buffer is re-armed with sentinels at the end:

```cuda
template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void poll_impl(DType* (&poll_buf)[kNumGPU],
                                          void* data, uint32_t num_items) {
  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using Storage = AlignedVector<packed_t<DType>, kVecSize>;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage storage[kNumGPU];

    while (true) {
      bool has_pos_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kNumGPU; ++i) {
        ld_global_volatile_16B(storage[i], poll_buf[i], offset);
#pragma unroll
        for (auto j = 0; j < kVecSize; ++j) {
          has_pos_zero |= is_pos_zero(storage[i][j].x);
          has_pos_zero |= is_pos_zero(storage[i][j].y);
        }
      }
      if (!has_pos_zero) break;
    }

    const Storage result = reduce_impl(storage);
    result.store(data, offset);

    Storage pos_zeros;
    pos_zeros.fill({get_pos_zero<DType>(), get_pos_zero<DType>()});
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      pos_zeros.store(poll_buf[i], offset);
    }
  }
}
```

The kernel, with PDL for cross-kernel overlap:

```cuda
template <typename DType, uint32_t kNumGPU, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void all_reduce_one_shot_push_kernel(
    const AllReducePushData __grid_constant__ params,
    const PushController __grid_constant__ ctrl) {
  const auto [buffer, input, output, rank, num_items, buffer_bytes,
              epoch_bytes] = params;

  PDLWaitPrimary<kUsePDL>();

  // Phase 1: Push data from input to all ranks' push buffers
  const auto epoch_offset = ctrl.epoch() * epoch_bytes;
  DType* push_buf[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i) {
    push_buf[i] = static_cast<DType*>(
        ptr_byte_offset(buffer[i], rank * buffer_bytes, epoch_offset));
  }
  push_impl(push_buf, input, num_items);

  PDLTriggerSecondary<kUsePDL>();

  // Phase 2: Poll local buffer, reduce, write output, reset
  DType* poll_buf[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i) {
    poll_buf[i] = static_cast<DType*>(
        ptr_byte_offset(buffer[rank], i * buffer_bytes, epoch_offset));
  }
  poll_impl(poll_buf, output, num_items);
```

```cuda
template <bool kUsePDL>
__device__ __forceinline__ void PDLWaitPrimary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

template <bool kUsePDL>
__device__ __forceinline__ void PDLTriggerSecondary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}
```

The size thresholds, from `vllm/distributed/device_communicators/push_all_reduce.py`:

```python
_FEATURE_DESCRIPTION = "Push-based AllReduce"
_DISABLE_ENV_VAR = "VLLM_DISABLE_PUSH_ALLREDUCE"

# Push threshold map: world_size -> buffer_bytes
# From SGLang's tuned thresholds for sm100 (B200)
PUSH_THRESHOLD_SM100 = {
    2: 4 * 1024 * 1024,  # 4 MB
    4: 2 * 1024 * 1024,  # 2 MB
    6: 1 * 1024 * 1024,  # 1 MB
    8: 720 * 1024,  # 720 KB
}

# Conservative default for untuned GPUs
DEFAULT_PUSH_BUFFER = 512 * 1024  # 512 KB


class PushAllReduce:
    """
    Push-based custom allreduce using epoch-based 2-buffer protocol.
    Ported from SGLang's CustomAllReduceV2 push allreduce.
    """
```

```python
        # Prerequisite checks
        if self.world_size not in (2, 4, 6, 8):
        ...
        props = torch.cuda.get_device_properties(device)
        self.num_sm = props.multi_processor_count
        ...
        # Allow env var override (V1 fix: validate 128-byte alignment)
        env_override = os.environ.get("VLLM_PUSH_AR_BUFFER_BYTES")
        ...
            # Round up to 128-byte alignment for kernel volatile stores
```

## Relevance
This is a production, merged, barrier-free one-shot push allreduce with a batch=1 TP=8 decode measurement at **TPOT 11.90 ms** — the same order as our 11.6198 ms — on a 61-layer MoE. Three directly portable design decisions. (1) The sentinel *is* the data: `+0.0` means "not arrived", and the producer rewrites any genuine `+0.0` in its payload to `-0.0` before pushing, which is numerically free and removes both barriers. On Ascend the equivalent is a pre-armed symmetric buffer plus `aclshmem_bfloat16_put_nbi`, with the poll done by AIV reading the data rather than a `FLAG_SIZE`-wide `DataCopyGM2UB` with `pipe_barrier(PIPE_ALL)` per iteration. (2) "All SMs active (SM_count CTAs vs 2 CTAs) for higher NVLink bandwidth" — HCCL's AIV one-shot does the opposite, funnelling the whole Consumer reduce onto `blockIdx_ == numBlocks_ - 1`. (3) `__grid_constant__` params plus `cudaLaunchKernelEx` is how the peer-pointer array is delivered without breaking graph capture; the Ascend analogue is the three-pointer `SUPERKERNEL_ARGS_DEF` plus `get_para_base()`. Note the TP ceiling: SGLang's tuned table stops at world_size 8 and the kernel hardcodes `kMaxNumGPU = 8`, so TP16 is unexplored territory upstream.

---

---

# Part 7 — The in-kernel collective on A3: `aclshmem`, epoch tagging, and what orders a remote write

Collected 2026-10-20 for question 2.3, "how an in-kernel collective is made safe without a host
barrier". Everything below is verbatim. Provenance note from the collector, verbatim:

> Primary corpus for TOPIC A is the CANN `shmem` library. The canonical upstream is
> `https://gitcode.com/cann/shmem` (gitcode HTML is JS-rendered and raw paths 404 for
> anonymous access); a byte-identical GitHub mirror that *is* raw-fetchable is
> `immengzi/shmem` (HEAD = `5408559b89`, xmengh, 2026-03-04). All `immengzi/shmem` URLs
> below are that mirror of the Huawei-copyright source; the copyright headers in the quoted
> blocks confirm provenance (`Copyright (c) 2025 Huawei Technologies Co., Ltd.`,
> `CANN Open Software License Agreement Version 2.0`).

## Relevance
This is the material for fusing the TP all-reduce into the layer kernel, against our measured TP
reduce of 5.35 ms, 100% serial, inside an 11.6198 ms TPOT.
- The ordering question has a one-line answer: `aclshmem_quiet` is `PipeBarrier<PIPE_ALL>()` plus
  `dcci_entire_cache()`, `aclshmem_fence` is implemented identically, and
  `aclshmem_putmem_signal` is literally `put_nbi` → `quiet` → `signal_op`. So **`quiet` is the
  thing that orders a remote write against the consumer's read**, and it is a full-pipe barrier
  plus a whole-DCache flush — i.e. not free, and 78 layers × 2 all-reduces would pay it 156 times.
  See also `ascend-fusion-mechanics.md` §35: on 2201 the DCache is the only incoherent cache and
  only the Scalar unit uses it, which is why LCCL's choice to move flags by MTE instead of scalar
  stores lets it skip `dcci` entirely.
- Epoch tagging is the upstream norm, not an invention of TileRT's ROCm build. The `shmem` barrier
  design note states it directly: 「instead of simply write a flag, each rank writes a 64-bit
  number ... indicating how many times this team has performed barrier」, with 「no cache
  consistency ensurance」 named as the reason flag slots must be cacheline-isolated. LCCL puts a
  32-bit magic in the **high** half of an int64 and rejects stale values with
  `(v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue` — exactly the "16-bit epoch tag
  in the high half of each transferred word" shape our ROCm build uses, at 32 bits.
- Three ways the epoch is generated with **no host barrier**, all published: a host-side
  per-communicator counter passed as a kernel scalar (LCCL `int64_t magic_ = 1`, post-incremented
  per call); a device-side atomic-add into a symmetric slot followed by `SyncAll<true>()` (LCCL
  `SET_MAGIC`); and a per-block self-incrementing read-modify-write of the block's own magic slot,
  which needs no atomic because exactly one core owns each slot (sgl-kernel-npu
  `NotifyDispatch::GetMagicValue()`). The third is the right shape for a batch=1 megakernel.
- `magic % PING_PONG_SIZE` selecting which half of the symmetric window a generation writes into is
  the standard companion trick — it removes the need to ever zero the flag region, which is what
  would otherwise force a barrier.
- **The decisive A3 constraint:** of the device-side AMOs, only `atomic_add` is available on
  A2/A3; `atomic_set`, `atomic_swap`, `atomic_compare_swap`, `atomic_inc`, `atomic_fetch_inc`,
  `atomic_fetch_add` and the bitwise AMOs are all 950-only. So there is no device-side atomic SET
  or CAS to bump an epoch with on our hardware — the epoch must come from a scalar argument, a
  single-owner slot, or an atomic *add*.
- `aclshmemx_signal_op` carries 「WARNING: Atomicity NOT Guaranteed.」 in the mirror, reworded in
  current gitcode master to 「On Ascend950, RDMA is used as the communication engine, supporting
  signal_set with guaranteed atomicity」 — i.e. the guarantee is explicitly 950-only, which is
  another reason A3 should use the add/epoch form rather than a set.
- A one-sided `put` on A3 is address translation plus an ordinary `DataCopyPad`, and
  `EpilogueAllReduce` shows the TP all-reduce fused into a GEMM epilogue with three in-kernel
  barriers and an atomic-add reduce-scatter by remote *pull* — the closest published analogue of
  what we want the layer kernel to do.
- Two landmines to design around now: `shmem_device_cc.h` states 「Barrier APIs conflict with
  SyncAll. Avoid mixing them together」, and both depend on `shmemx_set_ffts_config` being called
  at kernel entry.

# TOPIC A — making an in-kernel collective safe without a host barrier

## A1. `aclshmem_quiet` / `aclshmem_fence` — the declared memory-ordering contract

https://raw.githubusercontent.com/immengzi/shmem/master/include/device/gm2gm/shmem_device_mo.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```c
/**
 * @brief The aclshmem_quiet routine ensures completion of all operations on symmetric data objects issued by the
 *        calling PE.
 *        On systems with only scale-up network (HCCS), updates are globally visible, whereas on systems with
 *        both scale-up network HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the
 *        memory of a given PE are visible to that PE.
 *        Quiet operations issued on the CPU and the NPU only complete communication operations that were
 *        issued from the CPU and the NPU, respectively. To ensure completion of GPU-side operations from
 *        the CPU, using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
ACLSHMEM_DEVICE void aclshmem_quiet(void);
#define shmem_quiet aclshmem_quiet

/**
 * @brief In OpenACLSHMEM specification, aclshmem_fence assures ordering of delivery of Put, AMOs, and memory store routines
 *        to symmetric data objects, but does not guarantee the completion of these operations.
 *        However, due to hardware capabilities, we implemented aclshmem_fence same as aclshmem_quiet, ensuring both ordering
 *        and completion.
 *        Fence operations issued on the CPU and the NPU only order communication operations that were issued from the
 *        CPU and the NPU, respectively. To ensure completion of GPU-side operations from the CPU,
 *        using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
ACLSHMEM_DEVICE void aclshmem_fence(void);
#define shmem_fence aclshmem_fence
```

## A2. What `quiet` actually compiles to — two instructions, nothing else

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/shmem_device_mo.hpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```cpp
ACLSHMEM_DEVICE void aclshmemi_quiet()
{
    // clear instruction pipes
    AscendC::PipeBarrier<PIPE_ALL>();

    // flush data cache to GM
    dcci_entire_cache();
}

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE void aclshmem_quiet()
{
    aclshmemi_quiet();
}

ACLSHMEM_DEVICE void aclshmem_fence()
{
    aclshmemi_quiet();
}
```

## A3. `dcci` wrappers — the exact AscendC intrinsic, and the compiler-barrier requirement

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/shmemi_device_cc.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 334-380)

```cpp
ACLSHMEM_DEVICE void dcci_cacheline(__gm__ uint8_t * addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

ACLSHMEM_DEVICE void dcci_cachelines(__gm__ uint8_t* addr, uint64_t length)
{
    __gm__ uint8_t* start = (__gm__ uint8_t*)((uint64_t)addr / ACLSHMEM_DATA_CACHE_LINE_SIZE
        * ACLSHMEM_DATA_CACHE_LINE_SIZE);
    __gm__ uint8_t* end = (__gm__ uint8_t*)(((uint64_t)addr + length) / ACLSHMEM_DATA_CACHE_LINE_SIZE
        * ACLSHMEM_DATA_CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint64_t i = 0; i <= end - start; i+= ACLSHMEM_DATA_CACHE_LINE_SIZE) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<uint8_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[i]);
        __asm__ __volatile__("");
    }
}

ACLSHMEM_DEVICE void dcci_entire_cache()
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    
    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}
```

Same file, line 35 — the cacheline constant:

```cpp
constexpr uint64_t ACLSHMEM_DATA_CACHE_LINE_SIZE = 64;
```

## A4. `dcci` before `GetValue` on a remote-written GM flag — the official poll loop

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/shmem_device_p2p_sync.hpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 18-110)

```cpp
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_eq(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr != cmp_val);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_ne(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr == cmp_val);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_gt(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr <= cmp_val);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_ge(__gm__ volatile T *sig_addr, T cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while (*sig_addr < cmp_val);
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_fetch(__gm__ int32_t *sig_addr)
{
    dcci_cacheline((__gm__ uint8_t *)sig_addr);
    return *sig_addr;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until(__gm__ volatile T *ivar, int cmp, T cmp_value)
{
    switch (cmp) {
        case ACLSHMEM_CMP_EQ:
            return aclshmemi_signal_wait_until_eq<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_NE:
            return aclshmemi_signal_wait_until_ne<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_GT:
            return aclshmemi_signal_wait_until_gt<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_GE:
            return aclshmemi_signal_wait_until_ge<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_LT:
            return aclshmemi_signal_wait_until_lt<T>(ivar, cmp_value);
        case ACLSHMEM_CMP_LE:
            return aclshmemi_signal_wait_until_le<T>(ivar, cmp_value);
    }
}
```

Note the two load-bearing details: `__gm__ volatile T *` on the pointer, and `dcci_cacheline`
*inside* the loop body (do/while, so it runs before the first compare too).

## A5. `aclshmemx_signal_op` and `aclshmem_signal_wait_until` — declared API, with the atomicity warning

https://raw.githubusercontent.com/immengzi/shmem/master/include/device/gm2gm/shmem_device_p2p_sync.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 23-51)

```c
/**
 * @brief The aclshmemx_signal_op operation updates sig_addr with signal using operation sig_op on the specified PE.
 *        This operation can be used together with aclshmem_signal_wait_until for efficient
 *        point-to-point synchronization.
 * WARNING: Atomicity NOT Guaranteed.
 *
 * @param sig_addr              [in] Symmetric address of the signal word to be updated.
 * @param signal                [in] The value used to update sig_addr.
 * @param sig_op                [in] Operation used to update sig_addr with signal. Supported operations:
 *                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                    [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);
#define shmemx_signal_op aclshmemx_signal_op

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                                   ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @return Return the contents of the signal data object, sig_addr, at the calling PE that satisfies the wait condition.
 */
ACLSHMEM_DEVICE int32_t aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val);
#define shmem_signal_wait_until aclshmem_signal_wait_until
```

https://raw.githubusercontent.com/immengzi/shmem/master/include/host_device/shmem_common_types.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 108-123)

```c
enum aclshmem_signal_op_type_t {
    ACLSHMEM_SIGNAL_SET,
    ACLSHMEM_SIGNAL_ADD
};

/**
 * @brief Signal comparison operations, used by the signalee in peer-to-peer synchronization
 */
enum aclshmem_cmp_op_type_t {
    ACLSHMEM_CMP_EQ = 0,
    ACLSHMEM_CMP_NE,
    ACLSHMEM_CMP_GT,
    ACLSHMEM_CMP_GE,
    ACLSHMEM_CMP_LT,
    ACLSHMEM_CMP_LE
};
```

## A6. `signal_op` implementation — `AMO_SET` is a plain store + dcci; `AMO_ADD` is `SetAtomicAdd` via UB→GM

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/shmem_device_so.hpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 255-342)

```cpp
ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int32_t val)
{
    aclshmemi_store(addr, val);

    // flush data cache to GM after signal to ensure it is visiable to other ranks
    dcci_cacheline((__gm__ uint8_t *)addr);
}

ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int pe, int32_t val)
{
    aclshmemi_signal_set((__gm__ int32_t *)aclshmem_ptr(addr, pe), val);
}

ACLSHMEM_DEVICE void aclshmemi_highlevel_signal_set(__gm__ int32_t *dst, __gm__ int32_t *src, int pe)
{
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR
                                                                        + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)aclshmem_roce_ptr(dst, pe), (__gm__ uint8_t*)src, pe, 0, sizeof(int32_t),
        ub_tensor_64, ub_tensor_32, 0);
    aclshmemi_roce_quiet(pe, 0, ub_tensor_64, ub_tensor_32, 0);
}

ACLSHMEM_DEVICE void aclshmemi_signal_add(__gm__ int32_t *addr, int pe, int32_t val)
{
    if (AscendC::GetBlockIdx() == 0){
        auto ptr = aclshmem_ptr(addr, pe);
        if (ptr == nullptr) return;
        AscendC::SetAtomicNone();
        __gm__ int32_t* remote_ptr = reinterpret_cast<__gm__ int32_t*>(ptr);

        __ubuf__ int32_t* buf =(__ubuf__ int32_t*)(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
        *buf = val;

        AscendC::SetAtomicAdd<int32_t>();
        aclshmemi_copy_ub2gm(remote_ptr, buf, sizeof(int32_t));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);

        AscendC::SetAtomicNone();
    }
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);

        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }

        // in case when peer pe enters next barrier
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);

    // never reach
    return -1;
}

ACLSHMEM_DEVICE void aclshmemi_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    switch (sig_op) {
        case ACLSHMEM_SIGNAL_SET:
            aclshmemi_signal_set(sig_addr, pe, signal);
            break;
        case ACLSHMEM_SIGNAL_ADD:
            aclshmemi_signal_add(sig_addr, pe, signal);
            break;
    }
}

ACLSHMEM_DEVICE void aclshmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    aclshmemi_signal_op(sig_addr, signal, sig_op, pe);
}
```

`aclshmemi_store` / `aclshmemi_load` are plain dereferences — no barrier of their own:

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/shmemi_device_common.hpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```cpp
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_store(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

template<typename T>
ACLSHMEM_DEVICE T aclshmemi_load(__gm__ T *cache)
{
    return *((__gm__ T *)cache);
}
```

## A7. The upstream message-passing litmus test — exactly "what orders a remote write against the local read"

https://raw.githubusercontent.com/immengzi/shmem/master/tests/unittest/device/sync/order/order_kernel.cpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```cpp
#include "kernel_operator.h"
#include "shmem.h"
#include "gm2gm/shmemi_device_cc.h"

extern "C" ACLSHMEM_GLOBAL void quiet_order(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    util_set_ffts_config(config);
    __gm__ uint64_t *base = reinterpret_cast<__gm__ uint64_t*>(addr);
    if (rank_id == 0) {
        aclshmemi_store<uint64_t>(base, 0xAA);
        aclshmem_quiet();
        aclshmemi_store<uint64_t>(base + 32, 0xBB);
        dcci_cacheline((__gm__ uint8_t *)(base + 32));
    }

    if (rank_id == 1) {
        uint64_t seen_b;
        __gm__ uint64_t *remote = (__gm__ uint64_t *)aclshmem_ptr(base, 0);
        do {
            dcci_cacheline((__gm__ uint8_t *)(remote + 32));
            seen_b = aclshmemi_load<uint64_t>(remote + 32);
        } while (seen_b != 0xBB);
        dcci_cacheline((__gm__ uint8_t *)(remote));
        uint64_t seen_a = aclshmemi_load<uint64_t>(remote);

        aclshmemi_store<uint64_t>(base + 33, seen_b);
        aclshmemi_store<uint64_t>(base + 34, seen_a);
    }
}

extern "C" ACLSHMEM_GLOBAL void fence_order(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    util_set_ffts_config(config);
    __gm__ uint64_t *base = reinterpret_cast<__gm__ uint64_t*>(addr);
    if (rank_id == 0) {
        uint64_t a_val = 42, b_val = 84;
        aclshmemi_store<uint64_t>(base, a_val);
        aclshmem_fence();
        aclshmemi_store<uint64_t>(base + 16, b_val);
        dcci_cacheline((__gm__ uint8_t *)(base + 16));
    }

    if (rank_id == 1) {
        uint64_t seen_b;
        __gm__ uint64_t *remote = (__gm__ uint64_t *)aclshmem_ptr(base, 0);
        do {
            dcci_cacheline((__gm__ uint8_t *)(remote + 16));
            seen_b = aclshmemi_load<uint64_t>(remote + 16);
        } while (seen_b != 84);
        dcci_cacheline((__gm__ uint8_t *)remote);
        uint64_t seen_a = aclshmemi_load<uint64_t>(remote);

        aclshmemi_store<uint64_t>(base + 17, seen_b);
        aclshmemi_store<uint64_t>(base + 18, seen_a);
    }
}

void quiet_order_do(void* stream, uint64_t config, uint8_t *addr, int32_t rank_id, int32_t n_ranks)
{
    quiet_order<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}

void fence_order_do(void* stream, uint64_t config, uint8_t *addr, int32_t rank_id, int32_t n_ranks)
{
    fence_order<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}
```

Note the asymmetry it encodes: the writer needs `quiet`/`fence` between data and flag *plus*
an explicit `dcci_cacheline` on the flag; the reader needs `dcci_cacheline` on the flag inside
the poll loop *and a second `dcci_cacheline` on the data* after the flag is observed. There is
no implied data invalidate riding on the flag observation.

## A8. Epoch / generation tagging — the upstream rationale, verbatim

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/shmemi_device_cc.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 11-17 and 101-180)

File-header contract:

```
/*
This file provides device-side collective synchronization implementations, ensuring that:
1. ALL VEC CORES of all ranks of a team reach a sychonization point before doing subsequent operations.
2. All operations of ALL VEC CORES of all ranks of the team before the synchronization point are visible
   to ALL VEC CORES of all ranks of the team after the synchronization point.

*/
```

The dissemination-barrier design note — this is the explicit statement that a monotone counter
replaces a flag, and the explicit statement that there is no cache coherence to lean on:

```
/* Level 2: barrier between devices (within a host)

Dissemination Barrier

1. Algorithm process

The algorithm process could be separated into multiple rounds.
In each round, every participating rank signals its succeeding rank and waits its preceding rank's signal.
The distance of a rank and its successor increases exponentially with the round.

An 8-rank example is shown below:

           round 1            round 2            round 3
  rank 0  --------→  rank 1  --------→  rank 3  --------→  rank 7
  rank 1  --------→  rank 2  --------→  rank 4  --------→  rank 0
  rank 2  --------→  rank 3  --------→  rank 5  --------→  rank 1
  rank 3  --------→  rank 4  --------→  rank 6  --------→  rank 2
  rank 4  --------→  rank 5  --------→  rank 7  --------→  rank 3
  rank 5  --------→  rank 6  --------→  rank 0  --------→  rank 4
  rank 6  --------→  rank 7  --------→  rank 1  --------→  rank 5
  rank 7  --------→  rank 0  --------→  rank 2  --------→  rank 6

Refer to https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf for more details.

2. Implementation details

Current implementation maintains an array of MAX_RANK_SIZE for each rank, with element of pos i indicating whether the
rank has received signal of rank i.
In each round, every rank writes remote array and check local array to decide whether this round has finished.
Once all rounds finished, barrier ends.

Theoretically, each element is writen by only 1 rank and read by self, involving only p2p synchronization.
However, separate elements may exist on the same cacheline, so that concurrent write
acctually happens and may cause wrong result.

For example:
a. rank n is waiting for rank n-1's signal (in round 1).
             ↑   n
--------------------------------------------
      ...  | 0 | 0 | ...
--------------------------------------------

b. rank n-1 reads rank n's array, and write the element at position n-1 (in round 1).
             ↓   n
--------------------------------------------
      ...  | 1 | 0 | ...
--------------------------------------------

c. rank n-2 reads staled rank n's array (no cache consistency ensurance), and write
   the element at position n-2 (in round 2).
         ↓       n
--------------------------------------------
   ... | 1 | 0 | 0 | ...
--------------------------------------------

d. rank n-2 overwrites rank n-1，so rank n may miss rank n-1's signal and wait forever.
             ↑   n
--------------------------------------------
   ... | 1 | 0 | 0 | ...
--------------------------------------------

To avoid this issue, separate elements must exist on different cachelines. See aclshmemi_sync_bit for detailed definition.

Additionly, instead of simply write a flag, each rank writes a 64-bit number into the array,
indicating how many times this team has performed barrier.

The temporal and spatial complexity of this implementation are O(logN) and O(N), respectively.

3. Futher development
  a. Hierarchical synchronization.
    Sync within the host first, then sync between host. May achieve better
    performance by utilizing low-latency in-host network better.

  b. Group dissemination.
    Group the ranks so that each rank could issue multiple signals and waits concurrently,
    instead of 1 signal and 1 wait as above.

  c. Optimize spatial complexity to O(logN).
*/
```

The cacheline-padded flag type and the persistent epoch counters it refers to:

https://raw.githubusercontent.com/immengzi/shmem/master/include/host_device/shmem_common_types.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 257-258, 387-395)

```c
/// \brief ACLSHMEM synchronization bit array type, cache line-aligned storage structure for synchronization flags
typedef int32_t aclshmemi_sync_bit[ACLSHMEMI_SYNCBIT_SIZE / sizeof(int32_t)];
```

```c
    // Using aclshmemi_sync_bit instead of basic types to aclshmemi_store flag,
    // avoiding concurrent write due to cacheline sharing.
    // Refer to shmemi_barrier.h for more details.
    // These members are 'shmemi_sync_bit *' types actually, but are defined as 'uint64_t' due to compiler restriction.
    uint64_t sync_pool;          ///< NPU-level sync pool pointer (actual type is aclshmemi_sync_bit*, defined as uint64_t due to compiler restrictions)
    uint64_t sync_counter;       ///< NPU-level sync counter pointer (actual type is aclshmemi_sync_bit*, defined as uint64_t due to compiler restrictions)
    uint64_t core_sync_pool;     ///< Core-level sync pool pointer (actual type is aclshmemi_sync_bit*, defined as uint64_t due to compiler restrictions)
    uint64_t core_sync_counter;  ///< Core-level sync counter pointer (actual type is aclshmemi_sync_bit*, defined as uint64_t due to compiler restrictions)
```

## A9. Epoch tagging in the barrier code itself — `count = load(counter) + 1`, no zeroing, ever

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/shmemi_device_cc.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 46-83, 285-332)

The intra-device (cross-AI-core) soft barrier, and the `SyncAll` fallback switch:

```cpp
ACLSHMEM_DEVICE void aclshmemi_barrier_core_soft()
{
    if ASCEND_IS_AIC {
        return;
    }

    int32_t cur_block_idx = AscendC::GetBlockIdx();
    int32_t block_dim = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    auto sync_pool = aclshmemi_get_core_sync_pool();
    auto sync_counter = aclshmemi_get_core_sync_counter();
    int32_t count = aclshmemi_load((__gm__ int32_t *)(sync_counter)) + 1;

    int32_t shift = 1;
    int32_t offset = 0;
    while (shift < block_dim) {
        int32_t next = (cur_block_idx + shift) % block_dim;

        aclshmemi_signal_set((__gm__ int32_t *)(sync_pool + next * ACLSHMEM_LOG_MAX_AIV_PER_NPU + offset), count);
        aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)(sync_pool +
            cur_block_idx * ACLSHMEM_LOG_MAX_AIV_PER_NPU + offset), count);

        shift *= SHIFT_MULTIPLIER;
        offset++;
    }

    aclshmemi_store((__gm__ int32_t *)(sync_counter), count);
}

/* Level 1: barrier between vec cores (within a device) */
template<bool IS_AIV_ONLY>
ACLSHMEM_DEVICE void aclshmemi_barrier_core()
{
#ifdef __CCE_AICORE_ENABLE_MIX__
    AscendC::SyncAll<IS_AIV_ONLY>();
#else
    aclshmemi_barrier_core_soft();
#endif
}
```

The cross-device barrier actually used (`aclshmem_barrier_all` → `aclshmemi_barrier` → `v3`),
a centralised *pull*-mode barrier — note it writes its own slot locally and polls the *remote*
slot of every peer, so only one remote write per rank per barrier:

```cpp
/** Centralized Barrier (pull mode).
 *
 *  The temporal and spatial complexity of this implementation are O(N/K) and O(1), respectively.
 *  Performs better than Group Dissemination Barrier at small scale (eg. 8 ranks).
 */
template<bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_barrier_npu_v3(aclshmem_team_t team_idx)
{
    aclshmemx_team_t *team = aclshmemi_get_state()->team_pools[team_idx];
    int32_t vec_id = AscendC::GetBlockIdx();
    int32_t vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();

    int32_t my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int32_t start = team->start;
    int32_t stride = team->stride;
    int32_t size = team->size;
    auto sync_pool = aclshmemi_get_team_sync_pool(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int32_t k = ACLSHMEM_BARRIER_TG_DISSEM_KVAL;
    k = k < size ? k : size;
    k = k < vec_size ? k : vec_size;
    int32_t my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t *)sync_counter) + 1;

    aclshmemi_barrier_core<IS_AIV_ONLY>();

    if ASCEND_IS_AIV {
        for (int32_t i = vec_id; i < size; i += k) {
            if (i == my_pe_in_team) {
                // write local
                aclshmemi_signal_set((__gm__ int32_t *)sync_pool, count);
            } else {
                // read remote
                int32_t remote_pe = start + i * stride;
                aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)aclshmem_ptr(sync_pool, remote_pe), count);
            }
        }
        aclshmemi_store((__gm__ int32_t *)sync_counter, count);
    }
    aclshmemi_barrier_core<IS_AIV_ONLY>();
}

template<bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void aclshmemi_barrier(aclshmem_team_t team)
{
    if (team == -1) {
        return; // not in team
    }
    aclshmemi_barrier_npu_v3<IS_AIV_ONLY>(team);
}
```

The `cmp_val || cmp_val + 1` acceptance in `aclshmemi_signal_wait_until_eq_for_barrier`
(quoted in A6, `// in case when peer pe enters next barrier`) is the epoch-skew tolerance that
lets a peer run one generation ahead without deadlocking the waiter.

## A10. Epoch tagging in a real one-sided collective kernel — the `magic` parameter

https://raw.githubusercontent.com/immengzi/shmem/master/examples/allgather/allgather_kernel.cpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

Kernel entry carries an `int magic` from host; the big-data path scales it by 1024 per
iteration so the high bits are the generation and the low 10 bits are the chunk counter:

```cpp
// all_gather
template <typename T>
ACLSHMEM_DEVICE void all_gather_big_data(uint64_t fftsAddr, __gm__ T *input, __gm__ T *output, __gm__ T *gva, int elements,
                                      int magic)
{
    util_set_ffts_config(fftsAddr);

    const int64_t max_gva_num = GVA_BUFF_MAX_SIZE / sizeof(T);
    int times = (elements + max_gva_num - 1) / max_gva_num;
    int total_num = elements;

    __ubuf__ int64_t *ctrl_ub = (__ubuf__ int64_t *)(0);
    for (int i = 0; i < times; i++) {
        *ctrl_ub = 0;
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t len = total_num > max_gva_num ? max_gva_num : total_num;
        aclshmemx_barrier_all_vec();
        all_gather_origin(input + i * max_gva_num, output + i * max_gva_num, gva, max_gva_num, elements, len,
                          (magic + i) * 1024);
        total_num -= max_gva_num;
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
```

Producer side — `put_nbi`, then an MTE3→S hard-event pair, then the epoch-tagged flag
`flag = times + magic`:

```cpp
        int64_t times = 0;
        int64_t flag = 0;
        while (copy_total_size >= copy_ub_size) {
            SHMEMI_PROF_START(0);
            aclshmemx_mte_put_nbi(gva_data_gm + aivIndex * len_per_core + times * copy_ub_num,
                                  input_gm + aivIndex * len_per_core + times * copy_ub_num, tmp_buff, copy_ub_size,
                                  copy_ub_num, my_rank, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            times += 1;
            flag = times + magic;
            aclshmemx_signal_op(gva_sync_gm + flag_offset, flag, ACLSHMEM_SIGNAL_SET, my_rank);
            SHMEMI_PROF_END(0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            copy_total_size -= copy_ub_size;
        }
```

Consumer side — this is the generation test. `*flags_ub2 >> 10` vs `magic >> 10` rejects a word
left over from a previous launch; `ready_num = *flags_ub2 - magic` recovers how many chunks of
*this* generation have landed. No flag buffer is ever cleared:

```cpp
            aclshmem_int32_get_nbi(flags_ub2[group_idx], gva_sync_gm + group_idx * SYNC_FLAG_INTERVAL, 1, x);
            AscendC::PipeBarrier<PIPE_ALL>();

            if ((*flags_ub2[group_idx] >> 10) != (magic >> 10)) {
                continue;
            }

            int64_t ready_num = *flags_ub2[group_idx] - magic;
            if (ready_num <= 0 || *flags_ub1[group_idx] >= ready_num) {
                continue;
            }
```

Small-data (<2 MiB) path — the exact "no host barrier" prologue/epilogue, with `quiet` then an
*intra-device* soft barrier, then signal, then remote-slot wait:

```cpp
    aclshmemx_mte_put_nbi(gva_data_gm + gva_offset, input_gm + input_offset, tmp_buff, ub_size, num_per_core, my_rank,
                          EVENT_ID0);

    const int64_t core_per_rank = aivNum / pe_size;
    const int64_t core_rank_idx = aivIndex % core_per_rank;
    const int64_t x = aivIndex / core_per_rank;

    // Sync Ensure Corresponding Tasks Done.
    aclshmem_quiet();
    aclshmemi_barrier_core_soft();

    aclshmemx_signal_op(gva_sync_gm + flag_offset, magic, ACLSHMEM_SIGNAL_SET, my_rank);
    aclshmem_signal_wait_until((__gm__ int32_t *)aclshmem_ptr(gva_sync_gm, x) + flag_offset, ACLSHMEM_CMP_EQ, magic);

    // [AllGather Step 2] symmetric mem -> local output.
    num_per_core = elements / core_per_rank;
    output_offset = x * elements + core_rank_idx * num_per_core;
    gva_offset = core_rank_idx * num_per_core;
    if (core_rank_idx == core_per_rank - 1) {
        num_per_core = elements - num_per_core * core_rank_idx;
    }
    aclshmemx_mte_get_nbi(output_gm + output_offset, gva_data_gm + gva_offset, tmp_buff, ub_size, num_per_core, x,
                          EVENT_ID0);
```

Flag-slot stride constant (16 × int32 = 64 B = one cacheline), i.e. one cacheline per AIV:

```cpp
constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
```

## A11. `aclshmemx_mte_put_nbi` / `aclshmemx_mte_get_nbi` / `aclshmem_ptr` — all declared overloads

https://raw.githubusercontent.com/immengzi/shmem/master/include/device/gm2gm/engine/shmem_device_mte.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```c
/**
 * @brief Translate an local symmetric address to remote symmetric address on the specified PE.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return A remote symmetric address on the specified PE that can be accessed using memory loads and stores.
 */
ACLSHMEM_DEVICE __gm__ void *aclshmem_ptr(__gm__ void *ptr, int pe);
#define shmem_ptr aclshmem_ptr
```

```c
/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                           uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                           const non_contiguous_copy_param &copy_params, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                           AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                           AscendC::LocalTensor<T> buf, const non_contiguous_copy_param &copy_params,
                                           int pe, uint32_t sync_id);
#define shmem_mte_put_mem_nbi aclshmemx_mte_put_nbi
```

https://raw.githubusercontent.com/immengzi/shmem/master/include/device/ub2gm/engine/shmem_device_mte.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 — the UB→GM (no staging
buffer) variants, which are the right ones for a fused megakernel since the data is already in UB:

```c
/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, uint32_t elem_size, int pe,
                                           uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] LocalTensor on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src,
                                           uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, const non_contiguous_copy_param &copy_params,
                                           int pe, uint32_t sync_id);
```

Supporting constants / the non-contiguous descriptor:

https://raw.githubusercontent.com/immengzi/shmem/master/include/device/shmem_def.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```c
constexpr uint64_t ACLSHMEM_INTERNAL_UB_BUF_START_ADDR = 188 * 1024;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t ACLSHMEM_NUM_CQE_PER_POLL_CQ = 100;

/**
 * @struct non_contiguous_copy_param
 * @brief Non-Contiguous Datacopy Param.
 *
 * - uint32_t repeat: Data move times
 * - uint32_t length: Data move unit length
 * - uint32_t src_ld: Src data leading dimension. Interval between the head of the repeat and the
 *   head of the following repeat.
 * - uint32_t dst_ld: Dst data leading dimension.
*/
struct non_contiguous_copy_param {
    uint32_t repeat;
    uint32_t length;
    uint32_t src_ld;
    uint32_t dst_ld;
};
```

## A12. `aclshmem_barrier_all` — host and device declarations

https://raw.githubusercontent.com/immengzi/shmem/master/include/device/gm2gm/shmem_device_cc.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

The file opens with a restrictions block that is directly load-bearing for a fused megakernel —
note item 2, which forbids mixing these barriers with `AscendC::SyncAll` (Topic B):

```c
/*
    WARNING：Restrictions of Barrier APIs.

    1. Barrier APIs can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if
       it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy).
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.

    2. Barrier APIs conflict with SyncAll. Avoid mixing them together.

    3. We provide 2 kinds of barrier:
        a. aclshmem_barrier_xxx
            Barrier of all cores. On systems with only HCCS: All operations of all pes of a team on excuting
            stream before the barrier are visiable to all pes of the team after the barrier.
        b. aclshmemx_barrier_xxx_vec
            Barrier of all VEC cores. On systems with only HCCS: All operations of ALL VEC CORES of all pes of a
            team on excuting stream before the barrier are visiable to ALL VEC CORES of all pes of the team after
            the barrier.

    4. The scalar unit of cube core is not affected by aclshmem_barrier_xxx. Make sure don't use that.
*/
```

```c
/**
 * @fn ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config)
 * @brief Set runtime ffts address. Call this at MIX Kernel entry point (if the kernel contains barrier calls).
 *
 * @param config              [config] ffts config, acquired by util_get_ffts_config()
 */
ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config);
#define shmemx_set_ffts_config util_set_ffts_config

/**
 * @brief aclshmem_barrier is a collective synchronization routine over a team. Control returns from aclshmem_barrier
 *        after all PEs in the team have called aclshmem_barrier.
 *        aclshmem_barrier ensures that all previously issued stores and remote memory updates, including AMOs and
 *        RMA operations, done by any of the PEs in the active set are complete before returning. On systems with
 *        only scale-up network (HCCS), updates are globally visible, whereas on systems with both scale-up network
 *        HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the memory of a given PE are
 *        visible to that PE.
 *        Barrier operations issued on the CPU and the NPU only complete communication operations that were issued
 *        from the CPU and the NPU, respectively. To ensure completion of GPU-side operations from the CPU, using
 *        aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 * @param team              [in] team to do barrier
 */
ACLSHMEM_DEVICE void aclshmem_barrier(aclshmem_team_t team);
#define shmem_barrier aclshmem_barrier

/**
 * @brief aclshmem_barrier of all PEs.
 */
ACLSHMEM_DEVICE void aclshmem_barrier_all(void);
#define shmem_barrier_all aclshmem_barrier_all

/**
 * @brief Similar to aclshmem_barrier except that only vector cores participate. Useful in communication-over-compute
 *        operators. Cube core may call the api but takes no effect.
 *
 * @param team              [in] team to do barrier
 */
[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmem_barrier instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t team);
#define shmemx_barrier_vec aclshmemx_barrier_vec

/**
 * @brief aclshmemx_barrier_vec of all PEs.
 *
 */
[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmem_barrier_all instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec(void);
#define shmemx_barrier_all_vec aclshmemx_barrier_all_vec

/**
 * @brief Similar to aclshmem_barrier. In constract with the aclshmem_barrier routine, aclshmem_sync only ensures
 *        completion and visibility of previously issued memory stores and does not ensure completion of remote memory
 *        updates issued via ACLSHMEM rountines.
 *
 * @param team           [in] team to do barrier
 */
ACLSHMEM_DEVICE void aclshmem_sync(aclshmem_team_t team);

/**
 * @brief aclshmem_sync_all of all PEs.
 *
 */
ACLSHMEM_DEVICE void aclshmem_sync_all(void);
```

## A13. Symmetric heap allocation — `aclshmem_malloc` and the `aclshmemx_*` mem_type variants

https://raw.githubusercontent.com/immengzi/shmem/master/include/host/mem/shmem_host_heap.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```c
/**
 * @brief allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory is not initialized.
 *        If <i>size</i> is 0, then <b>aclshmem_malloc()</b> returns NULL.
 *
 * @param size             [in] bytes to be allocated
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void *aclshmem_malloc(size_t size);
#define shmem_malloc aclshmem_malloc


/**
 * @brief allocate memory for an array of <i>nmemb</i> elements of <i>size</i> bytes each and returns a pointer to the
 *        allocated memory. The memory is set to zero. If <i>nmemb</i> or <i>size</i> is 0, then <b>aclshmem_calloc()</b>
 *        returns either NULL.
 *
 * @param nmemb            [in] elements count
 * @param size             [in] each element size in bytes
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void *aclshmem_calloc(size_t nmemb, size_t size);
#define shmem_calloc aclshmem_calloc


/**
 * @brief allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory address will be a
 *        multiple of <i>alignment</i>, which must be a power of two.
 *
 * @param alignment        [in] memory address alignment
 * @param size             [in] bytes allocated
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void *aclshmem_align(size_t alignment, size_t size);
#define shmem_align aclshmem_align

/**
 * @brief Allocates a block of ACLSHMEM symmetric memory, the data in this memory is uninitialized
 *
 * @param size          [in] Memory allocation size (in bytes)
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the symmetric memory
 *
 */
ACLSHMEM_HOST_API void *aclshmemx_malloc(size_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Allocates a block of SHMEM symmetric memory and initializes all content to zero
 *
 * @param count         [in] Number of elements
 * @param size          [in] Number of bytes occupied by each element
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the symmetric memory
 */
ACLSHMEM_HOST_API void *aclshmemx_calloc(size_t count, size_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);
```

## A14. Signal/wait family beyond the single word — all internal entry points

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/shmemi_device_p2p_sync.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```cpp
ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int32_t val);

ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int pe, int32_t val);

ACLSHMEM_DEVICE void aclshmemi_highlevel_signal_set(__gm__ int32_t *dst, __gm__ int32_t *src, int pe);

ACLSHMEM_DEVICE void aclshmemi_signal_add(__gm__ int32_t *addr, int pe, int32_t val);

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val);

// Atomicity of ACLSHMEM_SIGNAL_SET not guaranteed
ACLSHMEM_DEVICE void aclshmemi_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_signal_wait_until_eq(__gm__ volatile T *sig_addr, T cmp_val);

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test(__gm__ volatile T *ivar, int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until_all(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_any(__gm__ volatile T *ivars, size_t nelems, __gm__ const int *status,
    int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE size_t aclshmemi_wait_until_some(__gm__ volatile T *ivars, size_t nelems, __gm__ size_t *indices,
    __gm__ const int *status, int cmp, T cmp_value);

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test_all(__gm__ T *sig_addr, size_t nelems, __gm__ const int *status, int cmp,
    T cmp_value, int stride = ACLSHMEMI_SYNCBIT_SIZE / sizeof(T));
```

## A15. Device-side atomic-add on symmetric memory (`aclshmem_<type>_atomic_add`)

https://raw.githubusercontent.com/immengzi/shmem/master/include/device/gm2gm/shmem_device_amo.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```c
/**
 * @brief Standard Atomic Add Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |half       | half      |
 * |bfloat16   | bfloat16  |
 * |float      | float     |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(FUNC)  \
    FUNC(int8, int8_t);           \
    FUNC(int16, int16_t);        \
    FUNC(int32, int32_t);        \
    FUNC(half, half);            \
    FUNC(bfloat16, bfloat16_t); \
    FUNC(float, float)

/**
 * @brief  Automatically generates aclshmem atomic add functions for different data types (e.g., int8, int16, int32, float, half, bfloat16).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 * 
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_add(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Perform contiguous data atomic add operation on
 * symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Value atomic add to destination.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE)                                                      \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(ACLSHMEM_ATOMIC_ADD_TYPENAME);
/** \endcond */
#define shmem_int8_atomic_add aclshmem_int8_atomic_add
#define shmem_int16_atomic_add aclshmem_int16_atomic_add
#define shmem_int32_atomic_add aclshmem_int32_atomic_add
#define shmem_half_atomic_add aclshmem_half_atomic_add
#define shmem_bfloat16_atomic_add aclshmem_bfloat16_atomic_add
#define shmem_float_atomic_add aclshmem_float_atomic_add
```

## A16. Symmetric-heap address arithmetic and the `ffts` precondition (Chinese, verbatim)

https://raw.githubusercontent.com/immengzi/shmem/master/docs/principles.md
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

Why `aclshmem_ptr(p, pe)` is pure address arithmetic — and the symmetry precondition that
makes a hand-rolled `tp_push` legal:

```
该部分会分配出两块内存：
* 用户申请的共享内存空间，每个rank申请的内存大小和初始化传入attributes的local_mem_size大小一致，后续可以通过`aclshmem_malloc`、`aclshmem_free`等共享内存接口管理。
* 用于在device保存shmem的state、team等元数据空间，当state和team等信息发生变化，进程将自动同步到元数据空间，元数据空间32M，不对外提供访问接口
```

```
* `aclshmem_malloc`内部采用first fit找到第一个满足要求的chunk，从该chunk中分离出对应大小的chunk，剩余空间（如果有）则作为新的空闲chunk。
* `aclshmem_free`释放后补充空闲chunk 。若新添加chunk与其他空闲chunk连续，则合并。
```

```
<p style="text-indent: 2em;"><strong>aclshmem_malloc和aclshmem_free接口使用时需要在所有进程同步调用，且需要分配或释放相同大小的内存。</strong>SHMEM内部默认aclshmem_malloc分配的内存在虚拟地址空间对称，即<strong>当前rank malloc的地址+heap_size=下一个rank malloc的地址</strong>。如果进程间分配内存不同会导致后续分配的内存首地址不对称，无法通过地址偏移访问到正确的数据。</p>
```

Where the epoch counters live, and the explicit split between inter-card and inter-core pools:

```
<p style="text-indent: 2em;">在device上分配并初始化后续team级别同步需要的资源sync_pool、sync_counter、core_sync_pool、core_sync_counter。前俩个成员用于team内所有的卡间同步、后两个成员用于卡内核间同步。</p>


**team_pool、sync_pool、sync_counter、core_sync_pool、core_sync_counter这些信息都会在state里储存。**
```

The `ffts` config — this is the hard dependency that couples shmem barriers to AscendC
`SyncAll` / `CrossCoreSetFlag` / `CrossCoreWaitFlag` (directly relevant to TOPIC B):

```
<p style="text-indent: 2em;">该部分仅获取一个ffts地址，后续可通过<code>shmemx_get_ffts_config</code>获得，在算子内通过<code>shmemx_set_ffts_config</code>设置，用于runtime同步，AscendC层面会影响的<code>SyncAll</code>、<code>CrossCoreSetFlag</code>和<code>CrossCoreWaitFlag</code>接口，shmem层面会影响到barrier相关接口。（</ode>shmemx_set_ffts_config</code>开销较小，建议算子内都调用一次该接口）。</p>
```

Host-side bootstrap barrier is TCP/socket based and is explicitly *not* `aclshmem_barrier`:

```
<p style="text-indent: 2em;">多进程间的建链是基于TCP socket实现的。所有进程会先基于uid内部的ip::port信息和rank0建立连接，连接时会根据uid内magic信息是否一致判断是否为同一通信域，不同则断开连接。连接成功后所有rank与rank0通信，基于此实现host侧的allgather和barrier等能力。（<strong>该部分能力并非aclshmem_barrier相关能力，而是类似于MPI_Barrier</strong>）</p>
```

## A17. Flag-free collectives in the official DeepSeek-V4 recipe — signal-with-ADD + WaitUntil, and the float "sum of flags" idiom

https://raw.githubusercontent.com/hicann/cann-recipes-infer/master/docs/models/deepseek_v4/deepseek_v4_shmem_guide.md
hicann / CANN recipes-infer, `master` (fetched 2026-10-20)

pyPTO-SHMEM API surface, including the non-standard clear-signal helpers that exist precisely
because a non-epoch-tagged flag must be zeroed between generations:

```
#### pyPTO-shmem API介绍

PyPTO中实现了 OPENSHMEM 部分通信语义，并且封装为API，供用户调用：

* **shmem_put**：将本地数据（local tensor）发送到对称内存（shmem tensor）中指定位置
* **shmem_get**：从对称内存（shmem tensor）中指定位置获取数据存放到本地（local tensor）
* **shmem_signal**：给其他卡发送信号
* **shmem_wait**：等待本卡信号，指定signal等于指定值
* **shmem_barrier_all**：通信域内卡间同步，当所有卡都到达此同步点才会解除继续执行

此外也实现了部分非标准 OPENSHMEM 语义：

* **create_shmem_tensor**：在指定通信域内创建表示对称内存的Tensor
* **shmem_clear_data**：将shmem tensor的数据清零
* **shmem_signal_data**：将shmem signal清零
* **my_symbolic_pe**：本卡符号化的rank id
```

The dispatch graph — `ShmemPut` of data, then `ShmemSignal(..., AtomicType::ADD)`, then
`WaitUntil` on the local signal tile. This is `putmem_signal` + `signal_wait_until` with
`AMO_ADD`, i.e. the count-based (not value-based) completion test:

```
LOOP("send_signal", FunctionType::DYNAMIC_LOOP, expertId, LoopRange(moeExpertNum)) {
    TileShape::Current().SetVecTile({1, 1});
    SymbolicScalar remoteRankId = expertId / expertNumPerRankScalar;
    SymbolicScalar remoteExpertOffset = expertId % expertNumPerRankScalar;
    Tensor shmemCountTile = View(shmemCount, {1, 1, 1, 1}, {remoteRankId, 0, remoteExpertOffset * epWorldSize + thisRank + 1, 0});
    Tensor totalOffsetTile = View(totalOffset, {1, 1}, {expertId, 0});
    Tensor shmemPutOut = ShmemPut(shmemSendOut, totalOffsetTile, shmemCountTile);
    TileShape::Current().SetVecTile({1, signalCol});
    Tensor shmemSignalTile = View(shmemSignal, {1, 1, 1, 1, signalCol}, {remoteRankId, 0, 0, 0, 0});
    shmemCountOut = ShmemSignal(shmemPutOut, shmemSignalTile, AtomicType::ADD);
}

LOOP("receive_token", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
    (void) i;
    TileShape::Current().SetVecTile({1, signalCol});
    Tensor shmemSignalLocalTile = View(shmemSignal, {1, 1, 1, 1, signalCol}, {thisRank, 0, 0, 0, 0});
    Tensor waitUntilOut = WaitUntil(shmemCountOut, shmemSignalLocalTile, moeExpertNum, true);
```

The TileLang dispatch kernel — note `T.sync_all()` in the Cube scope and the status loop that
waits on a *float sum* of status words rather than on individual flags:

```
        with T.Kernel(ai_core_num, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub([H + (32 + 12) // 2], "bfloat16")
            ...
            with T.Scope("C"):
                T.sync_all()
                ...
            with T.Scope("V"):
                send_token_num = cur_send_token_cnt // aiv_num
                remainder_token_num = cur_send_token_cnt % aiv_num
                start_send_token_id = send_token_num * true_vid
                ...
                # 发数据
                for cur_send_token_id in range(start_send_token_id, start_send_token_id + send_token_num):
                    ...
                    T.shmem_ub_put_nbi_new(x_ub, win_data, ub_size, dest_rank_id, (rank * Bs * local_expert_num + dest_expert_id * Bs + token_repeat_num[0]) * ub_size)
                # 发状态
                for cur_expert_id in range(start_expert_id, start_expert_id + aiv_expert_num):
                    ...
                    T.shmem_ub_put_nbi_new(win_status_ub_single, win_status, 8, dest_rank_id, local_expert_id * ep_world_size * 8 + rank * 8)
                # 循环等状态
                while (sum_of_flag[0] != aiv_expert_fp_num):
                    ...
                # 本地win区数据拷贝
                for i in range(status_num_per_core):
                    ...
```

The combine kernel's range-bounded sum-of-flags test — tolerant to partial arrival, which is
the dual of the epoch tag (`targetMin`/`targetMax` window instead of exact equality):

```
            with T.Scope("V"):
                T.tile.fill(statusUb, 1.0)
                ...
                # 发数据、发状态
                for loop in range(sendTokenNum[0]):
                    ...
                    T.shmem_ub_put_nbi_new(x_ub, winData, H, toRankId, winGM * H)

                    T.shmem_ub_put_nbi_new(statusUb, winStatus, 8, toRankId, winGM * 8)
                for curIdx in range(startTokenId[0], startTokenId[0] + tokenNum[0]):
                    sumOfFlag[0] = -1.0
                    stateGM = curIdx * K
                    calCnt = compareTarget
                    # 等状态
                    while((sumOfFlag[0] < targetMin) or (sumOfFlag[0] > targetMax)):
                        ...
                    # 本地win区数据拷贝
                    T.copy(winData[tokenIndexOffset, 0], winData_ub_bfloat)
                    T.copy(winData_ub_bfloat, combineOut[curIdx,0])
```

Tilelang `sync_all` is documented there as the inter-core barrier primitive:

```
* **T.alloc_ub**：申请ub上内存空间，使用样例为：T.alloc_ub(shape, dtype)
* **T.sync_all()**：核间同步
* **T.shmem_ub_put_nbi*_new***：Tilelang中封装的shmem接口，用于将本卡ub数据传送到Win区，使用样例为：T.shmem_ub_put_nbi_new(x_ub, win, ub_size, dest_rank_id, stride)，其中x_ub为本卡ub数据，win为申请得到的win区，ub_size为ub大小，dest_rank_id为对端卡id，stride为偏移量。
```

## A18. What a one-sided `put` actually is on A3 — address translation plus an ordinary `DataCopyPad`

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/engine/shmem_device_mte.hpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 (lines 30-45)

`aclshmem_ptr` is pure offset arithmetic against a per-PE mapped heap base. There is no
doorbell, no descriptor, no completion queue in this path:

```cpp
ACLSHMEM_DEVICE __gm__ void *aclshmem_ptr(__gm__ void *ptr, int pe)
{
    // Get Global State
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    if (is_host_mem_heap(ptr)) {
        uint64_t host_heap_offset =
            reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->host_heap_base);
        uint64_t remote_host_ptr = reinterpret_cast<uint64_t>(device_state->p2p_host_heap_base[pe]) + host_heap_offset;
        return reinterpret_cast<__gm__ void *>(remote_host_ptr);
    }
    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->heap_base);
    // Address translate
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->p2p_device_heap_base[pe]) + offset;
    return reinterpret_cast<__gm__ void *>(remote_ptr);
}
```

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/ub2gm/mte/shmem_device_mte.hpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

The UB→remote-GM put, in full. Note it is a bare `DataCopyPad` on an MTE3 pipe — which is why
the producer in A10 needs `SetFlag/WaitFlag<HardEvent::MTE3_S>` before writing the flag with a
scalar store:

```cpp
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __ubuf__ T *src, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(dst, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src);
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit)
        return;

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    aclshmemi_copy_ub2gm(remote_ptr, src, elem_size * sizeof(T));
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t elem_size,
                                           int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    aclshmemi_copy_ub2gm(remote_buff, src, elem_size * sizeof(T));
}
```

And the `copy_ub2gm` primitive it lands on — including the `SetL2CacheHint` /
`CACHE_MODE_DISABLE` lever, which is the one place the library lets you bypass L2 on a remote
write:

```cpp
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_ub2gm(__gm__ T* dstGva, __ubuf__ T* srcUb,
    uint32_t size, bool toL2Cache)
{
    ASCENDC_ASSERT((dstGva != nullptr), "input gva is null");

    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(srcUb);
    ubTensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGva));
    if (!toL2Cache) {
        gmTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    AscendC::DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_copy_gm2ub(__ubuf__ T* dstUb, __gm__ T* srcGva,
    uint32_t size, bool toL2Cache)
{
    ASCENDC_ASSERT((srcGva != nullptr), "input gva is null");
    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dstUb);
    ubTensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGva));
    if (!toL2Cache) {
        gmTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}
```

---

# TOPIC A, addendum — the official fused Matmul+AllReduce epilogue (the shape you are building)

## A19. `EpilogueAllReduce` — TP all-reduce fused into the GEMM epilogue, three in-kernel barriers, atomic-add reduce-scatter by remote *pull*

https://raw.githubusercontent.com/immengzi/shmem/master/examples/matmul_allreduce/epilogue/block/epilogue_allreduce.hpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04 — complete `operator()`

The structure to read off it: a double-buffered symmetric window indexed by
`flagIdx = calIdx % BufferNum`; `SetAtomicAdd<ElementC>()` armed around a `mte_get_nbi` loop so
the reduce-scatter is a *remote read that accumulates locally* (not a remote write); one AIV per
AI core does the comms (`aivIndex == 0`); and three `aclshmemx_barrier_all_vec()` calls —
before RS, between RS and AG, and after AG.

Class declaration and params:

```cpp
namespace Catlass::Epilogue::Block {
template <class... Args>
class EpilogueAllReduce {};

template <class BlockScheduler_, class CommBlockSwizzle_>
class EpilogueAllReduce<BlockScheduler_, CommBlockSwizzle_> {
public:
    using BlockScheduler = BlockScheduler_;
    using CommBlockSwizzle = CommBlockSwizzle_;

    // Type aliases
    using ArchTag = Arch::AtlasA2;

    using ElementC = half;
    using ElementAttachedSource = half;
    using ElementAttachedOutput = half;

    using LayoutStore = layout::RowMajor;

    using LayoutWorkspace = LayoutStore;

    using ElementDestination = ElementAttachedOutput;

    using LayoutDestination = LayoutStore;

    using ScheduleTypeOp1 = typename Gemm::Block::ReduceScatterSchedule;
    using ScheduleTypeOp2 = typename Gemm::Block::AllGatherSchedule;

    // Epilogue params definition
    struct Params {
        AscendC::GlobalTensor<ElementDestination> destination;
        LayoutDestination layoutDestination;
        int64_t strideDestination;
        __gm__ ElementAttachedSource *symmetricPtr;
        BlockScheduler gemmSwizzle;
        CommBlockSwizzle commSwizzle;
        MatrixCoord blockShape;
        MatrixCoord processShape;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(AscendC::GlobalTensor<ElementDestination> destination, const LayoutDestination &layoutDestination,
               int64_t strideDestination, __gm__ ElementAttachedSource *symmetricPtr, MatrixCoord blockShape,
               MatrixCoord processShape, BlockScheduler gemmSwizzle, CommBlockSwizzle commSwizzle)
            : destination(destination),
              layoutDestination(layoutDestination),
              strideDestination(strideDestination),
              symmetricPtr(symmetricPtr),
              blockShape(blockShape),
              processShape(processShape),
              gemmSwizzle(gemmSwizzle),
              commSwizzle(commSwizzle)
        {
        }
    };
```

Body:

```cpp
    CATLASS_DEVICE
    void operator()(MatrixCoord const &blockShape, MatrixCoord const &commBlockCount,
                    MatrixCoord const &actualCommBlockCount, uint32_t calIdx, uint32_t rankIdx, uint32_t rankSize,
                    uint32_t pValue)
    {
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        int32_t aivIndex = AscendC::GetSubBlockIdx();
        auto loopNumPerComm = aicoreNum * pValue;

        auto layoutPeerMemStore =
            layout::RowMajor(blockShape.row() * loopNumPerComm * BufferNum, blockShape.column(), blockShape.column());

        // re-sliced Block by comm cores
        uint32_t flagIdx = calIdx % BufferNum;
        MatrixCoord actualCommBlockShape = blockShape * actualCommBlockCount;
        MatrixCoord outputBlockOffset = blockShape * MatrixCoord{calIdx * loopNumPerComm, 0};

        MatrixCoord commCoordPeerMem{flagIdx, 0};
        MatrixCoord blockOffset = commCoordPeerMem * commBlockCount * blockShape;

        params.commSwizzle.template SetProblemSize<ScheduleTypeOp1, true>(actualCommBlockShape);
        auto realAicoreNum = params.commSwizzle.GetRealCore();
        auto commCoreLoops = params.commSwizzle.GetCoreLoop();

        AscendC::GlobalTensor<ElementC> peerMem;
        peerMem.SetGlobalBuffer(params.symmetricPtr);

        // Local matmul is completed, waiting until tasks on all devices are complete.
        aclshmemx_barrier_all_vec();

        AscendC::SetAtomicAdd<ElementC>();
        AscendC::PipeBarrier<PIPE_ALL>();

        if (aivIndex == 0 && aicoreIndex < realAicoreNum) {  // only use one AIV core
            for (uint32_t idx = aicoreIndex; idx < commCoreLoops; idx += realAicoreNum) {
                MatrixCoord idxTile = params.commSwizzle.GetBlockIdx(idx);
                MatrixCoord actualCommSubBlockShape =
                    params.commSwizzle.template GetBlockSize<ScheduleTypeOp1>(idxTile);
                MatrixCoord rankBlockOffset = params.commSwizzle.template GetRankOffset<ScheduleTypeOp1>(idxTile);
                MatrixCoord subBlockOffset = params.commSwizzle.GetBlockOffset(idxTile);
                uint32_t mRankIdx = idxTile.column();
                if (mRankIdx == rankIdx) {
                    continue;
                }

                auto offsetIn = blockOffset + rankBlockOffset;
                auto offsetOut = blockOffset + rankBlockOffset;

                auto residueProcessShape = actualCommSubBlockShape % params.processShape;
                MatrixCoord processCount = CeilDiv(actualCommSubBlockShape, params.processShape);
                uint32_t processLoop = processCount.row() * processCount.column();

                // [ReduceScatter] 1. Alloc TmpUB
                int tmpBufferSize = 32 * 1024 / sizeof(ElementC);  // 32 KB
                AscendC::LocalTensor<half> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<ElementC>(0);
                tmpBuffer1.SetSize(tmpBufferSize);
                int tmpBufferOffset = 96 * 1024;  // half of UB
                AscendC::LocalTensor<half> tmpBuffer2 =
                    resource.ubBuf.template GetBufferByByte<ElementC>(tmpBufferOffset);
                tmpBuffer2.SetSize(tmpBufferSize);

                // [ReduceScatter] 2. Pre Interface Sync
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                int pingpongId = 0;

                for (uint32_t processIndex = 0; processIndex < processLoop; ++processIndex) {
                    AscendC::TEventID event_id = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<ElementC> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;

                    MatrixCoord processCoord{processIndex / processCount.column(),
                                             processIndex % processCount.column()};
                    auto actualProcessShape =
                        GetActualShape(processCount, processCoord, params.processShape, residueProcessShape);

                    auto processOffset = processCoord * params.processShape;

                    auto inputOffset = offsetIn + subBlockOffset + processOffset;
                    auto outputOffset = offsetOut + subBlockOffset + processOffset;

                    uint32_t copySize = actualProcessShape.row() * actualProcessShape.column();

                    int64_t inputElemOffset = layoutPeerMemStore.GetOffset(inputOffset);
                    int64_t outputElemOffset = layoutPeerMemStore.GetOffset(outputOffset);

                    // [ReduceScatter] 2. Pre Interface Sync
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

                    // [ReduceScatter] 3. Start aclshmemx_mte_get_nbi
                    aclshmemx_mte_get_nbi(peerMem[outputElemOffset], peerMem[inputElemOffset], buf, copySize,
                                          mRankIdx % rankSize, event_id);

                    // [ReduceScatter] 4. Post Interface Sync
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                    pingpongId = (pingpongId + 1) % BufferNum;
                }
                // [ReduceScatter] 4. Post Interface Sync
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);  // To SetAtomic, Scalar wait MTE3
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();

        // ReduceScatter is completed, waiting until tasks on all devices are complete.
        aclshmemx_barrier_all_vec();

        if (aivIndex == 0 && aicoreIndex < realAicoreNum) {
            for (uint32_t idx = aicoreIndex; idx < commCoreLoops; idx += realAicoreNum) {
                MatrixCoord idxTile = params.commSwizzle.GetBlockIdx(idx);
                MatrixCoord actualCommSubBlockShape =
                    params.commSwizzle.template GetBlockSize<ScheduleTypeOp2>(idxTile);
                MatrixCoord rankBlockOffset = params.commSwizzle.template GetRankOffset<ScheduleTypeOp2>(idxTile);
                MatrixCoord subBlockOffset = params.commSwizzle.GetBlockOffset(idxTile);

                uint32_t mRankIdx = idxTile.column();

                auto offsetIn = blockOffset + rankBlockOffset;
                auto offsetOut = outputBlockOffset + rankBlockOffset;

                auto residueProcessShape = actualCommSubBlockShape % params.processShape;
                MatrixCoord processCount = CeilDiv(actualCommSubBlockShape, params.processShape);

                uint32_t processLoop = processCount.row() * processCount.column();

                // [AllGather] 1. Alloc TmpUB
                int tmpBufferSize = 32 * 1024 / sizeof(ElementC);  // 32 KB
                AscendC::LocalTensor<half> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<ElementC>(0);
                tmpBuffer1.SetSize(tmpBufferSize);
                int tmpBufferOffset = 96 * 1024;  // half of UB
                AscendC::LocalTensor<half> tmpBuffer2 =
                    resource.ubBuf.template GetBufferByByte<ElementC>(tmpBufferOffset);
                tmpBuffer2.SetSize(tmpBufferSize);

                // [AllGather] 2. Pre Interface Sync
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                int pingpongId = 0;

                for (uint32_t processIndex = 0; processIndex < processLoop; ++processIndex) {
                    MatrixCoord processCoord{processIndex / processCount.column(),
                                             processIndex % processCount.column()};
                    auto actualProcessShape =
                        GetActualShape(processCount, processCoord, params.processShape, residueProcessShape);

                    auto processOffset = processCoord * params.processShape;

                    uint32_t residueM = actualProcessShape.row();

                    while (residueM > 0) {
                        MatrixCoord outputLoopOffset = (offsetOut + processOffset + subBlockOffset) / gemmBlockShape;
                        MatrixCoord residueOutputOffset = (offsetOut + processOffset + subBlockOffset) % gemmBlockShape;

                        auto loopIdx = outputLoopOffset.row();
                        GemmCoord outputBlockGemmTileOffset = params.gemmSwizzle.GetBlockCoord(loopIdx);
                        GemmCoord outputBlockGemmActualSize =
                            params.gemmSwizzle.GetActualBlockShape(outputBlockGemmTileOffset);
                        MatrixCoord outputBlockTileOffset{outputBlockGemmTileOffset.m(), outputBlockGemmTileOffset.n()};

                        uint32_t actualMoveM = min(gemmBlockShape.row() - residueOutputOffset.row(), residueM);
                        if (residueOutputOffset.row() < outputBlockGemmActualSize.m()) {
                            actualMoveM = min(outputBlockGemmActualSize.m() - residueOutputOffset.row(), residueM);

                            auto inputOffset = offsetIn + subBlockOffset + processOffset;
                            auto outputOffset = outputBlockTileOffset * gemmBlockShape + residueOutputOffset;

                            auto actualMoveShape = MatrixCoord{actualMoveM, outputBlockGemmActualSize.n()};
                            layout::RowMajor layoutInput = layoutPeerMemStore.GetTileLayout(actualMoveShape);
                            layout::RowMajor layoutOutput = params.layoutDestination.GetTileLayout(actualMoveShape);
                            int64_t inputElemOffset = layoutInput.GetOffset(inputOffset);
                            int64_t outputElemOffset = layoutOutput.GetOffset(outputOffset);

                            uint32_t copySize = actualMoveShape.row() * actualMoveShape.column();

                            AscendC::TEventID event_id = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                            AscendC::LocalTensor<ElementC> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;

                            // [AllGather] 2. Pre Interface Sync
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

                            non_contiguous_copy_param copyParams;
                            copyParams.repeat = actualMoveShape.row();
                            copyParams.length = actualMoveShape.column();
                            copyParams.src_ld = layoutInput.stride(0);
                            copyParams.dst_ld = layoutOutput.stride(0);

                            // [AllGather] 3. Start aclshmemx_mte_get_nbi non-contiguous version
                            aclshmemx_mte_get_nbi(params.destination[outputElemOffset], peerMem[inputElemOffset], buf,
                                                  copyParams, mRankIdx % rankSize, event_id);

                            // [AllGather] 4. Post Interface Sync
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                            pingpongId = (pingpongId + 1) % BufferNum;
                        }
                        residueM -= actualMoveM;
                        processOffset += MatrixCoord{actualMoveM, 0};
                    }
                }
                // [AllGather] 4. Post Interface Sync
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }

        // AllGather is completed, waiting until tasks on all devices are complete.
        aclshmemx_barrier_all_vec();
    }
```

---

## A20. `aclshmem_putmem_signal` — put-with-signal DOES exist on Ascend, and `quiet` is the thing that orders it

This is the direct answer to "what orders a remote write against the local read": the library's
own put-with-signal is literally `put_nbi` → `quiet`/`fence` → `signal_op`, and nothing else.

https://raw.githubusercontent.com/immengzi/shmem/master/include/device/gm2gm/shmem_device_so.h
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

```c
/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE
 *       then update sig_addr
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:
 *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_signal(__gm__ void *dst, __gm__ void *src, size_t elem_size, __gm__ int32_t *sig_addr,
                                      int32_t signal, int sig_op, int pe);
#define shmem_putmem_signal aclshmem_putmem_signal
```

Typed variants are macro-generated over the full `ACLSHMEM_TYPE_FUNC` list, in both raw-pointer
and `GlobalTensor` form:

```c
/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, size_t elem_size,\
 * \_\_gm\_\_ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,       \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL);
/** \endcond */
#define shmem_put_half_mem_signal aclshmem_half_put_signal
#define shmem_put_float_mem_signal aclshmem_float_put_signal
#define shmem_put_double_mem_signal aclshmem_doubl_put_signal
#define shmem_put_int8_mem_signal aclshmem_int8_put_signal
#define shmem_put_int16_mem_signal aclshmem_int16_put_signal
#define shmem_put_int32_mem_signal aclshmem_int32_put_signal
#define shmem_put_int64_mem_signal aclshmem_int64_put_signal
#define shmem_put_uint8_mem_signal aclshmem_uint8_put_signal
#define shmem_put_uint16_mem_signal aclshmem_uint16_put_signal
#define shmem_put_uint32_mem_signal aclshmem_uint32_put_signal
#define shmem_put_uint64_mem_signal aclshmem_uint64_put_signal
#define shmem_put_char_mem_signal aclshmem_char_put_signal
#define shmem_put_bfloat16_mem_signal aclshmem_bfloat16_put_signal
```

```c
/**
 * @brief  Automatically generates aclshmem put signal functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE>\
 * src, size_t elem_size, __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
 *
 * @par Function Description
 *      Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local device of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the dest and source arrays.
 * - **sig_addr**    - [in] Symmetric address of the signal word to be updated.
 * - **signal**      - [in] The value used to update sig_addr.
 * - **sig_op**      - [in] Operation used to update sig_addr with signal. Supported operations:
 *                          ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                    size_t elem_size, __gm__ int32_t *sig_addr, int32_t signal,             \
                                                    int sig_op, int pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR);
```

https://raw.githubusercontent.com/immengzi/shmem/master/src/device/gm2gm/shmem_device_so.hpp
Huawei Technologies Co., Ltd. (CANN `shmem`), mirror HEAD 2026-03-04

The blocking form uses `aclshmem_quiet()`:

```cpp
ACLSHMEM_DEVICE void aclshmem_putmem_signal(__gm__ void *dst, __gm__ void *src, size_t elem_size, __gm__ int32_t *sig_addr,
                                      int32_t signal, int sig_op, int pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Set */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;
    aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    aclshmem_quiet();
    aclshmemi_signal_op(sig_addr, signal, sig_op, pe);
}

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,          \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)    \
    { /* ROCE */ /* RDMA */ /* MTE  */ /* Global State Set */                                                        \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;                      \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                     \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                    \
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,     \
                              copy_event_id);                                                                        \
        __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);                               \
        aclshmem_quiet();                                                                                            \
        aclshmemi_signal_op(sig_addr, signal, sig_op, pe);                                                           \
    }
```

The non-blocking form differs in exactly one line — `aclshmem_fence()` instead of
`aclshmem_quiet()`. Since (per A1/A2) `fence` is *implemented identically to* `quiet` on this
hardware, the two are currently the same instruction sequence:

```cpp

ACLSHMEM_DEVICE void aclshmem_putmem_signal_nbi(__gm__ void *dst, __gm__ void *src, size_t elem_size,
                                          __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Set */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;
    aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    aclshmem_fence();
    aclshmemi_signal_op(sig_addr, signal, sig_op, pe);
```

Note also where the staging UB comes from: `device_state->mte_config.aclshmem_ub` /
`.ub_size` / `.sync_id` — a library-reserved UB window (`ACLSHMEM_INTERNAL_UB_BUF_START_ADDR =
188 * 1024`, see A11), not caller-supplied. A hand-written megakernel that wants its own UB
budget must use the `ub2gm` `aclshmemx_mte_put_nbi` overloads (A11) and open-code the
`quiet` + `signal_op` pair itself.

## A21. LCCL's epoch mechanism, verified first-hand — magic in the high 32 bits, and flags moved by MTE instead of scalar stores

LCCL (`src/kernels/lcal/` inside Huawei's `ascend-transformer-boost`) is the production
low-latency collective library. Its whole cross-rank synchronisation for a one-shot all-reduce is
three calls. I re-fetched these files rather than relying on the delegated sweep because the
flag transport mechanism here differs from `aclshmem`'s in a way that matters.

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/ascendc_kernels/allreduce_one_shot.h
Huawei Technologies Co., Ltd., `Copyright (c) 2024`, CANN Open Software License Agreement Version 1.0 (GitHub mirror of ATB; fetched 2026-10-20)

```cpp

    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        if (blockIdx >= blockNum) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        CpInputToBuffAndOutput();
        sync.SetInnerFlag(magic, 1);

        sync.WaitRankInnerFlag(magic, 1, rank);
        sync.WaitRankInnerFlag(magic, 1, rankIDOfBlock);
        if (rankIDOfBlock != rank) {
            if constexpr (!std::is_same_v<T, U>) {
                if (!isEnableScale) {
                    Collectives::CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp);
                } else if (!isVectorScale) {
                    CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp, firstScale, offset);
                } else {
                    CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp, scaleGt, scaleNum, offset);
                }
            } else {
                Collectives::CpGM2GM<T, U>(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp);
            }
```

The data-staging half, for context — every block copies its slice both into the local IPC
window and straight to the output, then the reduce pass accumulates peers on top:

```cpp
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }

    FORCE_INLINE_AICORE void CpInputToBuffAndOutput()
    {
        Collectives::CpGM2GM<U>(dstIPCGlobal, srcInputGlobal, blockDataNum, COPYONLY);
        if constexpr (!std::is_same_v<T, U>) {
            if (!isEnableScale) {
                Collectives::CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY);
            } else if (!isVectorScale) {
                CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY, firstScale, offset);
            } else {
                CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY, scaleGt, scaleNum, offset);
            }
        } else {
```

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/ascendc_kernels/sync_collectives.h
Huawei Technologies Co., Ltd., `Copyright (c) 2024` (same mirror; fetched 2026-10-20)

Layout: an int64 flag per block, 4 int64 apart (32 B stride), with the inner region at the base
of the peer's shared window and the outer region offset by `segmentCount`:

```cpp

constexpr int64_t FLAG_UNIT_INT_NUM = 4;
constexpr int64_t SYNC_UNIT_SIZE = FLAG_UNIT_INT_NUM * sizeof(int64_t);
constexpr int64_t MAGIC_OFFSET = 32;
constexpr int64_t MAGIC_MASK = ~((1LL << MAGIC_OFFSET) - 1);
#ifdef ENABLE_LCCL_MIX
constexpr int32_t LCAL_BLOCK_NUM_MULTI = 2;
#else
constexpr int32_t LCAL_BLOCK_NUM_MULTI = 1;
#endif

class SyncCollectives {
public:
    __aicore__ inline SyncCollectives() {}

    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR *shareAddrs)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum() * LCAL_BLOCK_NUM_MULTI;
        segmentCount = GetBlockNum() * LCAL_BLOCK_NUM_MULTI * FLAG_UNIT_INT_NUM;
        localSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]);
        basicSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + segmentCount + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        TPipe pipe;
        pipe.InitBuffer(tBuf, GetBlockNum() * SYNC_UNIT_SIZE);
```

The epoch encoding — 32-bit magic in the high half, event/progress id in the low half:

```cpp
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value)
    {
        return (static_cast<int64_t>(magic) << MAGIC_OFFSET) | static_cast<int64_t>(value);
    }

    __aicore__ inline __gm__ int64_t* GetInnerFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline __gm__ int64_t* GetOuterFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + segmentCount + flagBlock * FLAG_UNIT_INT_NUM;
    }
```

The flag setters that use it:

```cpp
    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag(localSyncAddr + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline int32_t CalEventIdByMulBlockNum(int32_t blockMultiplier, int32_t targetCoreId)
    {
        return (blockMultiplier * blockNum) + targetCoreId;
    }

    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank,
        int32_t breakCycle = 0)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v, breakCycle);
    }

    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(basicSyncAddr, value);
    }
    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag((__gm__ int64_t*)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
    }

    __aicore__ inline void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }

    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    __aicore__ inline bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        return CheckOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }
```

The stale-rejection predicate. This is the exact test the caller asked about — one word carries
both the generation and the progress, and the waiter demands an exact generation match *and*
monotone progress:

```cpp
    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue,
        int32_t breakCycle = 0)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        do {
            if (breakCycle > 0) {
                int64_t systemCycleBefore = AscendC::GetSystemCycle();
                int64_t systemCycleAfter = AscendC::GetSystemCycle();
                while (systemCycleAfter - systemCycleBefore < breakCycle) {
                    systemCycleAfter = AscendC::GetSystemCycle();
                };
            }
            DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            isSync = true;
            for (auto i = 0; i < flagNum; ++i) {
                int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    break;
                }
            }
        } while (!isSync);
        tBuf.FreeTensor(localWait);
    }
```

The non-blocking variant with the same predicate:

```cpp
    __aicore__ inline bool CheckOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        bool isSync = true;
        for (auto i = 0; i < flagNum; ++i) {
            int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
            if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                isSync = false;
                break;
            }
        }
        tBuf.FreeTensor(localWait);
        return isSync;
```

And the write path:

```cpp
    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(1, 0);
        localSet.SetValue(0, setValue);
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        tBuf.FreeTensor(localSet);
    }
```

The read path:

```cpp
    __aicore__ inline int64_t GetFlag(__gm__ int64_t* waitAddr)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(1, 0);
        DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int64_t res = localWait.GetValue(0);
        tBuf.FreeTensor(localWait);
        return res;
    }
```

Mechanism note worth flagging: LCCL never does a scalar load or a scalar store to a flag. Both
directions go through the move engines — `localSet.SetValue(0, v)` then
`SetFlag/WaitFlag<HardEvent::S_MTE3>` then `DataCopy(globalSet, localSet, ...)` on the way out,
and `DataCopy(localWait, globalWait, ...)` then `SetFlag/WaitFlag<HardEvent::MTE2_S>` then
`GetValue` on the way in. That is why there is no `dcci` anywhere in this file, in contrast to
`aclshmem`'s scalar-store + `dcci_cacheline` approach (A4, A6). The `SetFlag` wrapper also opens
with an `MTE3_S` and an `MTE2_S` pair before touching UB, which is what orders a preceding data
`DataCopy` against the flag write.

The optional `breakCycle` argument is a cycle-counted backoff inside the spin, using
`AscendC::GetSystemCycle()`.

---

# TOPIC A, continued — epoch / generation tagging and one-shot all-reduce in real Ascend code

Items below are verbatim quotations collected from the URLs shown. Duplicates of the
`immengzi/shmem` headers already quoted in sections A1-A19 have been removed; one
`raw.gitcode.com/cann/shmem` fetch is retained to anchor provenance of the mirror.
`egretwAlker/cpl-ascend-transformer-boost` is a GitHub mirror of Huawei's
`ascend-transformer-boost` (ATB), whose `src/kernels/lcal/` tree is LCCL — the low-latency
collective library with the one-shot all-reduce AscendC kernels.

### item 0

https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/csrc/utils/inc/kernel/sync_collectives.h
vllm-project/vllm-ascend, commit bd8be2e7592f4e16b34b6185707ed0745733cf3a by shiro-zzzz, 2025-12-10

```cpp
#ifndef SYNC_COLLECTIVES_H
#define SYNC_COLLECTIVES_H

#include "comm_args.h"

using namespace AscendC;
using namespace Moe;

// Synchronization flag occupies length
constexpr int64_t FLAG_UNIT_INT_NUM = 4;
// Memory size occupied by each synchronization unit (Bytes)
constexpr int64_t SYNC_UNIT_SIZE = FLAG_UNIT_INT_NUM * sizeof(int64_t);
// High-order offset when using magic as a comparison value
constexpr int64_t MAGIC_OFFSET = 32;
constexpr int64_t MAGIC_MASK = ~((1LL << MAGIC_OFFSET) - 1);

class SyncCollectives {
public:
    __aicore__ inline SyncCollectives() {}

    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR *shareAddrs, TBuf<QuePosition::VECCALC> &tBuf)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
        // Length of a single indicator segment
        segmentCount = GetBlockNum() * FLAG_UNIT_INT_NUM;
        // Initialize the intra-card/inter-card synchronization address corresponding to the current core.
        localSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]);
        basicSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + segmentCount + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        this->tBuf = tBuf;
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag(localSyncAddr + eventID * FLAG_UNIT_INT_NUM, v);
    }

    /**
     * @brief Set the flag for the specified eventID of the designated card, with the value being a combination of magic and value.
     * @param magic The operator batch, which will be combined into the high 32 bits of the flag value to be set.
     * @param value The specific value to be set, which will be the low 32 bits of the flag value to be set.
     * @param eventID Physically, it is an offset from the shared memory base address (requires scaling, not an absolute value).
     * @param rank This rank is the rankId corresponding to the peerMems array in the CommArgs structure, not a global or local id.
     *              (Local is not applicable in the 91093 scenario, and global is not applicable in the 910B multi-machine scenario.)
     */
    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline int32_t CalEventIdByMulBlockNum(int32_t blockMultiplier, int32_t targetCoreId)
    {
        return (blockMultiplier * blockNum) + targetCoreId;
    }

    /**
     * @brief Wait for the flag of the specified eventID on the specified card to become a value
     *        composed of the combination of magic and value.
     * @param magic The operator batch, which will be combined into the high 32 bits of the flag
     *              value to be wait.
     * @param value The specific value to be wait, which will be the low 32 bits of the flag
     *              value to be wait.
     * @param eventID Physically, it is an offset from the shared memory base address (requires
     *                scaling, not an absolute value).
     * @param rank This rank is the rankId corresponding to the peerMems array in the CommArgs
     *              structure, not a global or local id. (Local is not applicable in the 91093
     *              scenario, and global is not applicable in the 910B multi-machine scenario.)
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v);
    }

    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[this->rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v);
    }

    /**
     * @brief Wait for the flags starting from the specified eventID on the specified card to become
     *        a value composed of the combination of magic and value.<br>
     *        Note: [eventID, eventID + flagNum)
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank, int64_t flagNum)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, flagNum, v);
    }

    // Set inner-card synchronization flag (memory A)
    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(basicSyncAddr, value);
    }

    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag((__gm__ int64_t*)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
    }

    // Wait for a single inner-card synchronization flag (memory A)
    __aicore__ inline void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }

    // Wait for all inner-card synchronization flags within the entire rank (memory A)
    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    // Check all inner-card synchronization flags within the entire rank (memory A)
    __aicore__ inline bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        return CheckOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    // Set inter-card synchronization flag (memory B)
    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(blockOuterSyncAddr, value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(setRank, setBlock);
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(flagAddr, value);
    }

    // Wait for a single inter-card synchronization flag (memory B)
    __aicore__ inline void WaitOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(waitRank, waitBlock);
        WaitOneRankPartFlag(flagAddr, 1, value);
    }

    // Wait for all inter-card synchronization flags within the entire rank (memory B)
    __aicore__ inline void WaitOneRankOuterFlag(int32_t magic, int32_t eventID, int64_t rank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(rank, 0);
        WaitOneRankPartFlag(flagAddr, blockNum, value);
    }

    // Wait for flagNum inter-card synchronization flags starting from startBlock for all ranks (memory B)
    __aicore__ inline void WaitAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // Offset reading of rank flags to prevent performance impact from concurrent copying by multiple cores
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            WaitOneRankPartFlag(flagAddr, flagNum, value);
        }
    }

    // Check flagNum inter-card synchronization flags starting from startBlock for all ranks (memory B)
    __aicore__ inline bool CheckAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock,
        int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // Offset reading of rank flags to prevent performance impact from concurrent copying by multiple cores
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            if (!CheckOneRankPartFlag(flagAddr, flagNum, value)) {
                return false;
            }
        }
        return true;
    }

    // Wait for all inter-card synchronization flags for all ranks, full rank synchronization (memory B)
    __aicore__ inline void WaitAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        WaitAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // Check all inter-card synchronization flags for all ranks, full rank synchronization (memory B)
    __aicore__ inline bool CheckAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        return CheckAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // Low-level interface, set synchronization flag
    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(1, 0);
        localSet.SetValue(0, setValue);

        // Copy global synchronization flag to local
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // Wait for SetValue to complete
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // Wait for UB->GM to complete
    }

    // Low-level interface, wait for synchronization flag
    __aicore__ inline void WaitFlag(__gm__ int64_t* waitAddr, int64_t waitValue)
    {
        WaitOneRankPartFlag(waitAddr, 1, waitValue);
    }

    // Read a flag, return an immediate number
    __aicore__ inline int64_t GetFlag(__gm__ int64_t* waitAddr)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(1, 0);
        // Copy global to local
        DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

        int64_t res = localWait.GetValue(0);
        return res;
    }

    // Get multiple consecutive synchronization flags within a single card
    __aicore__ inline void WaitOneRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank,
                                                    int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(waitRank, startBlock);
        WaitOneRankPartFlag(flagAddr, flagNum, value);
    }

    // Get synchronization flag within a single card (memory A)
    __aicore__ inline int64_t GetInnerFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM);
    }

    __aicore__ inline int64_t GetOuterFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + segmentCount + waitBlock * FLAG_UNIT_INT_NUM);
    }

    // In the rank Chunk Flag area, return success if the destRank chunk Flag value is 0, otherwise fail
    __aicore__ inline int64_t GetChunkFlag(int64_t rank, int64_t destRank, int64_t magic, int64_t timeout)
    {
        int64_t value = MergeMagicWithValue(magic, 0);
        int64_t status = GetChunkFlagValue((__gm__ int64_t*)(shareAddrs[rank]) +
                                            IPC_CHUNK_FLAG + destRank * FLAG_UNIT_INT_NUM, value, timeout);
        return status;
    }

    // Set the destRank chunk Flag value in the rank Chunk Flag area to value
    __aicore__ inline void SetChunkFlag(int64_t rank, int64_t destRank, int64_t magic, int64_t eventId)
    {
        int64_t value = MergeMagicWithValue(magic, eventId);
        SetFlag((__gm__ int64_t*)(shareAddrs[rank]) + IPC_CHUNK_FLAG + destRank * FLAG_UNIT_INT_NUM, value);
    }

    __aicore__ inline int64_t GetChunkRecvLen(int64_t rank, int64_t destRank, int64_t magic, int64_t timeout)
    {
        int64_t len = GetChunkFlagValue((__gm__ int64_t*)(shareAddrs[rank]) + IPC_CHUNK_FLAG +
                                        destRank * FLAG_UNIT_INT_NUM, 0, timeout, true, magic);
        return len;
    }

private:
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value)
    {
        // Merge magic as the high bits and eventID as the low bits into a value for comparison
        return (static_cast<int64_t>(static_cast<uint32_t>(magic)) << MAGIC_OFFSET) | static_cast<int64_t>(value);
    }

    __aicore__ inline __gm__ int64_t* GetInnerFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline __gm__ int64_t* GetOuterFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + segmentCount + flagBlock * FLAG_UNIT_INT_NUM;
    }

    // Wait for a part of synchronization flags within a rank
    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        int64_t checkedFlagNum = 0;
        do {
            // Copy global synchronization flags to local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM],
                     (flagNum - checkedFlagNum) * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            // Check if the synchronization flags are equal to checkValue
            isSync = true;
            int64_t remainToCheck = flagNum - checkedFlagNum;
            for (auto i = 0; i < remainToCheck; ++i) {
                // Continue waiting if any core has not reached the checkValue phase
                int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    checkedFlagNum += i;
                    break;
                }
            }
        } while (!isSync);
    }

    // Wait for all synchronization flags within a rank
    __aicore__ inline void WaitOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        WaitOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    // Check partial synchronization flags within a rank, copy only once
    __aicore__ inline bool CheckOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        // Copy global synchronization flags to local
        DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB
        // Check if the synchronization flags are equal to checkValue
        bool isSync = true;
        for (auto i = 0; i < flagNum; ++i) {
            // Continue waiting if any core has not reached the checkValue phase
            int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
            if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                isSync = false;
                break;
            }
        }
        return isSync;
    }

    __aicore__ inline int64_t GetChunkFlagValue(__gm__ int64_t* waitAddr, int64_t checkValue, int64_t timeout,
                                                bool checkNonZero = false, int64_t magic = 0)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;

        int64_t waitTimes = 0;
        int64_t v = 0;

        do {
            // Copy global sync flag to local
            DataCopy(localWait, globalWait[0], FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            isSync = true;
            v = localWait.GetValue(0);
            if (checkNonZero) {
                // Non-zero check mode
                if (((v & MAGIC_MASK) == (static_cast<int64_t>(magic) << MAGIC_OFFSET)) && (v & 0xFFFFFFFF)) {
                    return v & 0xFFFFFFFF;  // Return lower 32 bits when non-zero
                }
            } else {
                // Exact value check mode
                if (v == checkValue) {
                    return WAIT_SUCCESS;
                }
            }

            isSync = false;
            waitTimes++;

            if (timeout > INT64_MAX / MAX_WAIT_ROUND_UNIT || waitTimes >= (timeout * MAX_WAIT_ROUND_UNIT)) {
                isSync = true;
                return v; // Return the read flag value
            }
        } while (!isSync);

        return checkNonZero ? 0 : v;
    }

    // Check all sync flags within a rank, copy only once
    __aicore__ inline bool CheckOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        return CheckOneRankPartFlag(waitAddr, blockNum, checkValue);
    }
    int rank;
    int rankSize;
    int blockIdx;
    int blockNum;
    GM_ADDR *shareAddrs;
    int64_t segmentCount;  // Length of a single sync flag segment (count in int64_t)
    __gm__ int64_t* localSyncAddr;
    __gm__ int64_t* basicSyncAddr;  // Intra-card sync flag address for the current block
    __gm__ int64_t* blockOuterSyncAddr;  // Inter-card sync flag address for the current block
    TBuf<QuePosition::VECCALC> tBuf;
};

#endif // SYNC_COLLECTIVES_H```

## ITEM 2 — vllm-ascend `comm_args.h`: the shared IPC window layout the epoch flags live in (first 2MB = flag region, ping-pong buffer selected by magic parity, `WAIT_SUCCESS` sentinel, `MAX_WAIT_ROUND_UNIT` spin budget)

### item 1

https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/csrc/utils/inc/kernel/comm_args.h
vllm-project/vllm-ascend, commit bd8be2e7592f4e16b34b6185707ed0745733cf3a by shiro-zzzz, 2025-12-10

```cpp
#ifndef COMM_ARGS_H
#define COMM_ARGS_H
#include <cstdint>

#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__
#include "kernel_operator.h"

namespace Moe {
constexpr int CAM_MAX_RANK_SIZE = 384; // Maximum number of NPU cards supported by the communication library

constexpr int64_t IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024; // First 2MB as flag, then 100MB as data storage
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int UB_ALIGN_SIZE = 32;
constexpr int64_t MAGIC_ALIGN_COUNT = UB_ALIGN_SIZE / sizeof(int32_t);

constexpr uint8_t COMM_NUM = 2;  // Size of communication domain
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint8_t COMM_TP_IDX = 1;

constexpr int DFX_COUNT = 50;
constexpr int64_t WAIT_SUCCESS = 112233445566;
constexpr int64_t IPC_CHUNK_FLAG = 0; // Start offset for send recv, chunk flag region
constexpr int64_t MAX_WAIT_ROUND_UNIT = 10 * 1000 * 1000; // Threshold for waiting to get Flag under normal conditions within the same SIO

constexpr static int32_t UB_HEAD_OFFSET = 96;
constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN_SIZE;
constexpr static int64_t UB_FLAG_SIZE = 2 * 1024;
constexpr static int64_t MAX_CORE_NUM = 48;
constexpr static uint64_t STATE_WIN_OFFSET = 900 * 1024;
constexpr static int64_t COMPARE_ALIGN_SIZE = 256;

constexpr static int64_t UB_SINGLE_TOTAL_SIZE_MAX = 192 * 1024;
constexpr static int64_t START_OFFSET_FOR_SHARE = 512;

enum Op : int {
    COPYONLY = -1,
    ADD = 0,
    MUL = 1,
    MAX = 2,
    MIN = 3
};

struct CommArgs {
    int rank = 0;           // attr rank_id, global rank
    int localRank = -1;
    int rankSize = 0; // global rank size
    int localRankSize = -1;  // This parameter refers to the number of cards interconnected in fullmesh
    uint32_t extraFlag = 0; // 32 bit map, the specific meaning of each bit is above in this file
    int testFlag = 0;
    GM_ADDR peerMems[CAM_MAX_RANK_SIZE] = {}; // Buffer obtained from initialization, all allreduce is the same parameter
    /**
     * @param sendCountMatrix One-dimensional array with a size of rankSize*rankSize
     * eg: The value of sendCountMatrix[1] corresponds to the [0][1] of the two-dimensional array, indicating the number of data that card 0 needs to send to card 1
     */
    int64_t sendCountMatrix[CAM_MAX_RANK_SIZE * CAM_MAX_RANK_SIZE] = {};  // for all2allvc
    int64_t sendCounts[CAM_MAX_RANK_SIZE] = {}; // for all2allv
    int64_t sdispls[CAM_MAX_RANK_SIZE] = {};    // for all2allv
    int64_t recvCounts[CAM_MAX_RANK_SIZE] = {};  // for all2allv
    int64_t rdispls[CAM_MAX_RANK_SIZE] = {};     // for all2allv
    int64_t batchSize;
    int64_t hiddenSize;
    int64_t topk;
    int64_t sharedExpertRankNum;
    int64_t expertNumPerRank;
    int64_t dfx[DFX_COUNT] = {};
};
}
#endif // COMM_ARGS_H
```

## ITEM 3 — sgl-kernel-npu `NotifyDispatchA2`: magic is read-modify-written FROM the symmetric window itself at kernel entry (per-block, self-incrementing generation counter, no host barrier), and the magic parity selects the ping-pong half of the window

### item 2

https://raw.githubusercontent.com/sgl-project/sgl-kernel-npu/main/csrc/deepep/ops2/op_kernel/notify_dispatch_a2.h
sgl-project/sgl-kernel-npu, commit 45aa6ebb54694c7c9f3e97f64f963c44bcd3937c by luanyundu, 2026-03-02

```cpp

    constexpr static uint32_t MAX_BS = 4096;  // 每卡支持的最大bs
    // Synchronization flag occupies length
    constexpr static int64_t FLAG_UNIT_INT_NUM = 4;
    constexpr static int64_t MAGIC_MASK = ~((1LL << 32) - 1);

public:

// ---- (Init, lines 185-245): magic is loaded from GM, incremented, stored back, re-read ----
        this->root = 0;
        this->len = len;
        this->numExperts = numExperts;
        this->numTokens = numTokens;
        this->topkNum = topkNum;
        this->scale = nullptr;
        this->magic = 0;
        this->localRank = localRank;
        this->localRankSize = localRankSize;
        this->xRankSize = localRankSize;
        this->yRankSize = rankSize / localRankSize;
        this->xRankIdx = rank % localRankSize;
        this->yRankIdx = rank / localRankSize;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
        uint8_t ctxIdx;

        ctxIdx = COMM_EP_IDX;

        // 初始化RDMA相关变量
        REGISTER_TILING_DEFAULT(NotifyDispatchA2TilingData);
        GET_TILING_DATA_WITH_STRUCT(NotifyDispatchA2TilingData, tilingData, tiling);

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

        hccl_.InitV2(contextGM0, &tilingData);
        hccl_.SetCcTilingV2(offsetof(NotifyDispatchA2TilingData, mc2CcTiling1));
        this->winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)contextGM0;
        notifyMemoryOffset = winContext_[COMM_EP_IDX]->winSize - IPC_BUFF_MAX_SIZE * 2;
        // 设置并自增magic
        magicTensor_.SetGlobalBuffer((__gm__ uint64_t *)(hccl_.GetWindowsInAddr(rank) + IPC_DATA_OFFSET -
                                                         blockNum * sizeof(uint64_t) * EXP_TOKEN_COUNT_FLAG_CNT +
                                                         notifyMemoryOffset));

        pipe.InitBuffer(this->tBuf, TEMP_BUF_LEN);
        LocalTensor<uint64_t> tempLocal = tBuf.Get<uint64_t>();
        PipeBarrier<PIPE_ALL>();
        tempLocal(0) = magicTensor_.GetValue(blockIdx * EXP_TOKEN_COUNT_FLAG_CNT) + 1;
        // 使用atomic方式实现+1
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // 等待SetValue完成
        DataCopy(magicTensor_[blockIdx * EXP_TOKEN_COUNT_FLAG_CNT], tempLocal, EXP_TOKEN_COUNT_FLAG_CNT);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // 等待DataCopy完成
        magic = magicTensor_.GetValue(blockIdx * EXP_TOKEN_COUNT_FLAG_CNT);
        PipeBarrier<PIPE_ALL>();
        // 初始化目标rank的shareAddrs
        for (int i = 0; i < rankSize; i++) {
            this->shareAddrs[i] =
                hccl_.GetWindowsInAddr(i) + notifyMemoryOffset + (magic % PING_PONG_SIZE) * IPC_BUFF_MAX_SIZE;
        }
    }

    template <typename K, typename U = K>
    __aicore__ inline void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U> &sendDataInputGt,
                                           const GlobalTensor<K> &recvDataOutputGT, int op);
    template <typename F>
    __aicore__ inline void SetAtomic(int op);
    __aicore__ inline void UnsetAtomic(int op);
    template <HardEvent eventType>

// ---- (lines 330-390): MergeMagicWithValue / WaitOneRankPartFlag / SetInnerFlag / WaitInnerFlag ----
    __aicore__ inline uint64_t MergeMagicWithValue(uint64_t magic, uint64_t value)
    {
        // magic as the high part, eventID as the low part, combined into a value for comparison
        return (magic * 2ULL + value);
    }

    // Wait for a part of synchronization flags within a rank
    __aicore__ inline void WaitOneRankPartFlag(__gm__ uint64_t *waitAddr, int64_t flagNum, uint64_t checkValue)
    {
        GlobalTensor<uint64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<uint64_t> localWait = tBuf.GetWithOffset<uint64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        uint64_t checkedFlagNum = 0;
        do {
            // Copy global synchronization flags to local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM],
                     (flagNum - checkedFlagNum) * FLAG_UNIT_INT_NUM);
            SetWaitEvent<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            // Check if the synchronization flags are equal to checkValue
            isSync = true;
            uint64_t remainToCheck = flagNum - checkedFlagNum;
            for (auto i = 0; i < remainToCheck; ++i) {
                // Continue waiting if any core has not reached the checkValue phase
                uint64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    checkedFlagNum += i;
                    break;
                }
            }
        } while (!isSync);
    }

    __aicore__ inline void SetInnerFlag(uint64_t magic, uint64_t eventID, int64_t setRank, int64_t setBlock)
    {
        uint64_t value = MergeMagicWithValue(magic, eventID);
        // SetFlag((__gm__ uint64_t *)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
        __gm__ uint64_t *setAddr = (__gm__ uint64_t *)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM;

        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID0);
        SetWaitEvent<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<uint64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<uint64_t> localSet = tBuf.GetWithOffset<uint64_t>(1, 0);
        localSet.SetValue(0, value);

        // Copy global synchronization flag to local
        SetWaitEvent<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // Wait for a single inner-card synchronization flag
    __aicore__ inline void WaitInnerFlag(uint64_t magic, uint64_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        uint64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ uint64_t *)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }


// ---- (lines 1280,1296): member declarations ----
    uint64_t magic;
    GlobalTensor<uint64_t> magicTensor_;  // 用于存放magic，位于windowInstatusTensor_之前
```

## ITEM 4 — CANN `shmem` device API: `aclshmemx_signal_op` / `aclshmem_signal_wait_until` declarations + verbatim semantics doc (note the explicit "WARNING: Atomicity NOT Guaranteed."), and the `ACLSHMEM_SIGNAL_SET` / `ACLSHMEM_SIGNAL_ADD` / `ACLSHMEM_CMP_*` enums

### item 3

https://raw.githubusercontent.com/immengzi/shmem/master/examples/allgather/main.cpp
immengzi/shmem, repo pushed 2026-03-26 — host side: where the epoch comes from and how it is scaled

```cpp
46:constexpr int64_t SYNC_FLAG_INTERVAL = 16;
49:constexpr uint32_t MAGIC_MULTIPLIER = 1024;
113:        void *ptr = aclshmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));
119:                              (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic * MAGIC_MULTIPLIER);

    // magic is used to sync.
    int magic = 1;

    for (int i = 0; i < test_cases.size(); i++) {
        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Started." << std::endl;
        }
        uint32_t trans_size = test_cases[i];

        //  Small data kernel needs 8 AIV core, Big data kernel needs 16 AIV.
        if (trans_size * sizeof(T) < DATA_SIZE_THRESHOLD) {
            BLOCK_NUM = BLOCK_NUM_SMALL_DATA;
        } else {
            BLOCK_NUM = BLOCK_NUM_LARGE_DATA;
        }

        void *input_ptr;
        aclrtMalloc(&input_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        uint8_t *input_host;
        aclrtMallocHost(reinterpret_cast<void**>(&input_host), trans_size * sizeof(T));
        std::string inputFile = "../../examples/allgather/golden/allgather_" + std::to_string(trans_size) + "_" +
                                std::to_string(n_ranks) + "/input_gm_" + std::to_string(rank_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_ranks * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        // sync Buffer + data Buffer
        int aiv_num = BLOCK_NUM;
        void *ptr = aclshmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));

        // AllGather
        for (int zz = 0; zz < perf_times; zz++) {
            magic++;
            allgather_demo<T>(BLOCK_NUM, stream, fftsAddr, (uint8_t *)input_ptr,
                              (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic * MAGIC_MULTIPLIER);
        }
        status = aclrtSynchronizeStream(stream);
```

## ITEM 9 — CANN cannbot-skills, shmem-ops-design `core-allocation.md` §7.1/§7.2: the pattern named explicitly as "Epoch / Magic Number 防脏读" (anti-stale-read), with the per-AIV flag slot address formula and false-sharing interval

### item 4

https://raw.githubusercontent.com/hicann/cannbot-skills/master/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/core-allocation.md
hicann/cannbot-skills, repo pushed 2026-10-19 (date of this file's last commit not individually resolved)

```markdown
## 7. 同步机制详解

### 7.1 per-AIV 独立 Flag

每个 AIV 拥有独立的同步 flag 槽，避免多核写同一地址造成竞争。

**地址计算**：

```
SYNC_FLAG_INTERVAL = 16   // int32 粒度间隔，防止 false sharing
flag_addr = gva_sync_base + (rank * aiv_num + aiv_idx) * SYNC_FLAG_INTERVAL
```

**总分配量**：`sizeof(int32_t) * n_pes * aiv_num * SYNC_FLAG_INTERVAL`

**使用模式**：
- 发送端写自己的 flag：`signal_op(my_flag_addr, value, SIGNAL_SET, target_pe)`
- 接收端读对端的 flag：`signal_wait_until(remote_flag_addr, CMP_EQ, expected_value)`

参考：`allgather`、`kv_shuffle`

### 7.2 Epoch / Magic Number 防脏读

多轮迭代中，上一轮的 flag 值可能残留在远端。通过 magic number 区分不同轮次：

```
// 发送端：每轮递增
signal_value = (magic + iteration) * 1024

// 接收端：通过高位比较判断 epoch
if (remote_flag >> 10) != (expected_magic >> 10):
    // 仍是旧 epoch 的值，继续等待
```

参考：`allgather` 大数据路径

```

### item 5

https://raw.githubusercontent.com/hicann/cannbot-skills/master/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/core-allocation.md
same file, §4.3 cross-core (AIC<->AIV) flag protocol and §6 sync-primitive selection table

```markdown
AIC 核编号：coreIdx = GetBlockIdx()，coreNum = GetBlockNum()
AIV 核编号：subcoreIdx = GetSubBlockIdx()，subcoreNum = GetSubBlockNum()
```

AIC 和 AIV 在同一 block 内通过**跨核 flag** 交替执行，不需要 Host 侧启动多个 kernel。

### 4.2 两阶段流水（2-Stage Pipeline）

通算融合的核心并发机制是**双 workspace 交替**：

```
时间线：
  Stage 0    Stage 1    Stage 0    Stage 1    ...
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │AIV 通信│ │AIC 计算│ │AIV 通信│ │AIC 计算│  ← AIC 视角
  │ws[0]   │ │ws[0]   │ │ws[1]   │ │ws[1]   │
  └────────┘ └────────┘ └────────┘ └────────┘
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │  idle  │ │AIV 通信│ │AIC 计算│ │AIV 通信│  ← AIV 视角
  │        │ │ws[1]   │ │ws[1]   │ │ws[0]   │
  └────────┘ └────────┘ └────────┘ └────────┘

实际重叠：
  AIV 通信 ws[N]  与  AIC 计算 ws[N-1]  同时进行
```

- `WORKSPACE_STAGES = 2`：两块 symmetric workspace 交替使用
- 每轮通信填充 ws[stageId]，AIC 从 ws[1-stageId] 读取上一轮数据计算
- `commInterval` 控制每 N 个 GEMM block 触发一轮通信，直接影响流水粒度

### 4.3 跨核 Flag 同步协议

AIC 和 AIV 通过两组 flag 数组交替通知对方：

```
flagAicFinishStore[stageId]    // AIC → AIV：AIC 完成当前 stage 的计算/存储
flagAivFinishCompute[stageId]  // AIV → AIC：AIV 完成当前 stage 的通信

// AIC 侧循环
for each commIdx:
    stageId = commIdx % WORKSPACE_STAGES
    CrossCoreWaitFlag(flagAivFinishCompute[stageId])  // 等 AIV 通信完成
    ... 执行 GEMM ...
    CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId])  // 通知 AIV

// AIV 侧循环
for each commIdx:
    stageId = commIdx % WORKSPACE_STAGES
    CrossCoreWaitFlag(flagAicFinishStore[stageId])  // 等 AIC 计算完成
    ... 执行 AllGather / ReduceScatter ...
    aclshmem_barrier_all()                       // 跨 PE 同步
    CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId])  // 通知 AIC
```

**关键约束**：
- flag 数组长度 = `WORKSPACE_STAGES`，每个 stage 独立 flag
- 双方必须在每个 stage 各 set 一次 flag，否则死锁
- AIV 侧通信后需先做跨 PE barrier，再 set flag 给 AIC

### 4.4 典型融合模式



## 5. 跨 PE 同步原语选择

| 原语 | 语义 | 适用场景 |
| --- | --- | --- |
| `signal_op` + `signal_wait_until` | 点对点 PE 同步 | 已知对端 PE、需要精确握手（如 kv_shuffle sender↔receiver） |
| `aclshmem_barrier_all` | 全 PE 栅栏 | 所有 PE 数据一致后同步（**新算子首选**） |
| `AscendC::SyncAll` | 同一 PE 内多 AIV 核间同步 | 纯 Vector kernel 阶段切换（如 allgather Step 1→Step 2）；见 [SyncAll](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md) |
| `AscendC::PipeBarrier` | 单 AIV 内流水线同步 | 同一 core 内 MTE/Vector 等 pipe 之间的顺序与可见性 |
| `aclshmem_quiet` | 等待本 PE 已发出的**全引擎** RMA 完成（MTE + SDMA + UDMA + RDMA） | 混合引擎或收尾阶段，需一次性 drain 所有在途传输 |
| `aclshmemx_mte_quiet` | 等待本地 MTE 完成 | **仅** MTE put/get 后确保数据落地；勿用全局 `aclshmem_quiet` 代替 |
| `aclshmemx_sdma_quiet` | 等待本地 SDMA 完成 | SDMA 传输后确保完成 |
| `aclshmemx_udma_quiet` | 等待指定 PE 的 UDMA 完成 | PCIE/SIO 等 UDMA 通路 |
| `aclshmemx_roce_quiet` | 等待指定 PE 的 RDMA/RoCE 完成 | 跨机 RDMA 通路 |
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIC↔AIV 跨核同步 | 通算融合的 stage 交替 |

> **选型**：单引擎场景 **SHOULD** 使用 engine-specific quiet（如 MTE 用 `aclshmemx_mte_quiet`），与 `aclshmemx_sdma_quiet` 对称；仅在多引擎混用或无法区分引擎时再调用 `aclshmem_quiet`。

```

## ITEM 10 — CANN cannbot-skills `optimization-patterns.md` §1.1 and §3.1/§3.2: "不需要全局 barrier" (no global barrier needed) copy-to-symmetric-heap overlap, and the recommended signal encoding `flag = chunk_count + magic` with epoch validation by `(flag >> shift) == (magic >> shift)`

### item 6

https://raw.githubusercontent.com/hicann/cannbot-skills/master/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-optim/references/optimization-patterns.md
hicann/cannbot-skills, repo pushed 2026-10-19

```markdown
## 1. 通信优化

### 1.1 Copy-to-Symmetric-Heap Overlap

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 数据 **MUST** 先从本地 GM 拷贝到 symmetric heap，再由远端 PE 读取；拷贝时间成为串行开销 |
| 优化 | 将 core 分为 sender 组和 receiver 组，sender 按 chunk 拷贝到 symmetric heap 并逐 chunk 发 signal，receiver 轮询 signal 并立即拉取已就绪的 chunk |
| 来源 | `examples/allgather` — `all_gather_origin` kernel |

❌ 错误做法：
```cpp
// 所有数据拷贝完 → 全局 barrier → 再开始通信
copy_all_to_symmetric(input, gva_data, total_size);
aclshmem_barrier_all();
fetch_all_from_peers(output, gva_data, total_size);
```

✅ 正确做法：
```cpp
// Sender core：逐 chunk 拷贝 + 逐 chunk signal
if (aivIndex < core_group_num) {
    while (remaining >= chunk_size) {
        aclshmemx_mte_put_nbi(gva_data + offset, input + offset,
                              tmp_buff, ub_size, chunk_elems, my_rank, EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        flag = ++times + magic;
        aclshmemx_signal_op(gva_sync + flag_offset, flag, SIGNAL_SET, my_rank);
    }
}

// Receiver core：轮询 signal，就绪即拉取
while (!all_done) {
    for (int g = 0; g < core_group_num; g++) {
        aclshmem_int32_get_nbi(flag_ub, gva_sync + g * INTERVAL, 1, peer);
        PipeBarrier<PIPE_ALL>();
        int64_t ready = *flag_ub - magic;
        if (ready > fetched[g]) {
            // 立即拉取已就绪 chunk，使用 ping-pong buffer
            fetch_chunks(output, gva_data, fetched[g], ready, ping_pong_buf);
        }
    }
}
```

关键点：
- sender 和 receiver 的 core 数量各占一半
- signal 使用 magic + chunk_count 编码，receiver 通过解码判断就绪程度
- 不需要全局 barrier，数据逐 chunk 流式可用


### 3.1 Barrier → Signal/Wait

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 全局 barrier 使所有 PE 等最慢者 |
| 优化 | 替换为 producer-consumer 点对点 signal/wait |

❌ 错误做法：
```cpp
aclshmem_barrier_all();  // 每个 phase 全员等待
```

✅ 正确做法：
```cpp
// Producer 完成后只通知消费者
aclshmemx_signal_op(remote_sig, magic, ACLSHMEM_SIGNAL_SET, target_pe);
// Consumer 只等自己需要的数据就绪
aclshmem_signal_wait_until(local_sig, ACLSHMEM_CMP_EQ, magic);
```

### 3.2 Per-Chunk Signaling

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 整块数据传输完才 signal，消费者长时间空等 |
| 优化 | 每个 chunk 传输完立即 signal，消费者可逐 chunk 消费 |

signal 编码建议：
- `flag = chunk_count + magic`：magic 区分轮次，chunk_count 表示已就绪数量
- receiver 通过 `(flag >> shift) == (magic >> shift)` 验证轮次有效性
- 每个 core 使用独立 flag offset：`flag_offset = aivIndex * SYNC_FLAG_INTERVAL`

### 3.3 Phase 合并

| 场景 | 合并方式 |
```

## ITEM 11 — CANN cannbot-skills `atomic-add-pattern.md`: the two atomic-add mechanisms on Ascend (remote scalar AMO via `aclshmemx_{mte,udma,roce}_atomic_add` vs local bulk mode via `AscendC::SetAtomicAdd<T>()`), per-engine dtype/platform matrix, `mte_atomic_fetch_add` availability, and the explicit routing rule "单个 signal/flag 累加 -> 标量 mte_atomic_add"

### item 7

https://raw.githubusercontent.com/hicann/cannbot-skills/master/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/atomic-add-pattern.md
hicann/cannbot-skills, repo pushed 2026-10-19

```markdown
# AtomicAdd 累加模式参考手册

> 本文总结 SHMEM Device 侧使用不同引擎（MTE / UDMA / RDMA / SDMA fallback）实现跨 PE
> reduce 累加的正确模式和效率决策。适用于需要在多 PE 间进行 reduce 操作的通信算子。
> 集合通信算子。

---

## 1. 核心概念

在 SHMEM Device 侧存在**两类** atomic add 机制，必须区分清楚：

| 类型 | 机制 | 操作对象 | 数据规模 |
| --- | --- | --- | --- |
| **标量 atomic** | 通过引擎 API（`aclshmemx_mte_atomic_add` 等）对**远端**对称地址做标量加法 | 远端 PE 的 `__gm__` 地址 | 单个元素 |
| **批量 atomic** | `SetAtomicAdd<T>()` 切换当前 core 的 GM 写入模式，后续 MTE3 写入（如 `mte_get_nbi` 的目的端）自动变为 `D += src` | **本地** GM 地址 | tile 级别 |

`SetAtomicAdd<T>()` 不是"执行一次加法"，而是**把当前 core 后续的所有 GM 写入模式切换为 atomic add**。真正发生加法的是后续的 MTE3 写入操作。

| 概念 | 说明 |
| --- | --- |
| `SetAtomicAdd<T>()` | 切换当前 core 的 GM 写模式为 atomic add，T 为元素类型（`half`/`float`/`int32`/`bfloat16`） |
| `SetAtomicNone()` | 恢复 GM 写为普通覆盖模式 |
| 标量 atomic 触发者 | `aclshmemx_mte_atomic_add` / `aclshmemx_udma_atomic_add` / `aclshmemx_roce_atomic_add` |
| 批量 atomic 触发者 | `aclshmemx_mte_get_nbi` 的目的端 GM 写入 |
| 数学语义 | 普通写：`D = src`；atomic 模式：`D = D + src` |


### 2.2 MTE 引擎：两条路径

MTE 是机内 HCCS full-mesh 场景的主力引擎，提供标量和批量两条路径。

#### 2.2.1 标量：`aclshmemx_mte_atomic_add`

```
aclshmemx_mte_atomic_add((__gm__ T*)dst, value, pe);
```

| 属性 | 说明 |
| --- | --- |
| 操作方向 | 本 PE 写入**远端 PE** 的 dst 地址，执行 `remote_dst += value` |
| 数据规模 | 单个标量值 |
| 执行单元 | 910B/910C: MTE2；Ascend950: UB + MTE3（见下方平台差异表） |
| 同步 | 需 `SetFlag<S_MTE3>` / `WaitFlag<S_MTE3>` 隔离 Scalar→MTE3 |
| dtype | `int8_t` / `int16_t` / `int32_t` / `float` / `half` / `bfloat16_t`；Ascend950 额外支持 `uint32_t` / `int64_t` / `uint64_t` |

平台差异（来自头文件注释）：

| dtype | 910B/910C | Ascend950 |
| --- | --- | --- |
| `int8_t` / `int16_t` / `half` / `bfloat16_t` | MTE2 atomic | UB + MTE3 atomic |
| `int32_t` / `float` | MTE2 atomic | UB + MTE3 atomic |
| `uint32_t` / `int64_t` / `uint64_t` | 不支持 | Scalar AtomicAdd |

#### 2.2.2 批量：`SetAtomicAdd<T>()` + `mte_get_nbi`

```
AscendC::SetAtomicAdd<ElementD>();
// 所有 mte_get_nbi 的目的端 GM 写入自动变为 D += src
reduceScatter(gmBlockSrc, ..., gmBlockDst, ..., remoteRankIdx);
// drain MTE3
AscendC::SetAtomicNone();
```

| 属性 | 说明 |
| --- | --- |
| 操作方向 | **本地** core 对本地 `D` 做 atomic add 写入 |
| 数据规模 | tile 级别（如 `32 × 256` 元素），可多次 get 并发 in-flight |
| 执行单元 | MTE3（目的端写回阶段） |
| 同步 | `PipeBarrier<PIPE_ALL>()` 隔离模式切换；`SetFlag<MTE3_S>` / `WaitFlag<MTE3_S>` drain |
| dtype | `half` / `float` / `int32` / `bfloat16`（`SetAtomicAdd` 支持的 AscendC 类型） |

#### 2.2.3 两条路径对比

| | 标量 `mte_atomic_add` | 批量 `SetAtomicAdd` + get |
| --- | --- | --- |
| 操作对象 | 远端 PE 地址 | 本地 GM 地址（目的端） |
| 数据粒度 | 单个元素 | tile（多元素批量） |
| 并发模型 | 逐元素调用，各自独立完成 | 多个 get 可全部 in-flight，MTE3 管道并行累加 |
| 典型场景 | signal 计数器、flags、单元素累加 | reduce-scatter / allreduce 的 tile 级批量累加 |
| 吞吐 | 低（标量） | 高（批量流水） |

### 2.3 UDMA 引擎

```
aclshmemx_udma_atomic_add((__gm__ T*)dst, value, pe);
aclshmemx_udma_quiet(pe);  // 确保操作在远端完成
```

| 属性 | 说明 |
| --- | --- |
| 操作方向 | 写入远端 PE 地址 |
| 数据规模 | 单个标量值 |
| 批量能力 | **无**（UDMA 不提供类似 `SetAtomicAdd` 的批量模式） |
| 同步 | 必须在读取结果前调用 `aclshmemx_udma_quiet(pe)` |
| dtype | `int32_t` / `uint32_t` / `int64_t` / `uint64_t` / `float` |
| 注意事项 | 并发 RMA/AMO 到同一 PE 不支持 |

### 2.4 RDMA 引擎

```
aclshmemx_roce_atomic_add((__gm__ T*)dst, value, pe);
```

| 属性 | 说明 |
| --- | --- |
| 平台 | Ascend950 |
| 操作方向 | 跨 server 写入远端 PE 地址 |
| 数据规模 | 单个标量值 |
| 批量能力 | **无** |
| dtype | **仅** `int32_t` / `uint32_t` / `int64_t` / `uint64_t`（不支持浮点/half） |
| 注意事项 | 并发 RMA/AMO 到同一 PE 不支持 |

### 2.5 三引擎对标总表

| 引擎 | 标量 atomic API | 批量 reduce | 支持 dtype | 链路及平台 |
| --- | --- | --- | --- | --- |
| **MTE** | `aclshmemx_mte_atomic_add` | `SetAtomicAdd<T>()` + `mte_get_nbi` | int8/16/32, float, half, bf16 (+ uint32/64/int64 on 950) | HCCS full-mesh / 910B-950 |
| **UDMA** | `aclshmemx_udma_atomic_add` | — | int32/uint32/int64/uint64, float | PCIE/SIO / 910B-950 |
| **RDMA** | `aclshmemx_roce_atomic_add` | — | int32/uint32/int64/uint64 | RoCE 跨机 / Ascend950 |
| **SDMA** | — | — | — | HCCS/PCIE/SIO（仅 Host 侧 inline reduce）/ 910B-910C |

### 2.6 SDMA 引擎（现状）

SDMA **没有 device 侧 atomic add API**。代码库中不存在 `aclshmemx_sdma_atomic_add` 或等价物。

因此，在 SHMEM AIV kernel 中实现跨 PE reduce 时，如果 MTE 不可用（如跨 PCIE 或跨 server），
必须使用 §4 描述的 SDMA fallback 策略。

---

## 3. MTE 标量 vs 批量：如何选择

| 场景 | 推荐路径 | 理由 |
| --- | --- | --- |
| 单个 signal/flag 累加 | `aclshmemx_mte_atomic_add`（标量） | 一条调用即完成，无需切换 pipeline 模式 |
| tile 级批量 reduce（reduce-scatter / allreduce） | `SetAtomicAdd<T>()` + `mte_get_nbi`（批量） | 所有 peer 的 get in-flight，MTE3 管道并行 |
| 少量元素的远端累加（< 64 元素） | 标量 `mte_atomic_add` | SetAtomicAdd 模式切换开销 > 收益 |
| 需要读取累加前旧值（fetch_add） | 标量 `mte_atomic_fetch_add`（仅 UDMA/Ascend950） | 批量模式无 fetch 语义 |

---

## 4. SDMA 如何实现并发 Reduce
```

## ITEM 12 — CANN cannbot-skills: signal/wait review rules naming the "magic/epoch 策略" explicitly, and a per-chunk monotone-GE wait `aclshmem_signal_wait_until(source_signal, ACLSHMEM_CMP_GE, magic + chunk_seq)` (a flag-free streaming handshake: value only ever goes up)

### item 8

https://raw.githubusercontent.com/hicann/cannbot-skills/master/plugins-community/shmem-ops-generator/skills/shmem-ops-code-review/references/code-review-checklist.md
hicann/cannbot-skills, repo pushed 2026-10-19

```markdown
## 3. 同步正确性

### 3.1 Signal/Wait 配对

【描述】每个 `signal` 必须有对应的 `wait` 或 `test`。signal 值必须与 wait 的期望值匹配（magic/epoch 策略需一致）。

【错误做法】
```cpp
// 发送端 signal(magic=1)，接收端 wait(magic=2)
aclshmemx_signal_op(remote_sig, 1, SIGNAL_SET, pe);
// ...
aclshmem_signal_wait_until(local_sig, ACLSHMEM_CMP_EQ, 2);  // 永远等不到
```

### 3.2 Phase 边界同步

【描述】相邻 phase 之间的同步方式必须与 `design.md` 的 `sync` 模块一致。设计写 signal/wait 的不能实现为 Host barrier；设计写 device barrier 的不能省略。

### 3.3 Quiet/Fence 使用

【描述】如果 design 要求 quiet 保证远端写入可见性，实现中不能省略。单引擎 MTE 路径用 `aclshmemx_mte_quiet`；多引擎混用或收尾用 `aclshmem_quiet`。`fence` 当前实现等价于全引擎 `quiet`。

### 3.4 Internal / Deprecated API

【描述】custom-ops 新算子 **禁止** 调用 `aclshmemi_*`（见 [internal-api-boundary.md](../../shmem-ops-code-gen/references/internal-api-boundary.md)）。**禁止** 新增 `aclshmemx_barrier_all_vec`（用 `aclshmem_barrier_all`）。核间同步用 `AscendC::SyncAll`，勿用 `aclshmemi_barrier_core_soft`。

【错误做法】
```cpp
aclshmemi_barrier_core_soft();
aclshmemx_barrier_all_vec();
```

【正确做法】
```cpp
```

### item 9

https://raw.githubusercontent.com/hicann/cannbot-skills/master/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/device-profiling-guide.md
hicann/cannbot-skills, repo pushed 2026-10-19

```markdown
以下展示通信算子典型模式（信号-搬运分离）。

#### 4.3.1 信号-搬运分离模式

root PE 将数据搬运到对称堆并通知 receiver，非 root PE 等待 signal 后从对称堆读取：

```cpp
const int my_pe = aclshmem_my_pe();

// frame 0: root 的 copy_in —— 将 input 搬运到 symmetric buffer
if (my_pe == root_pe) {
    SHMEMI_PROF_START(0);
    int64_t done = 0;
    while (done < local_len) {
        uint32_t copy_elems = std::min((uint32_t)(local_len - done), ub_elems);
        aclshmemx_mte_put_nbi(gva_data + start + done, input + start + done,
                              tmp_buf, ub_size, copy_elems, root_pe, EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        aclshmemx_signal_op(gva_sync + flag_offset, magic + chunk_seq,
                            ACLSHMEM_SIGNAL_SET, root_pe);
        done += copy_elems;
    }
    SHMEMI_PROF_END(0);
}

// frame 2: read + finalize（外层包裹）
// frame 1: signal_wait（内层嵌套，仅包裹同步等待）
SHMEMI_PROF_START(2);
int64_t done = 0;
while (done < local_len) {
    SHMEMI_PROF_START(1);
    aclshmem_signal_wait_until(source_signal, ACLSHMEM_CMP_GE, magic + chunk_seq);
    SHMEMI_PROF_END(1);
    uint32_t copy_elems = std::min((uint32_t)(local_len - done), ub_elems);
    aclshmemx_mte_get_nbi(output + start + done, gva_data + start + done,
                          tmp_buf, ub_size, copy_elems, root_pe, EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    done += copy_elems;
}
SHMEMI_PROF_END(2);
```
```

## ITEM 13 — LCCL (Ascend low-latency collective lib, `ascend-transformer-boost` / `src/kernels/lcal`) ONE-SHOT ALLREDUCE kernel, complete. Its entire inter-rank synchronisation is three lines: `sync.SetInnerFlag(magic, 1); sync.WaitRankInnerFlag(magic, 1, rank); sync.WaitRankInnerFlag(magic, 1, rankIDOfBlock);` — no host barrier, no zeroing of the flag region.

### item 10

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/ascendc_kernels/allreduce_one_shot.h
Huawei Technologies Co., Ltd. (file header "Copyright (c) 2024"), mirrored in GitHub repo egretwAlker/cpl-ascend-transformer-boost; exact commit date unknown

```cpp
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALLREDUCE_ONE_SHOT_H
#define LCCL_ALLREDUCE_ONE_SHOT_H

#include "sync_collectives.h"
#include "allreduce_quant.h"

using namespace AscendC;
template <typename T, typename U = T>
class AllReduceOneShot : protected AllReduceQuant {
    constexpr static T oneCast = (T) 1;

public:
    FORCE_INLINE_AICORE AllReduceOneShot(int rank, int rankSize, uint32_t extraFlag)
        : AllReduceQuant(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
        if constexpr(!std::is_same_v<T, U>) {
            BuildScaleOffset(scale, scaleCount, offset);
        }
        atomOp = op;
        blockNum = blockNum / rankSize * rankSize;
        if (blockIdx >= blockNum) {
            DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
            return;
        }

        corePerRank = blockNum / rankSize;
        rankIDOfBlock = blockIdx / corePerRank;

        dataDMAPerCore = len / rankSize / corePerRank / scaleNum * scaleNum;
        dataReducePerCore = len / corePerRank / scaleNum * scaleNum;

        blockDataNum = dataDMAPerCore;
        if (blockIdx == rankSize * corePerRank - 1) {
            blockDataNum = len - blockIdx * dataDMAPerCore;
        }

        blockReduceNum = dataReducePerCore;
        if (blockIdx % corePerRank == corePerRank - 1) {
            blockReduceNum = len - blockIdx % corePerRank * dataReducePerCore;
        }

        __gm__ U* curRankGm = (__gm__ U*)shareAddrs[rank] + IPC_DATA_OFFSET / sizeof(U);
        __gm__ U* peerRankGm = (__gm__ U*)shareAddrs[rankIDOfBlock] + IPC_DATA_OFFSET / sizeof(U);
        __gm__ U* intputGm = (__gm__ U*)input;
        __gm__ T* outputGm = (__gm__ T*)output;

        srcInputGlobal.SetGlobalBuffer(intputGm + blockIdx * dataDMAPerCore);
        dstIPCGlobal.SetGlobalBuffer(curRankGm + blockIdx * dataDMAPerCore);
        copyOutputGlobal.SetGlobalBuffer(outputGm + blockIdx * dataDMAPerCore);
        srcIPCGlobal.SetGlobalBuffer(peerRankGm + blockIdx % corePerRank * dataReducePerCore);
        dstOutputGlobal.SetGlobalBuffer(outputGm + blockIdx % corePerRank * dataReducePerCore);
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
    }

    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        if (blockIdx >= blockNum) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        CpInputToBuffAndOutput();
        sync.SetInnerFlag(magic, 1);

        sync.WaitRankInnerFlag(magic, 1, rank);
        sync.WaitRankInnerFlag(magic, 1, rankIDOfBlock);
        if (rankIDOfBlock != rank) {
            if constexpr (!std::is_same_v<T, U>) {
                if (!isEnableScale) {
                    Collectives::CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp);
                } else if (!isVectorScale) {
                    CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp, firstScale, offset);
                } else {
                    CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp, scaleGt, scaleNum, offset);
                }
            } else {
                Collectives::CpGM2GM<T, U>(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp);
            }
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }

    FORCE_INLINE_AICORE void CpInputToBuffAndOutput()
    {
        Collectives::CpGM2GM<U>(dstIPCGlobal, srcInputGlobal, blockDataNum, COPYONLY);
        if constexpr (!std::is_same_v<T, U>) {
            if (!isEnableScale) {
                Collectives::CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY);
            } else if (!isVectorScale) {
                CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY, firstScale, offset);
            } else {
                CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY, scaleGt, scaleNum, offset);
            }
        } else {
            Collectives::CpGM2GM<T, U>(copyOutputGlobal, srcInputGlobal, blockDataNum, -1);
        }
    }

protected:
    GlobalTensor<U> srcInputGlobal;
    GlobalTensor<U> srcIPCGlobal;
    GlobalTensor<U> dstIPCGlobal;
    GlobalTensor<T> dstOutputGlobal;
    GlobalTensor<T> copyOutputGlobal;

    int rankIDOfBlock;
    int corePerRank;
    int dataDMAPerCore;
    int dataReducePerCore;
    int blockDataNum;
    int blockReduceNum;
    int atomOp;
    GlobalTensor<T> scaleGt;
    int64_t scaleNum = 1;
    T firstScale = 1;
    T offset = 0;
    bool isEnableScale = false;
    bool isVectorScale = false;

private:
    FORCE_INLINE_AICORE void BuildScaleOffset(GM_ADDR scale, int64_t scaleCount, GM_ADDR offset)
    {
        if (scale != nullptr && offset != nullptr) {
            this->offset =* reinterpret_cast<__gm__ T*>(offset);
            scaleGt.SetGlobalBuffer((__gm__ T*)scale);
            this->firstScale = scaleGt.GetValue(0);
            this->scaleNum = scaleCount < 1 ? 1 : scaleCount;
            isVectorScale = scaleCount > 1;
            isEnableScale = scaleCount > 0 && !(*(uint16_t *)(&(this->offset)) == 0 &&
                scaleCount == 1 && *(uint16_t *)(&firstScale) == *(uint16_t *)(&oneCast));
        }
    }
};

#endif // LCCL_ALLREDUCE_ONE_SHOT_H```

## ITEM 14 — LCCL `op_def.h` `SET_MAGIC` macro: the DEVICE-SIDE epoch generator. Core 0 does an atomic-add of 1 into `commArgs->magics[rankSize - 1]` via `SetAtomicOpType<int32_t>(Op::ADD)` + `DataCopyPad`, then every core reads it back after `SyncAll<true>()`. `GET_IPC_MEM_ARGS` then uses `(magic % PING_PONG_SIZE)` to pick which half of the IPC window this generation writes into. This is the exact flag-free / epoch-tagged mechanism with no host-side barrier.

### item 11

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/ascendc_kernels/op_def.h
Huawei / ascend-transformer-boost LCCL, date unknown (mirror)

```cpp
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_OP_DEF_H
#define LCCL_OP_DEF_H
#define GET_COMM_ARGS \
    GlobalTensor<int> commArgsGm; \
    commArgsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(commArgs), 5); \
    int rank = commArgsGm.GetValue(0); \
    int localRank = commArgsGm.GetValue(1); \
    int rankSize = commArgsGm.GetValue(2); \
    int localRankSize = commArgsGm.GetValue(3); \
    uint32_t extraFlag = commArgsGm.GetValue(4); \
    GM_ADDR dumpAddr = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->dumpAddr; \
    int32_t lcalBlockNum = GetBlockNum()

#ifdef ENABLE_LCCL_MIX
#define SET_MAGIC \
do { \
    __gm__ CommArgs * commArgsTmp = reinterpret_cast<__gm__ CommArgs *>(commArgs); \
    PipeBarrier<PIPE_ALL>(); \
    SetAtomicNone(); \
    SetMaskNormImpl(); \
    SetSyncBaseAddr(commArgsTmp->fftsVal); \
    SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1); \
    PipeBarrier<PIPE_ALL>(); \
    LocalTensor<int32_t> localSet; \
    localSet.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN); \
    localSet.address_.bufferAddr = reinterpret_cast<uint64_t>((__ubuf__ int32_t *)96); \
    GlobalTensor<int32_t> magicGt; \
    magicGt.SetGlobalBuffer((__gm__ int32_t *)commArgsTmp->magics); \
    if (GetBlockIdx() == 0) { \
        SetAtomicOpType<int32_t>(Op::ADD); \
        localSet.SetValue(0, 1); \
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0); \
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0); \
        DataCopyExtParams dataCopyParams(1, sizeof(int32_t), 0, 0, 0); \
        DataCopyPad(magicGt[rankSize - 1], localSet, dataCopyParams); \
        AscendC::SetAtomicNone(); \
        PipeBarrier<PIPE_ALL>(); \
    } \
    SyncAll<true>(); \
    DataCopyExtParams dataCopyParams(1, sizeof(int32_t), 0, 0, 0); \
    DataCopyPadExtParams<int32_t> padParams; \
    DataCopyPad(localSet, magicGt[rankSize - 1], dataCopyParams, padParams); \
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0); \
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0); \
    magic = static_cast<int64_t>(localSet.GetValue(0)); \
    PipeBarrier<PIPE_ALL>(); \
    constexpr int32_t aivNumPerAic = 2; \
    lcalBlockNum = GetBlockNum() * aivNumPerAic; \
} while (0)
#else
#define SET_MAGIC \
do {} while (0)
#endif

#define GET_IPC_MEM_ARGS(type) \
do { \
    SET_MAGIC; \
    GlobalTensor<GM_ADDR> peerMemsAddrGm; \
    peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ CommArgs *>(commArgs))->peerMems[0], \
                       LCAL_MAX_RANK_SIZE); \
    for (int i = 0; i < rankSize; ++i) { \
        shareAddrs[i] = (__gm__ type *) (peerMemsAddrGm.GetValue(i) + \
                        (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET)); \
    } \
    AscendC::PipeBarrier<PIPE_ALL>(); \
} while (0) \

#define CLASS_OP_LAUNCH(name, type) \
do { \
    name<type> opKernel(rank, rankSize, extraFlag); \
    opKernel.Init(KERNELS_ARGS_CALL()); \
    opKernel.Process();                 \
} while (0)

#define CLASS_OP_QUANT_910A5_LAUNCH(name, outputType, addType, inputType) \
do { \
    name<outputType, addType, inputType> opKernel(rank, rankSize, extraFlag); \
    opKernel.Init(KERNELS_ARGS_CALL()); \
    opKernel.Process();                 \
} while (0)

```

## ITEM 15 — LCCL `SyncCollectives` (the ATB/LCCL original of which vllm-ascend's is a fork): `MAGIC_OFFSET = 32`, `MAGIC_MASK`, `MergeMagicWithValue`, and the stale-rejection predicate in `WaitOneRankPartFlag`: `if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue)` — i.e. the epoch must match exactly AND the in-epoch value must have reached the target. Includes the `breakCycle` bounded-spin variant absent from the vllm-ascend fork.

### item 12

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/ascendc_kernels/sync_collectives.h
Huawei / ascend-transformer-boost LCCL, date unknown (mirror)

```cpp
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_SYNC_H
#define LCCL_SYNC_H

#include "comm_args.h"

using namespace AscendC;
using namespace Lcal;

constexpr int64_t FLAG_UNIT_INT_NUM = 4;
constexpr int64_t SYNC_UNIT_SIZE = FLAG_UNIT_INT_NUM * sizeof(int64_t);
constexpr int64_t MAGIC_OFFSET = 32;
constexpr int64_t MAGIC_MASK = ~((1LL << MAGIC_OFFSET) - 1);
#ifdef ENABLE_LCCL_MIX
constexpr int32_t LCAL_BLOCK_NUM_MULTI = 2;
#else
constexpr int32_t LCAL_BLOCK_NUM_MULTI = 1;
#endif

class SyncCollectives {
public:
    __aicore__ inline SyncCollectives() {}

    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR *shareAddrs)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum() * LCAL_BLOCK_NUM_MULTI;
        segmentCount = GetBlockNum() * LCAL_BLOCK_NUM_MULTI * FLAG_UNIT_INT_NUM;
        localSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]);
        basicSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + segmentCount + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        TPipe pipe;
        pipe.InitBuffer(tBuf, GetBlockNum() * SYNC_UNIT_SIZE);
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag(localSyncAddr + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline int32_t CalEventIdByMulBlockNum(int32_t blockMultiplier, int32_t targetCoreId)
    {
        return (blockMultiplier * blockNum) + targetCoreId;
    }

    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank,
        int32_t breakCycle = 0)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v, breakCycle);
    }

    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(basicSyncAddr, value);
    }
    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag((__gm__ int64_t*)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
    }

    __aicore__ inline void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }

    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    __aicore__ inline bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        return CheckOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(blockOuterSyncAddr, value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        __gm__ int64_t *flagAddr = GetOuterFlagAddr(setRank, setBlock);
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(flagAddr, value);
    }

    __aicore__ inline void WaitOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr = GetOuterFlagAddr(waitRank, waitBlock);
        WaitOneRankPartFlag(flagAddr, 1, value);
    }

    __aicore__ inline void WaitOneRankOuterFlag(int32_t magic, int32_t eventID, int64_t rank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr;
        flagAddr = GetOuterFlagAddr(rank, 0);
        WaitOneRankPartFlag(flagAddr, blockNum, value);
    }
    __aicore__ inline void WaitAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            WaitOneRankPartFlag(flagAddr, flagNum, value);
        }
    }

    __aicore__ inline bool CheckAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock,
        int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            if (!CheckOneRankPartFlag(flagAddr, flagNum, value)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void WaitAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        WaitAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    __aicore__ inline bool CheckAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        return CheckAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(1, 0);
        localSet.SetValue(0, setValue);
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        tBuf.FreeTensor(localSet);
    }

    __aicore__ inline void WaitFlag(__gm__ int64_t* waitAddr, int64_t waitValue)
    {
        WaitOneRankPartFlag(waitAddr, 1, waitValue);
    }

    __aicore__ inline int64_t GetFlag(__gm__ int64_t* waitAddr)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(1, 0);
        DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int64_t res = localWait.GetValue(0);
        tBuf.FreeTensor(localWait);
        return res;
    }

    __aicore__ inline void WaitOneRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank,
                                                    int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr;
        flagAddr = GetOuterFlagAddr(waitRank, startBlock);
        WaitOneRankPartFlag(flagAddr, flagNum, value);
    }

    __aicore__ inline int64_t GetInnerFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM);
    }

    __aicore__ inline int64_t GetOuterFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + segmentCount + waitBlock * FLAG_UNIT_INT_NUM);
    }

private:
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value)
    {
        return (static_cast<int64_t>(magic) << MAGIC_OFFSET) | static_cast<int64_t>(value);
    }

    __aicore__ inline __gm__ int64_t* GetInnerFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline __gm__ int64_t* GetOuterFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + segmentCount + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue,
        int32_t breakCycle = 0)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        do {
            if (breakCycle > 0) {
                int64_t systemCycleBefore = AscendC::GetSystemCycle();
                int64_t systemCycleAfter = AscendC::GetSystemCycle();
                while (systemCycleAfter - systemCycleBefore < breakCycle) {
                    systemCycleAfter = AscendC::GetSystemCycle();
                };
            }
            DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            isSync = true;
            for (auto i = 0; i < flagNum; ++i) {
                int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    break;
                }
            }
        } while (!isSync);
        tBuf.FreeTensor(localWait);
    }

    __aicore__ inline void WaitOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        WaitOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    __aicore__ inline bool CheckOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        bool isSync = true;
        for (auto i = 0; i < flagNum; ++i) {
            int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
            if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                isSync = false;
                break;
            }
        }
        tBuf.FreeTensor(localWait);
        return isSync;
    }

    __aicore__ inline bool CheckOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        return CheckOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    int rank;
    int rankSize;
    int blockIdx;
    int blockNum;
    GM_ADDR *shareAddrs;
    int64_t segmentCount;
    __gm__ int64_t* localSyncAddr;
    __gm__ int64_t* basicSyncAddr;
    __gm__ int64_t* blockOuterSyncAddr;
    TBuf<QuePosition::VECCALC> tBuf;
};

#endif // LCCL_SYNC _H```

## ITEM 16 — LCCL host side: `magic` is simply a per-communicator counter `int64_t magic_ = 1;` post-incremented on EVERY collective call and passed as a kernel scalar argument (`KERNELS_ARGS_FUN()` declares `int64_t magic`). `CommArgs` carries `int32_t magics[LCAL_MAX_RANK_SIZE]` in device memory for the mix-mode device-side variant. `Collectives::Init` offsets every peer window by `(magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET)`.

### item 13

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/lccl.cpp
Huawei / ascend-transformer-boost LCCL, date unknown (mirror)

```cpp
    if ((comm_->commArgs_.extraFlag & ExtraFlag::QUANT_FP16) != 0 &&
        (comm_->commArgs_.extraFlag & (ExtraFlag::QUANT_DELAY | ExtraFlag::QUANT_CURRENT)) == 0) {
        uint32_t blockDim = GetBlockNum(LcalType::ALL_REDUCE, rankSize_, Count2Size(count, dataType),
                                        comm_->localRankSize_, comm_->commArgs_.extraFlag);
        AscendCCLKernelArgs ascendArgs = {sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, op, 0, 0,
            scale, scaleCount, offset};
        comm_->magic_++;
        return LoadMTE(LcalType::ALL_REDUCE, ascendArgs, blockDim, dataType, stream);
    }

    if (rankSize_ <= 1) {
        return LoopBack(sendBuff, recvBuff, count, dataType, stream);
    }

    if ((comm_->commArgs_.extraFlag & (ExtraFlag::QUANT_DELAY | ExtraFlag::QUANT_CURRENT)) != 0) {
        uint32_t blockDim = GetBlockNum(LcalType::ALL_REDUCE, rankSize_, Count2Size(count, dataType),
                                        comm_->localRankSize_, comm_->commArgs_.extraFlag);
        AscendCCLKernelArgs args = { sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, op, 0, 0, scale,
            scaleCount};
        comm_->magic_++;
        return LoadMTE(LcalType::ALL_REDUCE, args, blockDim, dataType, stream);
    }

    uint32_t blockDim = GetBlockNum(LcalType::ALL_REDUCE, rankSize_, Count2Size(count, dataType),
                                    comm_->localRankSize_, comm_->commArgs_.extraFlag);
    AscendCCLKernelArgs args = {sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, op, 0};
    comm_->magic_++;
    return LoadMTE(LcalType::ALL_REDUCE, args, blockDim, dataType, stream);
}
```

### item 14

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/ascendc_kernels/collectives.h
Huawei / ascend-transformer-boost LCCL — kernel arg list + `Init` ping-pong window selection + dfx recording of magic

```cpp
using namespace Lcal;

#define KERNELS_ARGS_FUN() \
GM_ADDR input, GM_ADDR output, GM_ADDR commArgs, int64_t len, int64_t magic, int op, int root, int cycleCount, \
GM_ADDR scale, int64_t scaleCount, GM_ADDR offset

#define KERNELS_ARGS_CALL() \
input, output, commArgs, len, magic, op, root, cycleCount, scale, scaleCount, offset

#define KERNELS_GATHER_TABLE_ARGS_FUN() \
GM_ADDR embTable, GM_ADDR lookup, GM_ADDR revData, int64_t lookupLen, int64_t embTableLen, int64_t embTableDim

#define KERNELS_GATHER_TABLE_ARGS_CALL() \
embTable, lookup, revData, lookupLen, embTableLen, embTableDim

enum DfxPos : int {
    MAGIC,
    LEN,
    RUN_STATUS
};

class Collectives {
    constexpr static int32_t UB_HEAD_OFFSET = 96;
    constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + ALIGN_SIZE;
public:
    FORCE_INLINE_AICORE Collectives(int rank, int rankSize, uint32_t extraFlag) : rank(rank), rankSize(rankSize),
        extraFlag(extraFlag) {}

    FORCE_INLINE_AICORE ~Collectives()
    {
        const int64_t notRunning = 0xdead;
        dfx.SetValue(RUN_STATUS, notRunning);
    }

    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        dumpAddr_ = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->dumpAddr;
        GlobalTensor<GM_ADDR> peerMemsAddrGm;
        peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ CommArgs *>(commArgs))->peerMems[0],
                                        LCAL_MAX_RANK_SIZE);
        for (int i = 0; i < rankSize; ++i) {
            shareAddrs[i] = peerMemsAddrGm.GetValue(i) +
                            (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
        }
        dfx.SetGlobalBuffer((reinterpret_cast<__gm__ CommArgs *>(commArgs))->dfx,
            DFX_COUNT);
        this->root = root;
        this->len = len;
        this->magic = magic;
        this->localRank = reinterpret_cast<__gm__ CommArgs *>(commArgs)->localRank;
        this->localRankSize = reinterpret_cast<__gm__ CommArgs *>(commArgs)->localRankSize;
        this->xRankSize = localRankSize;
        this->yRankSize = rankSize / localRankSize;
        this->xRankIdx = rank % localRankSize;
        this->yRankIdx = rank / localRankSize;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum() * LCAL_BLOCK_NUM_MULTI;

        sync.Init(rank, rankSize, shareAddrs);
        dfx.SetValue(MAGIC, magic);
        dfx.SetValue(LEN, len);
        const int64_t running = 0xbeef;
        dfx.SetValue(RUN_STATUS, running);
    }

    template <typename T>
```

### item 15

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/include/comm_args.h
Huawei / ascend-transformer-boost LCCL — `CommArgs` with the device-resident `magics[]` array (Chinese comments verbatim)

```cpp
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_COMM_ARGS_H
#define LCCL_COMM_ARGS_H
#include <cstdint>

#if !defined(__DAV_C220_VEC__) && !defined(__DAV_C310__) && !defined(__DAV_C220_CUBE__)
using GM_ADDR = uint8_t*;
#else
#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__
#include "kernel_operator.h"
#endif
namespace Lcal {

constexpr int LCAL_MAX_RANK_SIZE = 128; // lcal通信库最大支持的npu卡数
constexpr int RANK_SIZE_TWO = 2;  // 可用SIO的规模，以及是否需要跨卡搬运数据核的分界规模
constexpr int64_t IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024; // 前2MB作为flag标志位，之后100MB作为数据存储
constexpr int64_t SYNC_FLAG_BIT_NUM = 10;  // cce 算子在用
constexpr int64_t MEM_DMA_UNIT_INT_NUM = 4;
constexpr int64_t EVENT_ID_MASK = 0xFFFFFFFF;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;

struct CommArgs {
    int rank = 0;           // attr rank_id, global rank
    int localRank = -1;
    int rankSize = 0; // global rank size
    int localRankSize = -1;  // 此参数是指fullmesh互联的卡数
    uint32_t extraFlag = 0; // 32 bit map，具体每一位的含义就在此文件正上方
    GM_ADDR peerMems[LCAL_MAX_RANK_SIZE] = {}; // 传入初始化获得的buff，所有allreduce都是同一个参数
    /**
     * @param sendCountMatrix 大小是rankSize*rankSize的一维数组
     * eg: sendCountMatrix[1] 的数值，对应二维数组的[0][1]，表示 卡0 要给 卡1 发送的数据个数
     */
    int64_t sendCountMatrix[LCAL_MAX_RANK_SIZE * LCAL_MAX_RANK_SIZE] = {}; // for all2allv
    int64_t dfx[DFX_COUNT] = {};
    GM_ADDR dumpAddr = nullptr;
    int32_t magics[LCAL_MAX_RANK_SIZE] = {0};
    uint64_t fftsVal = 0;
};

struct LcclDumpBlockInfo {
    uint32_t len = 0;
```

## ITEM 17 — sgl-kernel-npu `CamMoeCombineNormal`: a 1-bit TOGGLE epoch stored in the symmetric window itself. `InitMagic()` reads the per-core magic slot, writes back its complement, and the magic then selects which half of the window this invocation uses (`winDataSizeOffset_ = magic_ * ((totalWinSize_ - 4*STATE_SIZE) / 2)`). Note also the orthogonal "status is a float 1.0 (0x3F800000) summed to a target then zeroed" flag protocol in `SetStatusBySrcInfo` / `WaitBuffCopy` — that one DOES reset, in contrast to the magic.

### item 16

https://raw.githubusercontent.com/sgl-project/sgl-kernel-npu/main/csrc/deepep/ops2/op_kernel/cam_moe_combine_normal.h
sgl-project/sgl-kernel-npu, repo main branch as of 2026-10-20

```cpp
constexpr uint32_t TOKEN_IDX_OFFSET_IN_SRC_INFO = 1U;
constexpr uint32_t TOPK_IDX_OFFSET_IN_SRC_INFO = 2U;
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 8UL * 1024UL * 1024UL;
constexpr uint64_t MAGIC_WIN_OFFSET = 975UL * 1024UL;
constexpr uint32_t TOKEN_SRC_INFO_LEN = 3U;
constexpr uint32_t UB_32_ALIGN = 32U;
constexpr uint32_t MUL_256_ALIGN = 256U;

// ---- InitMagic (lines 136-147): read-complement-writeback toggle epoch ----
template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::InitMagic()
{
    GlobalTensor<int32_t> selfMagicTensor;
    selfMagicTensor.SetGlobalBuffer((__gm__ int32_t *)(hccl_.GetWindowsInAddr(epRankId_) + totalWinSize_ -
                                                       Moe::STATE_SIZE + MAGIC_WIN_OFFSET + coreIdx_ * WIN_512_ALIGN));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
    magic_ = selfMagicTensor(0);
    selfMagicTensor(0) = ((magic_ == 0) ? 1 : 0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
}


// ---- Init (lines 210-218): magic picks the window half ----
    InitGlobalBuffer(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, XOut, sendCostStatsOut);
    InitBuffLen();

    PipeBarrier<PIPE_ALL>();
    winDataSizeOffset_ = static_cast<uint64_t>(magic_) * ((totalWinSize_ - 4 * Moe::STATE_SIZE) / 2UL);
    localRankGM_ = GetBufferAddrByRankId(epRankId_);
    DataCacheCleanAndInvalid<SrcInfoType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        epRecvCountGM_[moeExpertNum_ - 1]);
    selfSendCnt_ = epRecvCountGM_(moeExpertNum_ - 1);

// ---- status-sum flag protocol (lines 236-238, 304-352) ----
    LocalTensor<uint32_t> statusTensor = stateBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(statusTensor, 0x3F800000, FLOAT_NUM_PER_ALIGN);


template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::SetStatusBySrcInfo(uint32_t srcRankId,
                                                                                    uint32_t srcTokenId,
                                                                                    uint32_t srcTopkId)
{
    LocalTensor<uint32_t> statusTensor = stateBuf_.Get<uint32_t>();
    GM_ADDR stateGM = GetStateAddrByRankId(srcRankId) + (srcTokenId * axisK_ + srcTopkId) * UB_32_ALIGN;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);
    DataCopy<uint32_t>(stateGMTensor, statusTensor, FLOAT_NUM_PER_ALIGN);
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::WaitBuffCopy(uint32_t tokenIndex,
                                                                              uint32_t startTokenIndex)
{
    LocalTensor<int32_t> tokenIdxLocal = tokenIdxBuf_.Get<int32_t>();
    int tempValidCount = 0;
    for (int topkId = 0; topkId < axisK_; ++topkId) {
        int expertId = tokenIdxLocal.GetValue((tokenIndex - startTokenIndex) * axisK_ + topkId);
        if (expertId < 0 || expertId >= moeExpertNum_) {
            continue;
        }
        ++tempValidCount;
    }
    uint32_t calCount = axisK_ * FLOAT_NUM_PER_ALIGN;
    GM_ADDR stateGM = GetStateAddrByRankId(epRankId_) + tokenIndex * axisK_ * UB_32_ALIGN;  // 计算地址偏移
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float *)stateGM);
    float current = (float)0.0;
    float target = (float)1.0 * tempValidCount * FLOAT_NUM_PER_ALIGN;
    SumParams sumPerKParams{1, calCount, calCount};
    LocalTensor<float> stateTensorLocal = stateBuf_.Get<float>();
    LocalTensor<float> tempStateTensorLocal = tempStateBuf_.Get<float>();
    while (current != target) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(stateTensorLocal, stateGMTensor, calCount);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(tempStateTensorLocal, stateTensorLocal, sumPerKParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        current = tempStateTensorLocal(0);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(tempStateTensorLocal, (float)0.0, calCount);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy<float>(stateGMTensor, tempStateTensorLocal, calCount);
}

template <TemplateMC2TypeClass>
```

## ITEM 18 — LCCL legacy `.cce` collectives: the bare `SetFlag` / `CheckFlag` / `CheckFlagNew` / `SyncWithinNPU` primitives where the flag value IS the magic itself (no merge, no mask). `CheckFlagNew` accepts `checkValue || checkValue + 1` — the same "peer already advanced one epoch" tolerance as shmem's `aclshmemi_signal_wait_until_eq_for_barrier`. Also note `MODIFIABLE_MAGIC_*_ARGS_CALL` macros that re-parameterise the magic for chunked multi-round launches.

### item 17

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/kernels/collectives.cce
Huawei / ascend-transformer-boost LCCL, file header "Copyright (c) 2024 Huawei Technologies Co., Ltd."; mirror commit date unknown

```cpp

#define ALLREDUCE_ARGS_FUN(T) \
__gm__ T *input, __gm__ T *output, int rank, int rankSize, int64_t len, int64_t magic, int op, int root, \
int localRankSize, int64_t loopTime, __gm__ int64_t *sendCountMatrix, GM_ADDR dumpAddr,      \
__gm__ T *buff0, __gm__ T *buff1, __gm__ T *buff2, __gm__ T *buff3, __gm__ T *buff4,\
__gm__ T *buff5, __gm__ T *buff6, __gm__ T *buff7

#define ALLREDUCE_ARGS_CALL(type) \
(__gm__ type *)input,  (__gm__ type *) output, rank, rankSize, len, \
magic, op, root, localRankSize, 0, nullptr, dumpAddr, shareAddrs[0], shareAddrs[1], shareAddrs[2], \
shareAddrs[3], shareAddrs[4], shareAddrs[5], shareAddrs[6], shareAddrs[7]

#define ALLREDUCE_ARGS_FUN_16P(T) \
__gm__ T *input, __gm__ T *output, int rank, int rankSize, int64_t len, int64_t magic, int op, int root, \
int localRankSize, int64_t loopTime, __gm__ int64_t *sendCountMatrix, GM_ADDR dumpAddr,          \
__gm__ T *buff0, __gm__ T *buff1, __gm__ T *buff2, __gm__ T *buff3, __gm__ T *buff4,            \
__gm__ T *buff5, __gm__ T *buff6, __gm__ T *buff7, __gm__ T *buff8, __gm__ T *buff9,            \
__gm__ T *buff10, __gm__ T *buff11, __gm__ T *buff12, __gm__ T *buff13, __gm__ T *buff14, __gm__ T *buff15

#define ALLREDUCE_ARGS_CALL_16P(type) \
(__gm__ type *)input,  (__gm__ type *) output, rank, rankSize, len, \
magic, op, root, localRankSize, 0, nullptr, dumpAddr, shareAddrs[0], shareAddrs[1], shareAddrs[2], \
shareAddrs[3], shareAddrs[4], shareAddrs[5], shareAddrs[6], shareAddrs[7], shareAddrs[8], shareAddrs[9], \
shareAddrs[10], shareAddrs[11], shareAddrs[12], shareAddrs[13], shareAddrs[14], shareAddrs[15] \

#define ALLREDUCE_ARGS_FUN_16P_Origin(T) \
__gm__ T *input, __gm__ T *output, int rank, int rankSize, int64_t len, int64_t magic, int op, int root, \
int localRankSize, __gm__ int64_t *sendCountMatrix, GM_ADDR dumpAddr, __gm__ T* buff[MAX_RANK_NUM_OF_ONE_910B2C]

#define ALLREDUCE_ARGS_CALL_16P_Origin() \
input, output, rank, rankSize, len, magic, op, root, localRankSize, sendCountMatrix, dumpAddr, buff

#define MODIFIABLE_MAGIC_PROCESSED_NUM_ALLREDUCE_ARGS_CALL_16P_Origin(processedNum, remainNum, magic) \
(input + (processedNum)), (output + (processedNum)), rank, rankSize, (remainNum), (magic), op, root, \
localRankSize, sendCountMatrix, dumpAddr, buff

#define MODIFIABLE_MAGIC_ALLREDUCE_ARGS_CALL_16P(magic) \
input, output, rank, rankSize, len, (magic), op, root, localRankSize, sendCountMatrix, dumpAddr, \
buff0, buff1, buff2, buff3, buff4, buff5, buff6, buff7, buff8, buff9, buff10, buff11, \
buff12, buff13, buff14, buff15

__attribute__((always_inline)) inline __aicore__ int64_t CeilDiv(int64_t source, int64_t cardinality)
{

// ---- flag primitives (lines 137-236) ----

__attribute__((always_inline)) inline __aicore__ void SetFlag(__ubuf__ int64_t *ctrlFlagsUB, __gm__ int64_t *ctrlFlagGM,
    int64_t checkValue)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    *ctrlFlagsUB = checkValue;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1); 
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
    CpUB2GM(ctrlFlagGM, ctrlFlagsUB, sizeof(int64_t));
    AscendC::PipeBarrier<PIPE_ALL>();
}

__attribute__((always_inline)) inline __aicore__ void SetFlagNonPipeBarrier(__ubuf__ int64_t *ctrlFlagsUB, __gm__ int64_t *ctrlFlagGM,
    int64_t checkValue)
{
    *ctrlFlagsUB = checkValue;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    CpUB2GM(ctrlFlagGM, ctrlFlagsUB, sizeof(int64_t));
}

__attribute__((always_inline)) inline __aicore__ void SetFlag(__ubuf__ int64_t *ctrlFlagsUB,
    __gm__ int64_t *ctrlFlagGM1, __gm__ int64_t *ctrlFlagGM2, int64_t checkValue)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    *ctrlFlagsUB = checkValue;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    CpUB2GM(ctrlFlagGM1, ctrlFlagsUB, sizeof(int64_t));
    CpUB2GM(ctrlFlagGM2, ctrlFlagsUB, sizeof(int64_t));
    AscendC::PipeBarrier<PIPE_ALL>();
}

__attribute__((always_inline)) inline __aicore__ void SetFlagNonPipeBarrier(__ubuf__ int64_t *ctrlFlagsUB,
    __gm__ int64_t *ctrlFlagGM1, __gm__ int64_t *ctrlFlagGM2, int64_t checkValue)
{
    *ctrlFlagsUB = checkValue;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    CpUB2GM(ctrlFlagGM1, ctrlFlagsUB, sizeof(int64_t));
    CpUB2GM(ctrlFlagGM2, ctrlFlagsUB, sizeof(int64_t));
}

__attribute__((always_inline)) inline __aicore__ void CheckFlag(__ubuf__ int64_t *ctrlFlagsUB,
    __gm__ int64_t *ctrlFlagGM, int64_t checkValue)
{
    while (true) {
        AscendC::PipeBarrier<PIPE_ALL>();
        CpGM2UB(ctrlFlagsUB, ctrlFlagGM, sizeof(int64_t));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (*ctrlFlagsUB == checkValue) {
            break;
        }
    }
}

__attribute__((always_inline)) inline __aicore__ void CheckFlagNew(__ubuf__ int64_t *ctrlFlagsUB,
    __gm__ int64_t *ctrlFlagGM, int64_t checkValue)
{
    while (true) {
        AscendC::PipeBarrier<PIPE_ALL>();
        CpGM2UB(ctrlFlagsUB, ctrlFlagGM, sizeof(int64_t));
        AscendC::PipeBarrier<PIPE_ALL>();
        if (*ctrlFlagsUB == checkValue || (*ctrlFlagsUB) == (checkValue + 1)) {
            break;
        }
    }
}

__attribute__((always_inline)) inline __aicore__ int64_t GetLcalBlockNum() {
    #ifdef ENABLE_LCCL_MIX
    constexpr int32_t aivNumPerAic = 2;
    return GetBlockNum() * aivNumPerAic;
    #else
    return GetBlockNum();
    #endif
}

__attribute__((always_inline)) inline __aicore__ void SyncWithinNPU(__ubuf__ int64_t* ctrlFlagsUB, __gm__ int64_t* buffRank, int64_t magic) {
    SetFlag(ctrlFlagsUB, (__gm__ int64_t*)buffRank + (GetBlockIdx() * MEM_DMA_UNIT_INT_NUM), magic);
    for (int64_t i = 0; i < GetLcalBlockNum(); i++) {
        if (i == GetBlockIdx()) {
            continue;
        }
        CheckFlag((__ubuf__ int64_t*)ctrlFlagsUB, (__gm__ int64_t*)buffRank + i * MEM_DMA_UNIT_INT_NUM, magic);
    }
}

__attribute__((always_inline)) inline __aicore__ void SyncWithinNPUNew(__ubuf__ int64_t* ctrlFlagsUB, __gm__ int64_t* buffRank, int64_t magic) {
    SetFlag(ctrlFlagsUB, (__gm__ int64_t*)buffRank + (GetBlockIdx() * MEM_DMA_UNIT_INT_NUM), magic);
    for (int64_t i = 0; i < GetLcalBlockNum(); i++) {
        if (i == GetBlockIdx()) {
            continue;
        }
        CheckFlagNew((__ubuf__ int64_t*)ctrlFlagsUB, (__gm__ int64_t*)buffRank + i * MEM_DMA_UNIT_INT_NUM, magic);
    }
}

template<typename T>
```

## ITEM 19 — CANN shmem official atomic/AMO semantics doc (Chinese): full per-op, per-engine, per-PLATFORM support matrix. Decisive for A3 (dav-2201): only `atomic_add` is listed as `A2/A3/950` (via "UB + MTE3", types int8/int16/bf16/half/int32/float). `atomic_set`, `atomic_swap`, `atomic_compare_swap`, `atomic_fetch_add`, `atomic_fetch_inc`, `atomic_inc`, and all bitwise AMOs are marked 950-only. Also documents which standard AMOs insert `PipeBarrier<PIPE_ALL>` and which are pass-through async.

### item 18

https://raw.githubusercontent.com/Zhangyq-1999/shmem-doc/main/_sources/api/atomic_api_sync_async_comparison.md.txt
Zhangyq-1999/shmem-doc (rendered CANN shmem documentation mirror), repo pushed 2026-08-01

```markdown
# ACLSHMEM Atomic 原子操作接口用法语义总结

> 本文档基于实际代码实现编写，覆盖各层接口：标准接口 `aclshmem_<type>_atomic_<op>`、MTE 后端 `aclshmemx_mte_atomic_<op>`、RDMA 后端 `aclshmemx_roce_atomic_<op>`。

---

## 一、接口层次关系

atomic 接口分为两层：

```text
aclshmem_<type>_atomic_<op>()          ← 标准接口
    │
    ├── topo == MTE  → aclshmemx_mte_atomic_<op>()   ← MTE 后端
    └── topo == ROCE → aclshmemx_roce_atomic_<op>()  ← RDMA 后端
```

标准接口根据 `device_state->topo_list[pe]` 自动分派到 MTE 或 RDMA 后端，并在后端调用之上额外添加同步保证。

---

## 二、MTE 后端接口：`aclshmemx_mte_atomic_<op>`

> 头文件：`include/device/gm2gm/engine/shmem_device_mte.h`
> 实现文件：`src/device/gm2gm/engine/shmem_device_mte.hpp`

**核心特征：所有 MTE 后端接口均为异步**，不论是否带返回值。接口仅发起硬件原子操作即返回，无 quiet/fence。

同步原语：`aclshmemx_mte_quiet()` 或 Kernel Runtime 级 `SetFlag/WaitFlag`。

### 2.1 接口清单

| 接口完整名 | 返回值 | 支持类型 | 硬件平台 |
| --- | --- | --- | --- |
| `aclshmemx_mte_atomic_fetch` | T | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_set` | void | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_swap` | T | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_compare_swap` | T | uint32, uint64 | 950 |
| `aclshmemx_mte_atomic_inc` | void | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_add` | void | int8, int16, bf16, half, int32, float (A2/A3/950) + uint32, uint64, int64 (950) | A2/A3/950 |
| `aclshmemx_mte_atomic_fetch_inc` | T | int32, uint32, float, int64, uint64 | 950 |
| `aclshmemx_mte_atomic_fetch_add` | T | int32, uint32, float, int64, uint64 | 950 |

### 2.2 同步要求

所有 MTE 后端接口调用后，如果存在**标量计算单元（Scalar）与搬运单元（MTE2/MTE3）之间的数据依赖**，开发者需根据实际数据流插入同步：

| 时机 | 场景 | 同步方式 |
| --- | --- | --- |
| 调用前 | 其他单元（如 MTE2）正在写同一 UB 区域 | `SetFlag/WaitFlag` on `MTE2_S` |
| 调用后 | 后续操作依赖原子操作在 GM 中的结果 | `SetFlag/WaitFlag` on `MTE3_MTE2` |
| 调用后 | 需要复用同一 UB 区域写入新数据 | `SetFlag/WaitFlag` on `MTE3_S` |

### 2.3 `aclshmemx_mte_atomic_add` 的类型差异（重要）

`aclshmemx_mte_atomic_add` 在 Ascend950 上按类型有两种实现路径：

| 类型组 | 支持类型 | 实现方式 | 同步要点 |
| --- | --- | --- | --- |
| UB + MTE3 路径 | int8, int16, bf16, half, int32, float | Scalar 写操作数到 UB → MTE3 读 UB → MTE3 atomic add 写 GM | 需关注 UB 相关 fence（MTE2_S、MTE3_S）和 GM fence（MTE3_MTE2） |

| UB + MTE3 路径 | int8, int16, bf16, half, int32, float | Scalar 写操作数到 UB → MTE3 读 UB → MTE3 atomic add 写 GM | 需关注 UB 相关 fence（MTE2_S、MTE3_S）和 GM fence（MTE3_MTE2） |
| Scalar AtomicAdd 路径 | uint32, uint64, int64 | Scalar 单元直接执行标量 AtomicAdd 写 GM，不经过 UB 和 MTE3 | 不需要 UB/MTE3 相关 fence；仅在 Scalar↔MTE2/MTE3 有 GM 数据依赖时按通用规则同步 |

---

## 三、RDMA 后端接口：`aclshmemx_roce_atomic_<op>`

> 头文件：`include/device/gm2gm/engine/shmem_device_rdma.h`
> 实现文件：`src/device/gm2gm/engine/shmem_device_rdma.hpp`

**核心特征：有返回值的接口为同步接口（内部 quiet），无返回值的接口为异步接口（需调用者 quiet）。**

同步原语：`aclshmemx_roce_quiet(pe, buf, sync_id)`。

RDMA AMO 的 target/source 参数会作为本地对称地址换算到指定 PE，因此必须指向对称内存；不能把任意
本地 GM 地址直接用作该参数。可通过 `aclshmem_malloc` 等对称内存分配接口准备目标对象。

### 3.1 接口清单（同步接口——内部完成 quiet）

| 接口完整名 | 返回值 | 支持类型 | 硬件平台 |
| --- | --- | --- | --- |
| `aclshmemx_roce_atomic_fetch` | T | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_fetch_add` | T | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_fetch_inc` | T | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_fetch_and` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_fetch_or` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_fetch_xor` | T | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_compare_swap` | T | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_swap` | T | 32/64 位整数 | 950 |

> 以上接口在返回前内部调用 `aclshmemx_roce_quiet`，**调用者无需额外同步**。

### 3.2 接口清单（异步接口——需调用者 quiet）

| 接口完整名 | 返回值 | 支持类型 | 硬件平台 |
| --- | --- | --- | --- |
| `aclshmemx_roce_atomic_set` | void | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_add` | void | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_inc` | void | 32/64 位整数 | 950 |
| `aclshmemx_roce_atomic_and` | void | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_or` | void | int32, uint32, int64, uint64 | 950 |
| `aclshmemx_roce_atomic_xor` | void | int32, uint32, int64, uint64 | 950 |

> 以上接口仅发起操作即返回，**调用者必须在依赖结果前调用 `aclshmemx_roce_quiet(pe, buf, sync_id)`**。

### 3.3 并发限制

> **警告**：使用 RDMA 作为底层传输时，不支持对同一 PE 的并发 RMA/AMO 操作。请使用 `device_state.rdma_config` 中的 `sync_id` 进行流水线同步。

---

## 四、标准接口：`aclshmem_<type>_atomic_<op>`

> 头文件：`include/device/gm2gm/shmem_device_amo.h`
> 实现文件：`src/device/gm2gm/shmem_device_amo.hpp`

标准接口根据拓扑自动分派到 MTE 或 RDMA 后端，并在后端调用之上**额外添加同步保证**。

标准 AMO 接口会将用户传入的 `dst`、`dest` 或 `source` 交给 `aclshmem_ptr(..., pe)` 换算为指定 PE 上的
目标地址，因此这些参数必须指向对称内存。该要求来自标准接口的地址换算方式，并不表示 MTE 引擎本身
只能访问对称内存；传输层自动分派不会改变这一接口语义。

```text
aclshmem_<type>_atomic_<op>(dst, val, pe)
    │
    ├── topo_list[pe] & MTE
    │   ├── fetch 变体:    调用 aclshmemx_mte_atomic_fetch_<op>() + PipeBarrier<PIPE_ALL>
    │   ├── swap/cas:      调用 aclshmemx_mte_atomic_swap/compare_swap() + PipeBarrier<PIPE_ALL>
    │   ├── add/inc:       透传 aclshmemx_mte_atomic_add/inc()，未加 PipeBarrier
    │   └── set/and/or/xor: 透传底层异步接口，未加 PipeBarrier
    │
    └── topo_list[pe] & ROCE
        ├── fetch 变体:    透传 aclshmemx_roce_atomic_fetch_<op>()（后端已 quiet）
        ├── swap/cas:      透传 aclshmemx_roce_atomic_swap/compare_swap()（后端已 quiet）
        ├── add/inc:       透传 aclshmemx_roce_atomic_add/inc()（后端异步，未加 quiet）
        └── set/and/or/xor: 透传底层异步接口（未加 quiet）


## 六、各标准接口支持的类型与硬件平台

### 6.1 `aclshmem_<type>_atomic_add`

| 传输通路 | 支持类型 | MTE 实现方式 | 硬件平台 |
|---|---|---|---|
| MTE | int8, int16, bf16, half, int32, float | UB + MTE3 | A2/A3/950 |
| MTE | uint32, uint64, int64 | Scalar AtomicAdd | 950 |
| RDMA | int32, uint32, int64, uint64 | — | 950 |

### 6.2 `aclshmem_<type>_atomic_fetch_add`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | int32, uint32, uint64, int64, float | 950 |
| RDMA | int32, uint32, int64, uint64 | 950 |

### 6.3 `aclshmem_<type>_atomic_inc`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | int32, uint32, uint64, int64 | 950 |
| RDMA | int32, uint32, int64, uint64, float | 950 |

### 6.4 `aclshmem_<type>_atomic_fetch_inc`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | int32, uint32, uint64, int64, float | 950 |
| RDMA | int32, uint32, int64, uint64 | 950 |

### 6.5 位运算系列 (`atomic_and/or/xor` 及 fetch 变体)

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| RDMA | int32, uint32, int64, uint64 | 950 |

> **注意**：位运算接口不支持 MTE 通路。

### 6.6 `aclshmem_<type>_atomic_fetch`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

### 6.7 `aclshmem_<type>_atomic_set`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

> CAST 变体（`ACLSHMEM_TYPE_FUNC_ATOMIC_SWAP_CAST`）：int32→uint32、int64→uint64、float→uint32 通过类型转换实现。

### 6.8 `aclshmem_<type>_atomic_swap`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | uint32, uint64, int32, int64, float | 950 |
| RDMA | uint32, uint64, int32, int64, float | 950 |

> CAST 变体同上。

### 6.9 `aclshmem_<type>_atomic_compare_swap`

| 传输通路 | 支持类型 | 硬件平台 |
|---|---|---|
| MTE | uint32, uint64, int32, int64 | 950 |
| RDMA | uint32, uint64, int32, int64 | 950 |

> CAST 变体（`ACLSHMEM_TYPE_FUNC_ATOMIC_CAS_CAST`）：int32→uint32、int64→uint64。

---

## 七、跨 PCIe（跨节点）限制

**MTE 通路不支持跨 PCIe**。跨节点场景请使用 RDMA 通路。

---

## 八、代码示例
```

## ITEM 20 — CURRENT official CANN shmem (gitcode `cann/shmem`, master) signal API docs, which differ from the older GitHub mirror: the atomicity caveat has been REWORDED from "WARNING: Atomicity NOT Guaranteed." to "On Ascend950, RDMA is used as the communication engine, supporting signal_set with guaranteed atomicity." The host-side variant additionally documents the cross-machine matrix: SIGNAL_SET supports RDMA cross-machine; SIGNAL_ADD does NOT, and only supports MTE cross-machine when HCCS is available.

### item 19

https://raw.gitcode.com/cann/shmem/raw/master/include/device/gm2gm/shmem_device_p2p_sync.h
cann/shmem on gitcode.com (official CANN SIG repo), file header "Copyright (c) 2025 Huawei Technologies Co., Ltd."; fetched 2026-10-20

```cpp
#ifndef SHMEM_DEVICE_P2P_SYNC_H
#define SHMEM_DEVICE_P2P_SYNC_H

#include "host_device/shmem_common_types.h"
#include "gm2gm/shmem_device_p2p_sync.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The aclshmemx_signal_op operation updates sig_addr with signal using operation sig_op on the specified PE.
 *        This operation can be used together with aclshmem_signal_wait_until for efficient
 *        point-to-point synchronization.
 *        On Ascend950, RDMA is used as the communication engine, supporting signal_set with guaranteed atomicity.
 *
 * @param sig_addr              [in] Symmetric address of the signal word to be updated.
 * @param signal                [in] The value used to update sig_addr.
 * @param sig_op                [in] Operation used to update sig_addr with signal. Supported operations:
 *                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                    [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);
#define shmemx_signal_op aclshmemx_signal_op

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                                   ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @return Return the contents of the signal data object, sig_addr, at the calling PE that satisfies the wait condition.
 */
ACLSHMEM_DEVICE int32_t aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val);
#define shmem_signal_wait_until aclshmem_signal_wait_until

```

### item 20

https://raw.gitcode.com/cann/shmem/raw/master/include/host/data_plane/shmem_host_so.h
cann/shmem on gitcode.com, fetched 2026-10-20

```cpp
 * @param pe                [in] PE number of the remote PE.
 * @note The data transfer path uses MTE only.
 */
ACLSHMEM_HOST_API void aclshmemx_putmem_signal(
    void* dst, void* src, size_t elem_size, void* sig_addr, int32_t signal, int sig_op, int pe);
#define shmem_putmem_signal aclshmemx_putmem_signal

/**
 * @brief This routine performs an atomic operation on a remote signal variable at the specified PE, with the operation
 *        executed on the given stream. It is used to modify a remote signal and is typically paired with wait routines
 *        like aclshmemx_signal_wait_until_on_stream.
 *
 * @param sig_addr              [in] Symmetric address of the signal variable to update on the target PE.
 * @param signal                [in] The value to be used in the atomic operation.
 * @param sig_op                [in] The operation to perform on the remote signal. Supported operations:
 *                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD.
 * @param pe                    [in] The PE number on which the remote signal variable is to be updated.
 * @param stream                [in] Used stream(use default stream if stream == NULL).
 *
 * @note Cross-machine support:
 *       - ACLSHMEM_SIGNAL_SET: Supports RDMA cross-machine.
 *       - ACLSHMEM_SIGNAL_ADD: Does not support RDMA cross-machine. Supports MTE cross-machine when HCCS is available.
 */
ACLSHMEM_HOST_API void aclshmemx_signal_op_on_stream(
    int32_t* sig_addr, int32_t signal, int sig_op, int pe, aclrtStream stream);
#define shmem_signal_op_on_stream aclshmemx_signal_op_on_stream

#ifdef __cplusplus
}
#endif

#endif
```

## ITEM 21 — CANN shmem `kv_shuffle` example: the cleanest "monotonically incremented, never reset" flag. The sync buffer is `aclrtMemset`-zeroed ONCE in the constructor and never again; `count_` is a host-side per-invocation counter (`count_++` before each launch) passed as kernel scalar `int32_t count`; the device writes `count` into the peer's per-(rank, AIV) slot with `ACLSHMEM_SIGNAL_SET` and waits `ACLSHMEM_CMP_EQ, count`. Flag slot address = `(rank * aiv_num + aiv_idx) * SYNC_FLAG_INTERVAL` with `SYNC_FLAG_INTERVAL = 16` int32 (64B stride) to avoid false sharing.

### item 21

https://raw.githubusercontent.com/immengzi/shmem/master/examples/kv_shuffle/kv_shuffle_kernel.cpp
immengzi/shmem (CANN shmem mirror), repo pushed 2026-03-26

```cpp
using namespace AscendC;
using fp16_t = op::fp16_t;
using bf16_t = op::bfloat16;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;

extern "C" __global__ __aicore__ void ShmemKVShuffle(
    uint64_t fftsAddr,
    GM_ADDR k_cache,
    GM_ADDR v_cache,
    GM_ADDR global_shuffle_table,
    GM_ADDR src_block_table,
    GM_ADDR dst_block_table,
    GM_ADDR sync_ptr,
    int64_t block_nums,
    int64_t kv_head_num, int64_t page_size, int64_t head_dim, int32_t count) {
    util_set_ffts_config(fftsAddr);

    int64_t n_pes = aclshmem_n_pes();
    int64_t local_rank = aclshmem_my_pe();
    int64_t pair_rank = *((__gm__ int64_t*)global_shuffle_table + 2 * local_rank);
    int64_t operation = *((__gm__ int64_t*)global_shuffle_table + 2 * local_rank + 1);
    int64_t pair_operation = *((__gm__ int64_t*)global_shuffle_table + 2 * pair_rank + 1);
    
    __gm__ int32_t* gva_sync_gm = (__gm__ int32_t *)sync_ptr;

    const int64_t aiv_num = AscendC::GetBlockNum();
    const int64_t aiv_idx = AscendC::GetBlockIdx();
    
    const int64_t local_rank_offset = (local_rank * aiv_num + aiv_idx) * SYNC_FLAG_INTERVAL;
    const int64_t pair_rank_offset = (pair_rank * aiv_num + aiv_idx) * SYNC_FLAG_INTERVAL;

    uint64_t copy_ping_k_ub = 0;
    uint64_t copy_pong_k_ub = 32 * 1024;
    uint64_t copy_ping_v_ub = 64 * 1024;
    uint64_t copy_pong_v_ub = 96 * 1024;
    uint32_t copy_ub_size = 32 * 1024;

    // 0 means send
    if (operation == 0) {
        aclshmem_signal_wait_until(gva_sync_gm + pair_rank_offset, ACLSHMEM_CMP_EQ, count);

// ---- (lines 113-158): send/recv handshake tail + host-side count_ ----
            }

            ping_pong_flag = 1 - ping_pong_flag;
        }
        aclshmemx_signal_op(gva_sync_gm + local_rank_offset, count, ACLSHMEM_SIGNAL_SET, pair_rank);
    } else if (pair_operation == 0) {
        aclshmemx_signal_op(gva_sync_gm + local_rank_offset, count, ACLSHMEM_SIGNAL_SET, pair_rank);
        aclshmem_signal_wait_until(gva_sync_gm + pair_rank_offset, ACLSHMEM_CMP_EQ, count);
    }
}

KVShuffleOps::KVShuffleOps(uint32_t block_dims, void* stream) : count_(0), block_dims_(block_dims), stream_(stream)
{
    fftsAddr_ = util_get_ffts_config();
    sync_ptr_ = aclshmem_malloc(sizeof(int32_t) * aclshmem_n_pes() * block_dims * SYNC_FLAG_INTERVAL);
    aclrtMemset(sync_ptr_, sizeof(int32_t) * aclshmem_n_pes() * block_dims * SYNC_FLAG_INTERVAL,
                0, sizeof(int32_t) * aclshmem_n_pes() * block_dims * SYNC_FLAG_INTERVAL);
}

KVShuffleOps::~KVShuffleOps()
{
    aclshmem_free(sync_ptr_);
}
    
    // 接受 Tensor 的函数
void KVShuffleOps::compute(
    uint8_t* k_cache,
    uint8_t* v_cache,
    uint8_t* global_shuffle_table,
    uint8_t* src_block_table,
    uint8_t* dst_block_table,
    int64_t block_nums,
    int64_t kv_head_num, int64_t page_size, int64_t head_dim)
{
    count_++;
    ShmemKVShuffle<<<block_dims_, nullptr, stream_>>>(
        fftsAddr_,
        k_cache,
        v_cache,
        global_shuffle_table,
        src_block_table,
        dst_block_table,
        (uint8_t *)sync_ptr_,
        block_nums,
        kv_head_num, page_size, head_dim, count_);
}
```

## ITEM 22 — CANN shmem AMO_ADD semantics in a runnable test: `aclshmemx_signal_op_on_stream(sig_addr, add_value, ACLSHMEM_SIGNAL_ADD, next_rank, stream)` then `aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_EQ, rank_id + 10, stream)` — the signal slot starts at `rank_id` (NOT zero) and the expected value is derived by arithmetic, i.e. ADD accumulates onto whatever was there. Chinese comments verbatim.

### item 22

https://raw.githubusercontent.com/immengzi/shmem/master/tests/unittest/host/sync/signal/signal_host_test.cpp
immengzi/shmem (CANN shmem mirror), repo pushed 2026-03-26

```cpp
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    int32_t *sig_addr = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    EXPECT_NE(sig_addr, nullptr);
    
    // 所有rank初始化为rank_id
    int init_value = rank_id;
    EXPECT_EQ(aclrtMemcpy(sig_addr, sizeof(int32_t), &init_value, sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE), 0);
    
    int32_t host_value;
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] before atomicadd, host_value = " << host_value << std::endl;
    std::cout << "[TEST] signal ADD test rank " << rank_id << " of " << n_ranks << std::endl;

    int next_rank = (rank_id + 1) % n_ranks;
    
    // 每个rank都向后继rank的信号加一个唯一值
    int32_t add_value = 10;  
    aclshmemx_signal_op_on_stream(sig_addr, add_value, ACLSHMEM_SIGNAL_ADD, next_rank, stream);
    std::cout << "[RANK " << rank_id << "] Add " << add_value << " to rank " << next_rank << std::endl;
    
    sleep(1);
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] after add, host_value = " << host_value << std::endl;
    int32_t expected = rank_id + 10;
    
    // 每个rank都等待自己的信号达到expected
    aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_EQ, expected, stream);
    std::cout << "[RANK " << rank_id << "] Waiting for == " << expected << std::endl;
    
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    
    // 验证结果
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    // 所有rank验证累加结果
    EXPECT_EQ(host_value, expected);
    std::cout << "[RANK " << rank_id << "] Verified: " << host_value << " == " << expected << std::endl;

    aclshmem_free(sig_addr);
    test_finalize(stream, device_id);
```

### item 23

https://raw.githubusercontent.com/mkbk-with-circle/ascend_inc_dispatch_combine/main/shmem/tests/unittest/device/sync/p2p/p2p_kernel.cpp
mkbk-with-circle/ascend_inc_dispatch_combine (vendored CANN shmem tree), repo pushed 2026-09-14 — in-kernel p2p chain handshake and the full set of typed `wait_until` / `wait_until_all` comparison modes exercised

```cpp

#define ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(FUNC)                                                                          \
    FUNC(float, float, 10.0f);                                                                                         \
    FUNC(int8, int8_t, 10);                                                                                            \
    FUNC(int16, int16_t, 10);                                                                                          \
    FUNC(int32, int32_t, 10);                                                                                          \
    FUNC(int64, int64_t, 10);                                                                                          \
    FUNC(uint8, uint8_t, 10);                                                                                          \
    FUNC(uint16, uint16_t, 10);                                                                                        \
    FUNC(uint32, uint32_t, 10);                                                                                        \
    FUNC(uint64, uint64_t, 10);                                                                                        \
    FUNC(char, char, 'b')

extern "C" ACLSHMEM_GLOBAL void p2p_chain(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    util_set_ffts_config(config);
    auto sig_addr = (__gm__ int32_t *)addr;
    int32_t val = *sig_addr;
    int next = (rank_id + 1) % rank_size;

    aclshmem_barrier_all();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    if (rank_id == 0) {
        aclshmemx_signal_op(sig_addr, 1, ACLSHMEM_SIGNAL_SET, next);
        aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_EQ, 1);
    } else {
        aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_EQ, 1);
        aclshmemx_signal_op(sig_addr, 1, ACLSHMEM_SIGNAL_SET, next);
    }
#endif

    aclshmem_barrier_all();
}

void p2p_chain_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)
{
    p2p_chain<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size);
}


#define P2P_CHAIN_WAIT_UNTIL(NAME, TYPE, CMP_VAL)                                                                      \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until(uint64_t config, GM_ADDR addr,                       \
        int rank_id, int rank_size)                                                                                    \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_EQ, CMP_VAL);                                                  \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_NE, (CMP_VAL) - 1);                                            \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);                                            \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_GE, CMP_VAL);                                                  \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);                                            \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);                                            \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_LE, CMP_VAL);                                                  \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);                                            \
                                                                                                                       \
        aclshmem_##NAME##_wait(ivar, (CMP_VAL) - 1);                                                                   \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL);

#define P2P_CHAIN_DO_WAIT_UNTIL(NAME, TYPE, CMP_VAL)                                                                   \
    void p2p_chain_do_##NAME##_wait_until(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)    \
    {                                                                                                                  \
        p2p_chain_##NAME##_wait_until<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size);                       \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL);

#define P2P_CHAIN_WAIT_UNTIL_ALL(NAME, TYPE, CMP_VAL)                                                                  \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until_all(uint64_t config, GM_ADDR addr,                   \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status)                                           \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_EQ, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_NE, (CMP_VAL) - 1);                        \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);                        \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_GE, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);                        \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);                        \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_LE, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);                        \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL_ALL);

```

## ITEM 23 — sgl-kernel-npu `NotifyDispatch::GetMagicValue()`: the minimal device-side epoch generator — read the per-block magic slot out of the symmetric window's STATE region (with `DataCacheCleanAndInvalid<..., DcciDst::CACHELINE_OUT>` to bypass stale cache), clamp to >= 1, write back `magic + 1`, return the OLD value. No atomics needed because exactly one core owns each slot. `magic % PING_PONG_SIZE` then picks the window half. Note also this file's alternative merge encoding `MergeMagicWithValue(magic, value) = magic * 2 + value` (1 bit of in-epoch payload instead of 32).

### item 24

https://raw.githubusercontent.com/sgl-project/sgl-kernel-npu/main/csrc/deepep/ops2/op_kernel/notify_dispatch.h
sgl-project/sgl-kernel-npu, main branch as of 2026-10-20

```cpp
// Assign values to winContext_[COMM_EP_IDX] and blockIdx before calling
template <typename T>
__aicore__ inline uint64_t NotifyDispatch<T>::GetMagicValue(void)
{
    uint64_t magic = 0;
    GlobalTensor<uint64_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = hccl_.GetWindowsInAddr(rank) + totalWinSize_ - Moe::STATE_SIZE * 3;
    selfDataStatusTensor.SetGlobalBuffer((__gm__ uint64_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[blockIdx * UB_ALIGN_SIZE]);
    magic = selfDataStatusTensor(blockIdx * UB_ALIGN_SIZE);
    if (magic <= 0) {
        magic = 1;
    }
    selfDataStatusTensor(blockIdx * UB_ALIGN_SIZE) = magic + 1;
    return magic;
}


// ---- Init (lines 955-966): magic selects window half ----
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    hccl_.InitV2(contextGM0, &tilingData);
    hccl_.SetCcTilingV2(offsetof(NotifyDispatchTilingData, mc2CcTiling1));

    winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)contextGM0;
    this->magic = GetMagicValue();
    ctxIdx = COMM_EP_IDX;
    uint64_t winDataOffset = ((totalWinSize_ - 4 * Moe::STATE_SIZE) / 2) * (this->magic % PING_PONG_SIZE);

    shareAddrs[rank] = GetWindAddrByRankId(rank, ctxIdx) + winDataOffset;


// ---- merge + wait (lines 240-300) ----
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // copy input to other rank share
    __aicore__ inline void InputToShareSlice()
    {
        __ubuf__ uint64_t *inputUB = (__ubuf__ uint64_t *)get_imm(0);
        int32_t copyOffset = blockIdx * rankNumPerCore;
        copyLen = rankSize - copyOffset < rankNumPerCore ? rankSize - copyOffset : rankNumPerCore;
        if (copyLen > 0) {
            readGt = sendDataInputGt[copyOffset * perRankDataNum];
            CpGM2GMPingPong<T>(copyLen * perRankDataNum * sizeof(T), readGt, writeGt[copyOffset * perRankDataNum],
                               COPYONLY);
            uint64_t v = MergeMagicWithValue(magic, 1);
            *inputUB = v;
            AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            for (int i = copyOffset; i < copyOffset + copyLen; ++i) {
                CpUB2GM((__gm__ uint64_t *)(shareAddrs[i]) + rank * FLAG_UNIT_INT_NUM, inputUB, sizeof(uint64_t));
            }
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline uint64_t MergeMagicWithValue(uint64_t magic, uint64_t value)
    {
        // magic as the high part, eventID as the low part, combined into a value for comparison
        return (magic * 2ULL + value);
    }

    // Wait for a part of synchronization flags within a rank
    __aicore__ inline void WaitOneRankPartFlag(__gm__ uint64_t *waitAddr, int64_t flagNum, uint64_t checkValue)
    {
        GlobalTensor<uint64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<uint64_t> localWait = tBuf.GetWithOffset<uint64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        uint64_t checkedFlagNum = 0;
        do {
            // Copy global synchronization flags to local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM],
                     (flagNum - checkedFlagNum) * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            // Check if the synchronization flags are equal to checkValue
            isSync = true;
            uint64_t remainToCheck = flagNum - checkedFlagNum;
            for (auto i = 0; i < remainToCheck; ++i) {
                // Continue waiting if any core has not reached the checkValue phase
                uint64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    checkedFlagNum += i;
                    break;
                }
            }
        } while (!isSync);
    }

```

## ITEM 24 — CANN cann-recipes-infer `ops/shmem/distributed_embedding`: real device-side AMO primitives in an AscendC/SIMT kernel — `Simt::AtomicCas`, `Simt::AtomicAdd`, `asc_atomic_inc`, and the shmem symmetric-memory CAS `simt::aclshmem_uint32_atomic_compare_swap` / `simt::aclshmem_int64_atomic_compare_swap`. Includes the comment on fixed-width type requirements for the device CAS overloads.

### item 25

https://raw.githubusercontent.com/hicann/cann-recipes-infer/master/ops/shmem/distributed_embedding/op_kernel/distributed_hashtable.cpp
hicann/cann-recipes-infer, master as of 2026-10-20

```cpp
    } else {
        simt::aclshmem_getmem(dst, src, sizeof(pair_type), pe);
    }
}

// The device atomic CAS overloads use the fixed-width C types. In particular,
// on this platform `long long` is a distinct type from `int64_t` even though
// both are 64 bits wide, so convert through the supported overload while
// preserving the key bit pattern.
template <typename Tkey>
__simt_callee__ inline Tkey AtomicCasKey(__gm__ Tkey *address, Tkey compare, Tkey value)
{
    static_assert(sizeof(Tkey) == sizeof(uint32_t) || sizeof(Tkey) == sizeof(uint64_t),
                  "AtomicCasKey supports only 32-bit and 64-bit keys");
    if constexpr (sizeof(Tkey) == sizeof(uint32_t)) {
        return static_cast<Tkey>(Simt::AtomicCas(
            reinterpret_cast<__gm__ uint32_t *>(address), static_cast<uint32_t>(compare),
            static_cast<uint32_t>(value)));
    } else {
        return static_cast<Tkey>(Simt::AtomicCas(
            reinterpret_cast<__gm__ int64_t *>(address), static_cast<int64_t>(compare),
            static_cast<int64_t>(value)));
    }
}

// MurMerHash3函数
template <typename Tkey>
__simt_callee__ inline uint32_t MurmurHash3(const __gm__ Tkey *key, int len, uint32_t seed)
{
    static_assert(sizeof(Tkey) == sizeof(uint32_t) || sizeof(Tkey) == sizeof(uint64_t),
                  "MurmurHash3 supports only 32-bit and 64-bit keys");

// ---- (lines 330-345, 420-440): AtomicAdd counters ----
    for (int64_t i = threadIdx + blockIdx * threadNum; i < hashTableSize; i += blockNum * threadNum)
    {
        if (hashTable[i].key != unusedKey)
        {
            localSize++;
        }
    }
    if (localSize > 0)
    {
        Simt::AtomicAdd(containerSize, localSize);
    }
}

// kernel入口
template <typename Tkey, typename Tvalue>
__global__ __vector__ void


    if (threadIdx == 0) { *blockAcc = 0; }
    if (idx >= searchLength) { return; }
    asc_syncthreads();

    pair_type pair = LoadPairVectorized<Tkey, Tvalue>(tableHandle + offset + idx);
    if (pair.key != unusedKey) {
        uint32_t localIndex = Simt::AtomicAdd(blockAcc, static_cast<uint32_t>(1));
        pUbKey[localIndex] = pair.key;
        pUbValue[localIndex] = pair.value;
    }
    asc_syncthreads();

    if (threadIdx == 0) {
        *globalAcc = Simt::AtomicAdd(dDumpCounter, *blockAcc);
    }
    asc_syncthreads();

    if (threadIdx < *blockAcc) {
        d_key[*globalAcc + threadIdx] = pUbKey[threadIdx];
        d_value[*globalAcc + threadIdx] = pUbValue[threadIdx];

// ---- (lines 700-760): symmetric-memory CAS via shmem ----
    uint32_t hashVal, uint64_t hostTblSize, uint32_t targetPe, Tkey unusedKey)
{
    size_t currIdx = static_cast<size_t>(hashVal) % hostTblSize;
    size_t counts = 0;
    while (true) {
        if (counts++ >= hostTblSize) {
            break;
        }
        // SHMEM SIMT CAS maps the HOST_SIDE symmetric address internally.
        Tkey oldKey;
        if constexpr (sizeof(Tkey) == sizeof(uint32_t)) {
            oldKey = static_cast<Tkey>(simt::aclshmem_uint32_atomic_compare_swap(
                reinterpret_cast<__gm__ uint32_t *>(&hostHashTable[currIdx].key),
                static_cast<uint32_t>(unusedKey), static_cast<uint32_t>(insertKey),
                static_cast<int32_t>(targetPe)));
        } else {
            oldKey = static_cast<Tkey>(simt::aclshmem_int64_atomic_compare_swap(
                reinterpret_cast<__gm__ int64_t *>(&hostHashTable[currIdx].key),
                static_cast<int64_t>(unusedKey), static_cast<int64_t>(insertKey),
                static_cast<int32_t>(targetPe)));
        }
        if (oldKey == unusedKey || oldKey == insertKey) {
            ShmemPutScalar<Tvalue>(&hostHashTable[currIdx].value, insertVal, static_cast<int32_t>(targetPe));
            break;
        }
        currIdx = (currIdx + 1) % hostTblSize;
    }
}

// Single-PE D2H consumes the original input arrays without receive-buffer staging.
template<typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtSinglePeInsert(
    __gm__ pair_type *devHashTable, __gm__ pair_type *hostHashTable,
    const __gm__ Tkey *keys, const __gm__ Tvalue *values, uint64_t keyNum,
    uint64_t tableSize, uint64_t hostTblSize, uint32_t myPe, Tkey unusedKey,
    uint32_t blockIdx, uint32_t blockNum)
{
    uint32_t threadIdx = Simt::GetThreadIdx();
    uint32_t threadNum = Simt::GetThreadNum();
    for (uint64_t idx = blockIdx * threadNum + threadIdx; idx < keyNum; idx += blockNum * threadNum) {
        Tkey key = keys[idx];
        if (key == unusedKey) {
            continue;
        }
        Tvalue value = values[idx];
        uint32_t hashVal = MurmurHash3<Tkey>(keys + idx, sizeof(Tkey), 0);
        uint64_t bucket = static_cast<uint64_t>(hashVal) % tableSize;
        for (uint64_t step = 0; step < tableSize; ++step) {
            Tkey oldKey = AtomicCasKey(&devHashTable[bucket].key, unusedKey, key);
            if (oldKey == unusedKey || oldKey == key) {
                devHashTable[bucket].value = value;
                break;
            }
            bucket = (bucket + 1) % tableSize;
        }
        // Preserve the full Host copy even when the Device table is full.
        InsertHostPair<Tkey, Tvalue>(hostHashTable, key, value, hashVal, hostTblSize, myPe, unusedKey);
    }
}

template<typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
```

## ITEM 25 — CANN cannbot-skills canonical AscendC shmem kernel TEMPLATE for a flag-free pure-communication op: symmetric buffer laid out as `[sync_flags: aivNum * SYNC_FLAG_INTERVAL int32] [data_buf ...]`, `magic` as a kernel scalar argument, sender writes `flag_val = chunk_count + magic` with `ACLSHMEM_SIGNAL_SET`, receiver waits `ACLSHMEM_CMP_GE, magic + 1`. This is the recommended skeleton, Chinese annotations verbatim.

### item 26

https://raw.githubusercontent.com/hicann/cannbot-skills/master/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/templates/communication/templates-kernel.md
hicann/cannbot-skills, repo pushed 2026-10-19

```cpp

using namespace AscendC;
using fp16_t = op::fp16_t;
using bf16_t = op::bfloat16;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;

// ==== 定制：按 design.md 的 schedule.phases 实现具体通信逻辑 ====
template <typename T>
ACLSHMEM_DEVICE void <op_name>_impl(uint64_t fftsAddr,
                                     __gm__ T *input,
                                     __gm__ T *output,
                                     __gm__ T *symmetric,
                                     int elements,
                                     int magic)
{
    util_set_ffts_config(fftsAddr);

    const int64_t aivNum = GetBlockNum();
    const int64_t aivIndex = GetBlockIdx();
    const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    // ---- 分核 ----
    // ==== 定制：按 schedule.core_partition 划分 sender / receiver / sync 核 ====
    // 示例：前半核为 sender（本地 → symmetric），后半核为 receiver（远端 symmetric → 本地）
    int core_group_num = aivNum / 2;
    bool is_sender = (aivIndex < core_group_num);

    // Symmetric buffer 布局
    __gm__ int32_t *sync_flags = (__gm__ int32_t *)symmetric;
    __gm__ T *data_buf = (__gm__ T *)(sync_flags + aivNum * SYNC_FLAG_INTERVAL);

    // UB 临时缓冲（ping-pong）
    __ubuf__ T *ping_buf = reinterpret_cast<__ubuf__ T *>(uint64_t(1024 + 32));
    __ubuf__ T *pong_buf = reinterpret_cast<__ubuf__ T *>(uint64_t(96 * 1024 + 32));

    uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t ub_elems = ub_size / sizeof(T);

    if (is_sender) {
        // ==== Phase: 本地 GM → symmetric (sender 核) ====
        // ==== 定制：按 design.md 的 phase 逻辑调整 put 目标和 chunk 策略 ====
        int64_t per_core = elements / core_group_num;
        int64_t my_offset = aivIndex * per_core;
        if (aivIndex == core_group_num - 1) {
            per_core = elements - my_offset;
        }

        int64_t remaining = per_core * sizeof(T);
        int64_t done_elems = 0;
        int64_t chunk_count = 0;

        while (remaining > 0) {
            uint32_t copy_elems = (remaining > ub_size) ? ub_elems : (remaining / sizeof(T));

            SHMEMI_PROF_START(0);  // copy_in phase
            aclshmemx_mte_put_nbi(data_buf + my_offset + done_elems,
                                  input + my_offset + done_elems,
                                  ping_buf, ub_size, copy_elems,
                                  my_rank, EVENT_ID0);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            SHMEMI_PROF_END(0);

            chunk_count++;
            int64_t flag_val = chunk_count + magic;
            aclshmemx_signal_op(sync_flags + flag_offset, flag_val,
                                ACLSHMEM_SIGNAL_SET, my_rank);

            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            done_elems += copy_elems;
            remaining -= copy_elems * sizeof(T);
        }
    } else {
        // ==== Phase: 远端 symmetric → 本地 output (receiver 核) ====
        // ==== 定制：按 design.md 的 peer_model 和 addressing 调整 get 来源 ====
        int receiver_idx = aivIndex - core_group_num;
        // 示例：每个 receiver 核从一个远端 PE 拉取数据
        int64_t src_pe = receiver_idx;  // ==== 定制：映射逻辑 ====

        // 等待远端 sender 完成后 get
        // ==== 定制：按 schedule.phases 的 dependency 和 sync 调整等待逻辑 ====
        aclshmem_signal_wait_until(
            (__gm__ int32_t *)aclshmem_ptr(sync_flags, src_pe) + flag_offset,
            ACLSHMEM_CMP_GE, magic + 1);

        int64_t per_core = elements;
        int64_t dst_offset = src_pe * elements;
        int64_t remaining = per_core * sizeof(T);
        int64_t done_elems = 0;
        int pingpong = 0;

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        while (remaining > 0) {
            uint32_t copy_elems = (remaining > ub_size / 2) ? (ub_size / 2 / sizeof(T))
                                                            : (remaining / sizeof(T));
            TEventID eid = (pingpong == 0) ? EVENT_ID0 : EVENT_ID1;
            __ubuf__ T *buf = (pingpong == 0) ? ping_buf : pong_buf;

            WaitFlag<HardEvent::MTE3_MTE2>(eid);

            SHMEMI_PROF_START(1);  // remote_get phase
            aclshmemx_mte_get_nbi(output + dst_offset + done_elems,
                                  data_buf + done_elems,
                                  buf, ub_size / 2, copy_elems,
                                  src_pe, eid);
            SHMEMI_PROF_END(1);

            SetFlag<HardEvent::MTE3_MTE2>(eid);

            done_elems += copy_elems;
            remaining -= copy_elems * sizeof(T);
            pingpong = 1 - pingpong;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

// ---- Kernel 入口（按 dtype 展开）----
// ==== 定制：按 design.md 的 dtypes 调整支持的类型 ====
#define <OP_NAME>_FUNC_DEF(type)                                                    \
    extern "C" __global__ __aicore__ void Shmem<OpName>_##type(                     \
        uint64_t fftsAddr, GM_ADDR input, GM_ADDR output,                           \
        GM_ADDR symmetric, int elements, int magic)                                 \
    {                                                                               \
        <op_name>_impl<type>(fftsAddr, (__gm__ type *)input,                        \
                             (__gm__ type *)output, (__gm__ type *)symmetric,        \
                             elements, magic);                                      \
    }

<OP_NAME>_FUNC_DEF(int32_t);
<OP_NAME>_FUNC_DEF(float16_t);
<OP_NAME>_FUNC_DEF(bfloat16_t);

// ---- Host-callable wrapper ----
template <class T>
void <op_name>_kernel(uint32_t block_dim, void *stream, uint64_t fftsAddr,
                      uint8_t *input, uint8_t *output, uint8_t *symmetric,
                      int elements, int magic)
```

## ITEM 26 — CANN cannbot-skills `api.md` §5 "Sync：signal、wait/test、barrier、quiet/fence": the authoritative one-paragraph Chinese contract for the signal family, including the non-atomicity caveat and the distinction between `aclshmem_sync` (local stores only) and `aclshmem_barrier` (collective)

### item 27

https://raw.githubusercontent.com/hicann/cannbot-skills/master/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/api.md
hicann/cannbot-skills, repo pushed 2026-10-19

```markdown
## 5. Sync：signal、wait/test、barrier、quiet/fence

### 5.1 signal 与 wait/test

signal 用于点对点完成通知：

- `aclshmemx_signal_op` 在 Device 侧更新远端 `sig_addr`，支持 `ACLSHMEM_SIGNAL_SET` 和 `ACLSHMEM_SIGNAL_ADD`，但该接口自身不保证一般意义上的原子性。
- `aclshmem_put_signal` 和 `aclshmem_put_signal_nbi` 将数据搬运与 signal 更新合并，适合“数据到达后通知远端”的模式。
- `aclshmem_signal_wait_until` 按 `ACLSHMEM_CMP_EQ/NE/GT/GE/LT/LE` 等条件等待本地 signal 满足条件。
- typed wait/test 系列支持单变量、all/any/some wait set、vector compare 等复杂等待模式。
- Host 侧有 `aclshmemx_signal_op_on_stream`、`aclshmemx_signal_wait_until_on_stream`，用于把 signal 操作放入 stream 顺序中。

### 5.2 barrier 和 sync

barrier 能力同时存在于 Host 和 Device：

- `aclshmem_barrier(team)`、`aclshmem_barrier_all()` 对 team 或 world 做集合同步。
- `aclshmem_sync(team)`、`aclshmem_sync_all()` 只保证此前本地 memory stores 的完成和可见性，不保证远端 RMA 更新完成。
- Host 侧提供 stream 版本 `aclshmemx_barrier_on_stream`、`aclshmemx_barrier_all_on_stream`。
注意事项：

| 要点 | 说明 |
| --- | --- |
| Device barrier 前提 | 要求 MIX kernel；纯 AIV kernel 或缺少有效 cube/vector 指令时可能被优化成不满足要求的 kernel 形态 |
| deprecated 接口 | `aclshmemx_barrier_vec`/`aclshmemx_barrier_all_vec` 已标记 deprecated；新代码用 `aclshmem_barrier`/`aclshmem_barrier_all` |
| Host vs Device 完成域 | CPU 发起的 barrier 只完成 CPU 通信；NPU 发起的 barrier 只完成 NPU 通信。Host 等待 kernel 内通信完成应结合 `aclrtSynchronizeStream`/`aclshmemx_handle_wait`/stream 版本接口 |

### 5.3 quiet / fence

Device 侧 memory ordering 能力：

- `aclshmem_quiet` 确保调用 PE 之前发出的**全引擎** symmetric data object 操作完成（MTE + SDMA + UDMA + RDMA）。
- **单引擎场景 SHOULD 使用 engine-specific quiet**：MTE → `aclshmemx_mte_quiet`；SDMA → `aclshmemx_sdma_quiet`；UDMA → `aclshmemx_udma_quiet(pe)`；RDMA/RoCE → `aclshmemx_roce_quiet`。勿在仅需 MTE drain 时调用全局 `aclshmem_quiet`。
- `aclshmem_fence` 在当前实现中与 `aclshmem_quiet` 等价，既保证顺序也保证完成。
- HCCS-only 场景下完成后可认为全局可见；HCCS + RDMA 场景下，当前说明只保证对目标 PE 内存的可见性。
- 亦可配合 event/flag（如 `SetFlag/WaitFlag<MTE3_S>`）或 Host `aclrtSynchronizeStream` 等等价同步。

## 6. Atomic：远端累加和本地归约辅助

公开 Device atomic 能力目前集中在 atomic add：
```

---

### TOPIC A — additional items (multi-round toggle epochs, LCCL ipc_queue sequence numbers)

### item 0

https://raw.githubusercontent.com/sgl-project/sgl-kernel-npu/main/csrc/deepep/ops/op_kernel/cam_moe_combine_normal_multi_round.h
sgl-project/sgl-kernel-npu, main as of 2026-10-20

````cpp
#endif
constexpr uint64_t STATE_WIN_SIZE_HALF = STATE_WIN_SIZE / 2;
#ifdef __DAV_C310__
constexpr uint64_t MAGIC_WIN_OFFSET = 1100UL * 1024UL;
constexpr uint64_t HYBRID_DATA_EXTRA_OFFSET =
    Moe::A5WindowLayout::kLlSelectorMetadataSize + Moe::A5WindowLayout::kLlStateSize +
    Moe::A5WindowLayout::kLlSelectorMetadataSize + Moe::A5WindowLayout::kLlStateSize;
#else
constexpr uint64_t MAGIC_WIN_OFFSET = 975UL * 1024UL;
constexpr uint64_t HYBRID_DATA_EXTRA_OFFSET =
    Moe::A3WindowLayout::kLlSelectorMetadataSize + Moe::A3WindowLayout::kLlStateSize +

// ---- addressing by magic_ / roundMagic_ (lines 86-110, 130-140) ----
    __aicore__ inline void InitRoundRecvData();

    __aicore__ GM_ADDR GetStateAddrByRankId(const int32_t rankId)
    {
        return GetBaseWindAddrByRankId(epWinContext_, rankId, epRankId_) + winDataSizeOffset_ +
               NORMAL_NOTIFY_STATE_OFFSET;
    }

    __aicore__ GM_ADDR GetBufferAddrByRankId(const int32_t rankId)
    {
        uint64_t dataOffset = STATE_WIN_SIZE + roundMagic_ * combineDataBuffSize_;
        if (isHybridDeployment_) {
            dataOffset += HYBRID_DATA_EXTRA_OFFSET;
        }
        return GetStateAddrByRankId(rankId) + dataOffset;
    }

    __aicore__ inline GM_ADDR GetRoundStateAddrByRankId(const int32_t rankId)
    {
        return GetBaseWindStateAddrByRankId(epWinContext_, rankId, epRankId_) +
               roundMagic_ * Moe::ROUND_STATE_MAX_SIZE + ROUND_STATE_OFFSET;
    }

    __aicore__ inline void SplitCoreCal(uint32_t totalNum, uint32_t &perCoreNum, uint32_t &startIdx, uint32_t &endIdx)
    {

    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t coreIdx_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0};
    uint32_t magic_{0};
    uint32_t roundMagic_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t baseWinSize_{0};
    uint32_t hRecvXTypeLen_{0};
    uint32_t h32AlignFloatLen_{0};

// ---- InitMagic (lines 211-226) ----

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::InitMagic()
{
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpParam *)contextGM0;

    GlobalTensor<int32_t> selfMagicTensor;
    selfMagicTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(GetStatusDataSpaceGm(epWinContext_) + MAGIC_WIN_OFFSET + coreIdx_ * WIN_512_ALIGN));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
    magic_ = selfMagicTensor(0);
    selfMagicTensor(0) = ((magic_ == 0) ? 1 : 0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
}


// ---- Process round loop with roundMagic_ flip (lines 798-816) ----

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        uint32_t realRound = (realMaxBs_ + perRoundTokens_ - 1) / perRoundTokens_;
        while (roundIndex_ < realRound) {
            CopyBufferToShareAndSetStatus();
            ReadBufferFromRemote();
            if (realRound > 1) {
                SetRoundStatus();
                WaitRoundStatus();
                roundMagic_ = roundMagic_ == 0 ? 1 : 0;
                SyncAll<true>();
            }
            roundIndex_ += 1;
        }
    }
}
````

## ITEM 28 — LCCL `IpcQueue`: a cross-rank ring buffer whose ONLY synchronisation is the epoch-tagged flag. The flag's low bits carry a monotonically increasing sequence number (`count`), extracted with `EVENT_ID_MASK` to recover the producer's progress: `int64_t val = sync->GetInnerFlag(checkRank, checkBlock) & EVENT_ID_MASK; count = val + 1; front = (val + 1) % depth;`. Multi-producer variants take the MINIMUM of all peers' sequence numbers. No separate flag buffer, no reset.

### item 1

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/src/ascendc_kernels/ipc_queue.h
Huawei / ascend-transformer-boost LCCL, file header "Copyright (c) 2024 Huawei Technologies Co., Ltd."; mirror commit date unknown

````cpp
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_IPC_QUEUE_H
#define LCCL_IPC_QUEUE_H
#include "sync_collectives.h"
using namespace AscendC;

template <typename T>
class IpcQueue {
public:
    FORCE_INLINE_AICORE IpcQueue() {}

    FORCE_INLINE_AICORE void Init(SyncCollectives *sync, int64_t magic, GM_ADDR workSpace, uint64_t bufferNum,
                                   uint64_t blockNum)
    {
        this->sync = sync;
        this->magic = magic;
        depth = bufferNum / blockNum;
        front = 0;
        rear = 0;
        count = 0;
        this->blockNum = blockNum;
        buff.SetGlobalBuffer((__gm__ T*)workSpace, bufferNum);
        blockIdx = GetBlockIdx();
    }

    FORCE_INLINE_AICORE bool Full()
    {
        if ((rear + 1) % depth == front) {
            return true;
        }
        return false;
    }

    FORCE_INLINE_AICORE GlobalTensor<T> EnQue()
    {
        uint64_t rearOld = rear;
        rear = (rear + 1) % depth;
        return buff[rearOld * blockNum];
    }

    FORCE_INLINE_AICORE void DeQue(int checkRank, int checkBlock = -1)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        sync->WaitInnerFlag(magic, count, checkRank, checkBlock);
        PipeBarrier<PIPE_ALL>();
        int64_t val = sync->GetInnerFlag(checkRank, checkBlock) & EVENT_ID_MASK;
        count = val + 1;
        front = (val + 1) % depth;
    }

    FORCE_INLINE_AICORE void DeQue(int *rankList, int checkCount, int checkBlock = -1)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        int64_t minIndex = LLONG_MAX;
        for (int i = 0; i < checkCount; i++) {
            sync->WaitInnerFlag(magic, count, rankList[i], checkBlock);
            PipeBarrier<PIPE_ALL>();

            int64_t val = sync->GetInnerFlag(rankList[i], checkBlock) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        count = minIndex + 1;
        front = (minIndex + 1) % depth;
    }
    FORCE_INLINE_AICORE void DeQue(int *rankList, int *blockIdxList, int checkCount)
    {
        if (!Full()) {
            return;
        }

        int64_t minIndex = LLONG_MAX;
        for (int i = 0; i < checkCount; i++) {
            sync->WaitInnerFlag(magic, count, rankList[i], blockIdxList[i]);
            PipeBarrier<PIPE_ALL>();

            int64_t val = sync->GetInnerFlag(rankList[i], blockIdxList[i]) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        count = minIndex + 1;
        front = (minIndex + 1) % depth;
    }

    FORCE_INLINE_AICORE GlobalTensor<T> ReadFront()
    {
        uint64_t frontOld = front;
        front = (front + 1) % depth;
        return buff[frontOld * blockNum];
    }

private:
    int64_t magic;
    uint64_t depth;
    uint64_t front;
    uint64_t rear;
    uint64_t count;
    uint64_t blockNum;
    GlobalTensor<T> buff;
    SyncCollectives *sync;
    int blockIdx;
};
#endif // LCCL_IPC_QUEUE_H
````

### item 2

https://raw.githubusercontent.com/egretwAlker/cpl-ascend-transformer-boost/HEAD/src/kernels/lcal/include/comm_args.h
Huawei / ascend-transformer-boost LCCL — the constants that define the epoch/payload bit split and the IPC window geometry

````cpp
#include "kernel_operator.h"
#endif
namespace Lcal {

constexpr int LCAL_MAX_RANK_SIZE = 128; // lcal通信库最大支持的npu卡数
constexpr int RANK_SIZE_TWO = 2;  // 可用SIO的规模，以及是否需要跨卡搬运数据核的分界规模
constexpr int64_t IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024; // 前2MB作为flag标志位，之后100MB作为数据存储
constexpr int64_t SYNC_FLAG_BIT_NUM = 10;  // cce 算子在用
constexpr int64_t MEM_DMA_UNIT_INT_NUM = 4;
constexpr int64_t EVENT_ID_MASK = 0xFFFFFFFF;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int UB_ALIGN_SIZE = 32;

// 2step算法中，2个aiv真正用作数据预处理
constexpr int64_t PRE_CORE_REAL_NUM = 2;

constexpr int64_t AIV_PER_AICORE = 2;

constexpr int DFX_COUNT = 50;

constexpr int64_t HALF_NUM = 2;

constexpr int64_t THREE_NUM = 3;

constexpr int64_t FOUR_NUM = 4;

constexpr int64_t VADD_MAX_REPEAT = 255;
constexpr int64_t VADD_UNIT_BYTE = 256;

// vadd单位粒度是256B，vadd最大repeat次数为255，两个相乘的结果
constexpr int64_t MAX_VADD_SIZE = VADD_MAX_REPEAT * VADD_UNIT_BYTE;
constexpr int64_t BLOCK_UNIT_BYTE = 32;
constexpr int64_t VADD_UNIT_TO_BLOCK_UNIT_RATIO = VADD_UNIT_BYTE / BLOCK_UNIT_BYTE;    // 8

constexpr bool ATOMIC_ENABLE = false;

constexpr int32_t LCCL_DUMP_UINT_SIZE = 1 * 1024 * 1024;
enum Op : int {
    COPYONLY = -1,
````

### TOPIC A — negative findings (sources checked that yielded nothing)

NOTHING: https://github.com/sgl-project/sglang — sglang's own tree has no in-kernel Ascend collective; the only hits for `lccl` are docs telling MindSpore users to `export MS_ENABLE_LCCL=off`. The hand-written AscendC ops live in the separate `sgl-project/sgl-kernel-npu` repo (covered above).
NOTHING: https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/csrc/utils/inc/kernel/moe_distribute_base.h — contains HCCL window/context struct definitions (`flag`, `hostStateInfo`, `aicpuStateInfo`) but no magic/epoch protocol.
NOTHING: https://raw.gitcode.com/cann/ops-hccl/raw/master/README.md — returns `{"code":"CH.00402000","title":"Project not found","message":"Project not found:cann/ops-hccl"}`; that repo path does not exist on gitcode.
NOTHING: https://raw.githubusercontent.com/hicann/cann-recipes-infer/master/ops/shmem/distributed_embedding/op_kernel/distributed_hashtable.h — 30-line forwarding header only; the substance is in the `.cpp` (ITEM 24).
NOTHING: https://raw.githubusercontent.com/Zhangyq-1999/shmem-doc/main/_sources/api/device_api.rst.txt — 72 lines of Doxygen `doxygenfile::` directives only; no inline API text. The real content is generated from the headers (ITEM 4, ITEM 20).
NOTHING: Chinese web search for 「昇腾 LCCL one-shot allreduce 低时延 原理 magic 标志位」, 「昇腾 AscendC kernel 内 跨卡 同步 标志位 magic 轮次 对称内存 轮询」, 「LCCL 昇腾 低时延通信库 AllReduce 原理 共享内存 IPC buff 博客」 — no Ascend-specific blog on zhihu/csdn/cnblogs/bbs.huaweicloud.com surfaced; results were GPU-collective papers and an unrelated "LCCL" name collision in arXiv:2512.03644. All substantive material on this question is in source, not blog prose.


---

# TOPIC A — remote-write visibility, cache maintenance, and the official memory model

Items below are verbatim quotations collected from the URLs shown. Note two provenance facts the
sweep established: the hiascend.com doc portal publishes nothing newer than CANN 9.0.0 (`900`)
and has no `aclshmem` book at all — the normative ACLSHMEM semantics exist only in the
`gitcode.com/cann/shmem` headers and `docs/api/*.md`; and `asc.gitcode.com` hosts the
auto-built `asc-devkit` documentation, which contains the authoritative NPU memory-model
chapters (items 6-8) naming `NPU架构版本2201` explicitly.

### item 0

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0177.html
Huawei / 昇腾社区 CANN 社区版 9.0.0 (CANNCommunityEdition/900) Ascend C API 参考 — 缓存控制; page date unknown (content endpoint https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0177.html)
The official DataCacheCleanAndInvalid API-reference page: 产品支持情况 / 功能说明 / 函数原型 / 参数说明 (entireType = SINGLE_CACHE_LINE | ENTIRE_DATA_CACHE, dcciDst = CACHELINE_ALL/UB/OUT/ATOMIC) / 返回值说明 / 约束说明 / 调用示例, including 示例3 on multi-core same-cache-line random overwrite.
```
### DataCacheCleanAndInvalid

### 产品支持情况

产品

|

是否支持（支持配置dcciDst的原型）

|

是否支持（不支持配置dcciDst的原型）

|

Atlas 350 加速卡

|

√

|

√

|

Atlas A3 训练系列产品/Atlas A3 推理系列产品

|

√

|

√

|

Atlas A2 训练系列产品/Atlas A2 推理系列产品

|

√

|

√

|

Atlas 200I/500 A2 推理产品

|

√

|

√

|

Atlas 推理系列产品AI Core

|

x

|

√

|

Atlas 推理系列产品Vector Core

|

x

|

x

|

Atlas 训练系列产品

|

x

|

x

|

### 功能说明

在AI Core内部，Scalar单元和DMA单元都可能对Global Memory进行访问。

图1 DataCache内存层次示意图

如上图所示：

DMA搬运单元读写Global Memory，数据通过DataCopy等接口在UB等Local Memory和Global Memory间交互，没有Cache一致性问题；

Scalar单元访问Global Memory，首先会访问每个核内的Data Cache，因此存在Data Cache与Global Memory的Cache一致性问题。

该接口用来刷新Cache，保证Cache的一致性，使用场景如下：

读取Global Memory的数据，但该数据可能在外部被其余核修改，此时需要使用DataCacheCleanAndInvalid接口，直接访问Global Memory，获取最新数据；

用户通过Scalar单元写Global Memory的数据，希望立刻写出，也需要使用DataCacheCleanAndInvalid接口。

针对Atlas 350 加速卡，原子操作过程中，如果希望改变后续数据的饱和模式，需要先使用DataCacheCleanAndInvalid接口将Cache Line中现存的数据立刻写出，再调用SetCtrlSpr设置后续数据的饱和模式。

### 函数原型

支持通过配置dcciDst确保Data Cache与GM存储的一致性

1
2

|

template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)

|

支持通过配置dcciDst确保Data Cache与Local Memory存储的一致性

1
2

|

template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst)

|

不支持配置dcciDst，仅支持保证Data Cache与GM的一致性

1
2

|

template <typename T, CacheLine entireType>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)

|

### 参数说明

表1 模板参数说明

参数名

|

描述

|

T

|

dst的数据类型。

|

entireType

|

指令操作的模式：

SINGLE_CACHE_LINE：只刷新传入地址所在的Cache Line，注意如果该地址非64B对齐，只会操作传入地址到64B对齐的部分。

ENTIRE_DATA_CACHE：此时传入的地址无效，核内会刷新整个Data Cache，但是耗时较大，性能敏感的场景慎用。

|

dcciDst

|

表示使用该接口来保证Data Cache与哪一种存储保持一致性，类型为DcciDst枚举类。

CACHELINE_ALL：与CACHELINE_OUT效果一致。

CACHELINE_UB：表示通过该接口来保证Data Cache与UB的一致性。

CACHELINE_OUT：表示通过该接口来保证Data Cache与Global Memory的一致性。

CACHELINE_ATOMIC：
Atlas 350 加速卡，原子操作过程中保证Data Cache和Global Memory的一致性。

Atlas A3 训练系列产品/Atlas A3 推理系列产品，预留参数，暂未支持。

Atlas A2 训练系列产品/Atlas A2 推理系列产品，预留参数，暂未支持。

Atlas 200I/500 A2 推理产品，预留参数，暂未支持。

Atlas 推理系列产品AI Core，预留参数，暂未支持。

|

表2 参数说明

参数名

|

输入/输出

|

描述

|

dst

|

输入

|

需要刷新Cache的Tensor。

|

### 返回值说明

无

### 约束说明

无

### 调用示例

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34

|

// 示例1：SINGLE_CACHE_LINE 模式，假设mmAddr_为0x40（64B对齐）
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
for( int i = 0; i < 8; i++) {
global.SetValue(i, AscendC::GetBlockIdx());
}
// 由于首地址64B对齐，调用DataCacheCleanAndInvalid指令后，会立刻刷新前8个数
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
// 示例2：SINGLE_CACHE_LINE 模式，假设mmAddr_为0x20（非64B对齐）
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
for( int i = 0; i < 8; i++) {
global.SetValue(i, AscendC::GetBlockIdx());
}
// 由于首地址非64B对齐，调用1条指令，只会刷新起始地址至64B字节对齐的部分，即前4个数
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
// 需要再次调用DataCacheCleanAndInvalid指令，刷新后4个数
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[4]);
// 示例3：SINGLE_CACHE_LINE 模式，假设mmAddr_为0x40（64B对齐），多核处理场景（本样例仅做示例说明，便于开发者理解使用限制，非正常使用样例）
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_);
global.SetValue(AscendC::GetBlockIdx(), AscendC::GetBlockIdx());
// 算子中多核操作虽然不在同一个地址，但在同一个Cache Line, 会出现数据的随机覆盖，和通用CPU的行为不同
// 调用DataCacheCleanAndInvalid指令后，由于多核操作的时间不一致，最终结果存在随机性，后执行的核会覆盖前面核的结果
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
// 示例4：ENTIRE_DATA_CACHE 模式，假设mmAddr_为0x20（非64B对齐）
// 本样例仅做示例说明，便于开发者理解使用限制，非正常使用样例
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
for( int i = 0; i < 8; i++) {
global.SetValue(i, AscendC::GetBlockIdx());
}
// 刷新整个Data Cache，性能较差
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(global);

|

父主题： 缓存控制
```

### item 1

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00015.html
Huawei / 昇腾社区 CANN 社区版 9.0.0, Ascend C 算子开发（编程指南）最佳实践 — 头尾开销优化; page date unknown
States that the compilation framework AUTO-INSERTS a DCCI at operator tail when the developer writes GM via `* __gm__` or GlobalTensor.SetValue without calling DataCacheCleanAndInvalid, and that --cce-no-dcache-flush=true downgrades it to DCI (invalidate only, no clean).
```
### 设置DCI编译选项来减少算子尾开销

该性能优化建议适用于如下型号：

Atlas 350 加速卡

【优先级】高

【描述】算子执行结束时，需要将DCache置为无效，防止后续算子继续使用DCache中的数据而受到影响。可以通过在编译选项中添加--cce-no-dcache-flush=true，用于在算子尾部增加DCI（DataCacheInvalid）指令来使DCache失效。如果不开启该选项，则会默认增加DCCI（DataCacheCleanAndInvalid）指令来使DCache失效。

插入DCI指令相比于插入DCCI指令，其减少了数据从DCache同步到GM（Clean）的过程，性能上会有一定优势。插入DCCI是一种额外的容错保证，如果开发者使用了* __gm__的方式改写GM内存，或者调用GlobalTensor.SetValue函数时，没有正确的调用DataCacheCleanAndInvalid接口来保证Cache一致性，编译框架自动插入DCCI恰好可以保证算子精度正常。

所以在如下场景，可以通过开启该编译选项来降低算子尾部开销：

算子使用* __gm__的方式改写GM内存，或者调用GlobalTensor.SetValue函数时，正确的使用DataCacheCleanAndInvalid接口，手动将数据从DCache中回刷到GM上，保证Cache的一致性。不依赖编译框架自动插入DCCI指令来保证一致性。

算子不包含使用* __gm__的方式改写GM内存，或者调用GlobalTensor.SetValue函数的代码。

父主题： 头尾开销优化
```

### item 2

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00031.html
Huawei / 昇腾社区 CANN 社区版 9.0.0, Ascend C 编程指南 — 内存访问原理 / Scalar读写数据; page date unknown
The official statement of the DataCache mechanism behind Scalar GM access: per-AIC/AIV private DataCache, 64B cache line, SetValue only marks the line Dirty, and 多核间访问…需要开发者手动调用DataCacheCleanAndInvalid来保证数据的一致性.
```
### Scalar读写数据

AI Core中Scalar计算单元负责各类型的标量数据运算和程序的流程控制。根据硬件架构设计，Scalar仅支持对Global Memory和Unified Buffer的读写操作，而不支持对L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer等其他类型存储的访问。下文分别介绍了Scalar读写Global Memory和Unified Buffer的方式和Scalar读写数据时的同步机制。

### Scalar读写Global Memory

如上图所示，Scalar读写GM数据时会经过DataCache，DataCache主要用于提高标量访存指令的执行效率，每一个AIC/AIV核内均有一个独立的DataCache。下面通过一个具体示例来讲解DataCache的具体工作机制。

globalTensor1是位于GM上的Tensor：

执行完GetValue(0)后，globalTensor1的前8个元素会进入DataCache，后续GetValue(1)~GetValue(7)不需要再访问GM，而可以直接从DataCache的Cache Line中读取数据，提高了标量连续访问的效率。

执行完SetValue(8, val)后，globalTensor1的index为8~15的元素会进入DataCache，SetValue只会修改DataCache中的Cache Line数据，同时将Cache Line的状态设置为Dirty，表明Cache Line中的数据与GM中的数据不一致。

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18

|

AscendC::GlobalTensor<int64_t> globalTensor1;
globalTensor1.SetGlobalBuffer((__gm__ int64_t *)input);
// 从0~7共计8个uint64_t类型，DataCache的Cache Line长度为64字节
// 执行完GetValue(0)后，GetValue(1)~GetValue(7)可以直接从Cache Line中读取，不需要再访问GM
globalTensor1.GetValue(0);
globalTensor1.GetValue(1);
globalTensor1.GetValue(2);
globalTensor1.GetValue(3);
globalTensor1.GetValue(4);
globalTensor1.GetValue(5);
globalTensor1.GetValue(6);
globalTensor1.GetValue(7);

// 执行完SetValue(8)后，不会修改GM上的数据，只会修改DataCache中Cache Line数据
// 同时Cache Line的状态置为dirty，dirty表示DataCache中Cache Line数据与GM中的数据不一致
int64_t val = 32;
globalTensor1.SetValue(8, val);
globalTensor1.GetValue(8);

|

根据上文的工作机制（如下图所示），多核间访问globalTensor1会出现数据不一致的情况，如果其余核需要获取GM数据的变化，则需要开发者手动调用DataCacheCleanAndInvalid来保证数据的一致性。

### Scalar读写Unified Buffer

Scalar读写Unified Buffer时，可以使用LocalTensor的SetValue和GetValue接口。示例如下：

1
2
3
4
5
6
7

|

for (int32_t i = 0; i < 16; ++i) {
inputLocal.SetValue(i, i); // 对inputLocal中第i个位置进行赋值为i
}

for (int32_t i = 0; i < srcLen; ++i) {
auto element = inputLocal.GetValue(i); // 获取inputLocal中第i个位置的数值
}

|

### Scalar读写数据时的同步

Scalar读写Global Memory和Unified Buffer时属于PIPE_S（Scalar流水）操作，当用户使用SetValue或者GetValue接口，且算子工程使能自动同步时，不需要手动插入同步事件。

如果用户关闭算子工程的自动同步功能时，则需要手动插入同步事件：

1
2
3
4
5
6
7
8
9
10
11
12
13

|

// GetValue为Scalar操作，与后续的Duplicate存在数据依赖
// 因此Vector流水需要等待Scalar操作结束
float inputVal = srcLocal.GetValue(0);
SetFlag<HardEvent::S_V>(eventID1);
WaitFlag<HardEvent::S_V>(eventID1);
AscendC::Duplicate(dstLocal, inputVal, srcDataSize);

// SetValue为Scalar操作，与后续的数据搬运操作存在数据依赖
// 因此MTE3流水需要等待Scalar操作结束
srcLocal.SetValue(0, value);
SetFlag<HardEvent::S_MTE3>(eventID2);
WaitFlag<HardEvent::S_MTE3>(eventID2);
AscendC::DataCopy(dstGlobal, srcLocal, srcDataSize);

|

父主题： 内存访问原理
```

### item 3

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00188.html
Huawei / 昇腾社区 CANN 社区版 9.0.0, Ascend C API 参考 — AI CPU API / DataStoreBarrier; page date unknown
The official DataStoreBarrier entry (功能说明/约束说明) PLUS the official 调用示例 showing the canonical invalidate-then-poll loop on a GM lock: `while (*(volatile __gm__ uint64_t*)lock.GetPhyAddr(0) == 0) { DataCacheCleanAndInvalid<...SINGLE_CACHE_LINE, CACHELINE_OUT>(lock); }` with the writer side using DataStoreBarrier.
```
### DataStoreBarrier

### 产品支持情况

产品
|

是否支持
|

Atlas 350 加速卡
|

√
|

Atlas A3 训练系列产品
/
Atlas A3 推理系列产品

|

√
|

Atlas A2 训练系列产品
/
Atlas A2 推理系列产品

|

√
|

Atlas 200I/500 A2 推理产品

|

x
|

Atlas 推理系列产品
AI Core
|

x
|

Atlas 推理系列产品
Vector Core
|

x
|

Atlas 训练系列产品

|

x
|

### 功能说明

数据同步屏障指令，该指令会阻塞当前线程执行，确保所有先前的写内存操作完成并对其它硬件单元可见后，才继续执行后续指令。用于AI CPU与AI Core多核之间的数据一致性。

### 需要包含的头文件

1

|

#include "aicpu_api.h"

|

### 函数原型

1

|

DataStoreBarrier(void)

|

### 参数说明

无

### 返回值说明

无

### 约束说明

该接口仅支持通过<<<...>>>调用，并在异构编译场景使用。

### 调用示例

在AI CPU算子Kernel侧实现代码中调用AscendC::DataStoreBarrier()，确保AI CPU算子对Tiling数据的修改写入内存，使得AI Core算子能够正确读取Tiling数据：

struct TilingInfo {
uint64_t lock; // AI CPU/AI Core之间同步的锁
int8_t type;
int8_t mode;
int8_t len;
};
struct KernelArgs {
uint32_t *xDevice;
uint32_t *yDevice;
uint32_t *zDevice;
TilingInfo *ti; // 与AI Core共享的参数，用于同步tiling选择
};

template<typename T, int8_t mode, int8_t len>
__aicore__ void hello_world_impl(GM_ADDR m)
{
if constexpr (std::is_same_v<T, float>) {
AscendC::printf("Hello World: float mode %u len %u.\n", mode, len);
} else if constexpr (std::is_same_v<T, int>) {
AscendC::printf("Hello World: int mode %u len %u.\n", mode, len);
}
}

// AI Core算子总入口
// tilingInfo: 和AI CPU算子共同传递的参数，用于数据共享
template<typename T, int8_t mode, int8_t len>
__mix__(1,2) __global__ __aicore__ void hello_world(GM_ADDR m, GM_ADDR TilingPtr)
{
__gm__ struct KernelInfo::TilingInfo *ti = (__gm__ struct KernelInfo::TilingInfo *)TilingPtr;
AscendC::GlobalTensor<uint64_t> lock;
lock.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(&ti->lock));
if ASCEND_IS_AIV {
if (AscendC::GetBlockIdx() == 0) {
while (*reinterpret_cast<volatile __gm__ uint64_t*>(lock.GetPhyAddr(0)) == 0) {   // 下沉模式，AI Core等待AICPU tiling计算完成
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
AscendC::DcciDst::CACHELINE_OUT>(lock);    //直接访问Global Memory，获取最新数据
}
}
}
// 上面是1个核等待AI CPU tiling计算完成，这里进行核间同步
AscendC::SyncAll<false>();
// 根据tiling参数值选择不同模板
if (ti->type ==0 && ti->mode == 1 && ti->len == 2) {
hello_world_impl<float, 1, 2>(m);
} else if (ti->type == 1 && ti->mode == 2 && ti->len == 4) {
hello_world_impl<int, 2, 4>(m);
}
// 执行完留一个核释放lock
if ASCEND_IS_AIV {
if (AscendC::GetBlockIdx() == 0) {
lock.SetValue(0, 0);  // 刷新 lock
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
AscendC::DcciDst::CACHELINE_OUT>(lock);    //刷新Dcache，同步与GM之间的数据
}
}
}

extern "C" __global__ __aicpu__ uint32_t MyAicpuKernel(void *arg)
{
KernelArgs* cfg = (KernelArgs*)arg;
AscendC::printf("MyAicpuKernel inited!\n");
cfg->ti->lock = 1;
cfg->ti->type = 1;
cfg->ti->mode = 2;
cfg->ti->len = 4;
AscendC::DataStoreBarrier(); // 对tilingInfo进行写同步
AscendC::printf("MyAicpuKernel inited type %u mode %u len %u end!\n", cfg->ti->type, cfg->ti->mode, cfg->ti->len);
return 0;
}

父主题： AI CPU API
```

### item 4

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0069.html
Huawei / 昇腾社区 CANN 社区版 9.0.0, HCCL API 参考 — HcclCommSymWinRegister; page date unknown
The official definition of 对称内存 (symmetric memory) as a CANN memory model — 全局可见 but 无需显式地址交换, identical per-rank virtual layout, remote physical mapping — this is the HCCL-side counterpart of the aclshmem symmetric heap.
```
### HcclCommSymWinRegister

### 产品支持情况

产品
|

是否支持
|

Atlas 350 加速卡
|

☓
|

Atlas A3 训练系列产品
/
Atlas A3 推理系列产品

|

√
|

Atlas A2 训练系列产品
/
Atlas A2 推理系列产品

|

☓
|

Atlas 200I/500 A2 推理产品

|

☓
|

Atlas 推理系列产品

|

☓
|

Atlas 训练系列产品

|

☓
|

### 功能说明

用户在申请虚拟内存和物理内存并完成映射后，可调用此接口将虚拟内存注册为对称内存。

对称内存是一种内存管理模型，允许并行处理单元（例如每个rank）以一种“全局可见”但“无需显式地址交换”的方式访问彼此的内存。

对称内存的基本实现模型如下图所示，每个rank通过提前预留相同大小、相同布局的虚拟地址来实现。

为每个rank申请虚拟内存，假设每个rank对应的虚拟内存大小为heap_size，通信域中的rank数量为rank_size，则通信域中总虚拟内存大小为heap_size*rank_size。

每个rank的虚拟地址布局都是相同的。

不同rank的物理地址映射到每个rank对应位置的虚拟地址，实现对其他rank内存的访问。

对称内存功能使得HCCL可以直接对业务传入的内存进行操作，无需经过中间缓冲区（HCCL buffer），从而减少内存拷贝开销。

### 函数原型

1

|

HcclResult HcclCommSymWinRegister(HcclComm comm, void *addr, uint64_t size, CommSymWindow *winHandle, uint32_t flag)

|

### 参数说明

参数名
|

输入/输出
|

说明
|

comm
|

输入
|

HCCL通信域，建议使用超节点内最大的通信域，即覆盖最大卡数的通信域。

初始化通信域时可以通过HcclCommConfig的hcclSymWinMaxMemSizePerRank参数设置每个rank预留的对称内存大小，若不设置hcclSymWinMaxMemSizePerRank，使用默认值16GB。

当前通信域预留的总虚拟对称内存大小为：rankSize * HcclCommConfig.hcclSymWinMaxMemSizePerRank。
|

addr
|

输入
|

预留的虚拟内存地址。

虚拟内存需要调用aclrtReserveMemAddress接口预留。
|

size
|

输入
|

对称内存窗口的大小，0 < size <= HcclCommConfig.hcclSymWinMaxMemSizePerRank，并且size不能超过“与addr做映射的物理内存”大小（即调用aclrtMallocPhysical接口申请的Device物理内存）。

注：对称内存注册按物理内存的大小对齐，实际注册的对称内存窗口大小等于“与addr做映射的物理内存”的大小。
|

winHandle
|

输出
|

存放“对称内存窗口资源句柄“的指针。

HcclCommSymWindow类型的定义请参见HcclCommSymWindow。
|

flag
|

输入
|

是否启用对称内存，当前仅支持传入1。
|

### 返回值

HcclResult：接口成功返回HCCL_SUCCESS，其他失败。

### 约束说明

仅支持
Atlas A3 训练系列产品
/
Atlas A3 推理系列产品
的超节点内通信。

仅支持通信算子展开模式为AI CPU的场景。

仅支持超节点内AI Server间使用HCCS链路进行SDMA通信的场景，不支持使用RoCE进行RDMA通信的场景（即不支持设置环境变量HCCL_INTER_HCCS_DISABLE为"TRUE"，单机场景该环境变量无效）。

仅支持对称组网，即每个Server内卡数相同的场景。

该接口仅支持集合通信算子AllGather、ReduceScatter、AllReduce、AlltoAll。

需确保通信域中的所有rank同时调用该注册接口。

所有rank的输入地址映射的物理内存大小一致（对称内存注册按物理内存的大小对齐）。

所有rank调用该接口时，输入的size参数需要保持一致。

使用对称内存功能时，算子的输入、输出内存必须调用此接口注册为对称内存。

调用该接口注册的内存需要使用HcclCommSymWinDeregister接口解注册。

### 调用示例

// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
// 按需修改通信域配置
config.hcclSymWinMaxMemSizePerRank = 10; //单位GB, 默认值为16。设置对称堆预留的虚拟内存大小 = rankSize * config.hcclSymWinMaxMemSizePerRank;

// 获取通信域参数
uint32_t rankSize = 4;
uint32_t rankId = 0;
int32_t deviceId;
ACLCHECK(aclrtGetDevice(&deviceId));
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// 初始化集合通信域
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// 创建任务流
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

// 物理内存属性配置
aclrtPhysicalMemProp prop;
prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
prop.memAttr = ACL_HBM_MEM_HUGE;
prop.location.id = deviceId;
prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
prop.reserve = 0;

// 获取对齐粒度，通常为2M
size_t granularity;
ACLCHECK(aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity));

// size按粒度对齐
size_t size = 2 * 1024 * 1024;
size_t allocSize = (size + granularity - 1) / granularity * granularity;

// 预留虚拟内存
void *virPtr;
ACLCHECK(aclrtReserveMemAddress(&virPtr, allocSize, 0, nullptr, 1));

// 申请物理内存
aclrtDrvMemHandle memHandle;
ACLCHECK(aclrtMallocPhysical(&memHandle, allocSize, &prop, 0));

// 建立物理到虚拟的映射
ACLCHECK(aclrtMapMem(virPtr, allocSize, 0, memHandle, 0));

size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
HcclCommSymWindow symWin;
// 注册对称内存
HCCLCHECK(HcclCommSymWinRegister(hcclComm, virPtr, sendBytes + recvBytes, &symWin, 1));

// 使用对称内存
void *sendBuff = virPtr;
void *recvBuff = static_cast<char*>(sendBuff) + sendBytes;

// 调用集合通信算子
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// 阻塞等待任务流中的集合通信任务执行完成
ACLCHECK(aclrtSynchronizeStream(stream));

// 解注册对称内存
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// 释放内存
ACLCHECK(aclrtUnmapMem(virPtr));
ACLCHECK(aclrtFreePhysical(memHandle));
ACLCHECK(aclrtReleaseMemAddress(virPtr));

// 销毁任务流
ACLCHECK(aclrtDestroyStream(stream));

// 销毁通信域
HCCLCHECK(HcclCommDestroy(hcclComm));

父主题： 对称内存
```

### item 5

https://asc.gitcode.com/guide/programming_guide/advanced_programming/memory_model/cache_coherence.html
Huawei / cann/asc-devkit master docs (auto-built site, 版本 master 提交 5f50fc4c6ada, 构建于 2026/10/20 11:47:29); source md: https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/memory_model/cache_coherence.md
THE authoritative Ascend cache-coherence chapter: per-core DCache 64B line sizes for NPU arch 2201 and 3510, Clean/Dirty/Invalid/DCCI definitions, the explicit statement that hardware maintains coherence ONLY for same-core Scalar DCache producer+consumer, the write-端 Dirty / 读端旧副本 taxonomy, the T0..T6 producer/consumer timelines showing DCCI needed on BOTH sides, the Atomic-bypasses-DCache cases, and the full RAW/WAW/WAR software-maintenance table (Scalar.STORE -> Scalar.DCCI -> Scalar.DSB -> sync -> reader DCCI -> LOAD).
```
### 缓存一致性

本文介绍NPU编程中常见的缓存一致性问题，说明Data Cache（DCache）、Instruction Cache（ICache）、SIMT DCache与Global Memory（GM）的数据一致性。芯片的硬件架构参考NPU架构版本2201和NPU架构版本3510。

### 缓存一致性的基本概念​

### Cache、Cache Line与缓存副本​

Cache用于缓存后续可能重复访问的数据或指令，降低访问GM的开销。一次Cache加载通常不是只加载被访问的单个字节或单个变量，而是把该地址所在的一段连续数据作为一个整体加载到Cache中，这个整体称为Cache Line。

Cache Line是Cache加载、失效、写回的最小操作粒度。例如访问GM内存时，会将GM里一个Cache Line长度内相邻的数据加载到L2 Cache缓存，利用空间局部性提高访问效率。Cache Line是硬件真正搬入、标记、替换和写回缓存数据的最小粒度，直接影响性能和一致性。NPU架构版本3510支持L2 Cache、DCache、ICache、SIMT DCache、NDDMA Cache；NPU架构版本2201支持L2 Cache、DCache和ICache。本文讨论的通用Cache单元信息如下：

NPU架构版本3510Cache单元信息

存储单元 | 位置 | 数据一致性说明 | NPU架构版本3510Cache Line大小 | NPU架构版本3510Cache大小 |

L2 Cache | 核外共享缓存 | 多个核共享，不考虑多个核之间的Cache副本不一致问题 | 128B/512B | 128MB |

AIC ICache | Scalar取指缓存 | 只读，不考虑多个核之间的数据不一致问题 | 128B | 32KB |

AIV ICache | Scalar取指缓存 | 只读，不考虑多个核之间的数据不一致问题 | 128B | 16KB |

AIC DCache | Scalar数据缓存 | 每个核独立缓存，需要考虑多个核之间的数据一致性问题 | 64B | 32KB |

AIV DCache | Scalar数据缓存 | 每个核独立缓存，需要考虑多个核之间的数据一致性问题 | 64B | 32KB |

SIMT DCache | SIMT数据缓存 | 每个核独立缓存，需要考虑多个核之间的数据一致性问题 | 128B | 32KB~128KB |

NPU架构版本2201Cache单元信息

存储单元 | 位置 | 数据一致性说明 | NPU架构版本2201Cache Line大小 | NPU架构版本2201Cache大小 |

L2 Cache | 核外共享缓存 | 多个核共享，不考虑多个核之间的Cache副本不一致问题 | 512B | 192MB |

AIC ICache | Scalar取指缓存 | 只读，不考虑多个核之间的数据不一致问题 | 128B | 32KB |

AIV ICache | Scalar取指缓存 | 只读，不考虑多个核之间的数据不一致问题 | 128B | 16KB |

AIC DCache | Scalar数据缓存 | 每个核独立缓存，需要考虑多个核之间的数据一致性问题 | 64B | 16KB |

AIV DCache | Scalar数据缓存 | 每个核独立缓存，需要考虑多个核之间的数据一致性问题 | 64B | 32KB |

同一份数据可能同时存在于GM、L2 Cache、DCache、SIMT DCache等多个位置。对于每个核独立维护的Cache，同一地址或同一Cache Line可能在不同核内形成不同副本：

某个核已经在本核Cache中修改了数据，但GM或其他核的Cache中仍保留旧值；

某个核读取前已经缓存过旧值，即使GM后来被其他路径更新，后续读取仍可能继续命中旧副本。

这类副本不一致就是本文关注的缓存一致性问题。

### Clean、Dirty、Invalid、写策略与DCCI​

Cache中的数据可以按与下一级存储的关系理解为以下几类状态：

Clean：Cache Line中的数据与下一级存储中的数据一致，直接丢弃该Cache Line不会丢失更新。

Dirty：Cache Line中的数据已经被当前Cache修改，但尚未写回下一级存储。此时Cache中的数据与下一级存储中的数据不一致，如果其他路径直接读取GM，可能读到旧值。

Invalid：Cache Line无效，后续访问不能继续使用该Cache Line中的旧数据，需要重新从下一级存储加载。

Data Cache Clean and Invalid（DCCI）用于处理Cache副本。Clean操作用于把Dirty Cache Line写回到目标存储，Invalid操作用于使Cache Line失效，避免后续继续命中旧副本。

实际使用时，可根据一致性目标选择不同DCCI接口：使用asc_dcci_single刷新指定地址所在的单个Cache Line；使用asc_dcci_entire_out刷新面向GM的一致性目标；原子相关目标按产品支持情况选择asc_dcci_entire_atomic等接口。

对于需要等待DCCI触发的写回真正完成的场景，还需要配合Data Sync Barrier（DSB）等待前序内存访问完成，DSB只用于阻塞后续指令执行，直到之前的内存访问指令执行结束，参数可选择DSB_ALL、DSB_DDR或DSB_UB。

Cache写入策略决定写操作是否会形成Dirty副本，常见策略包括直写和写回：

直写（write-through）：写操作更新Cache副本的同时，将数据写出到下一级存储。直写路径通常不会留下需要后续Clean回下一级存储的Dirty副本，但写操作是否已经完成仍需要通过DSB等同步机制确认；如果读端已经缓存旧副本，仍需要按读端路径执行Invalid。

写回（write-back）：写操作先更新Cache副本，并将Cache Line标记为Dirty；下一级存储要等到DCCI Clean、Cache Line替换等时机才会被更新。写回路径更容易产生“写端已经执行Store，但GM或Unified Buffer（UB）仍是旧值”的问题。

### 缓存一致性与内存一致性的区别​

缓存一致性关注的是同一地址或同一Cache Line在不同缓存副本之间的值是否一致。例如，某个核的DCache已经把A修改为新值，但GM中仍是旧值；或者GM已经被其他路径更新，但当前核DCache仍保留旧副本。此时问题的关键是是否需要写回Dirty副本、失效旧副本，或者绕过Cache访问。

内存一致性关注的是不同执行单元、不同线程或不同核之间的读写顺序是否符合程序预期。例如，生产者先写数据再写标志，消费者看到标志后是否一定能按预期读取到数据。同步接口、内存栅栏主要解决执行顺序和访问完成问题；DCCI主要解决Cache副本写回和失效问题。实际编程时，两类问题经常同时出现，但本文重点讨论Cache副本与GM、UB之间的数据一致性。

以下以DCache为例展示缓存一致性问题，A和B位于GM上同一条64B DCache Cache Line内：

Text

初始状态：
GM上64B地址长度的数据分布[A=0, B=0, ...]

时间点    Core 0                                  Core 1
======    ====================================    ====================================
T0        Scalar.LOAD A
// Core 0 DCache缓存整条Cache Line：
// [A=0, B=0, ...]

T1                                                Scalar.LOAD B
// Core 1 DCache缓存整条Cache Line：
// [A=0, B=0, ...]

T2        Scalar.STORE A = 1
// Core 0 DCache变成：
// [A=1, B=0, ...]

T3                                                Scalar.STORE B = 2
// Core 1 DCache变成：
// [A=0, B=2, ...]

两个核虽然修改的是不同字段，但各自DCache里保存的是整条Cache Line的副本。后续写回GM时，如果以整条Cache Line为单位写回，就可能出现不同结果：

Text

Case 1：Core 0先写回，Core 1后写回

时间点    Core 0                                  Core 1
======    ====================================    ====================================
T4        Scalar.DCCI
// Core 0以Cache Line为粒度写回：
// GM Cache Line = [A=1, B=0, ...]

T5                                                Scalar.DCCI
// Core 1以Cache Line为粒度写回：
// GM Cache Line = [A=0, B=2, ...]

Case 2：Core 1先写回，Core 0后写回

时间点    Core 0                                  Core 1
======    ====================================    ====================================
T4                                                Scalar.DCCI
// Core 1以Cache Line为粒度写回：
// GM Cache Line = [A=0, B=2, ...]

T5        Scalar.DCCI
// Core 0以Cache Line为粒度写回：
// GM Cache Line = [A=1, B=0, ...]

期望结果是： GM Cache Line = [A=1, B=2, ...]

但是因为每个核都有自己的缓存数据，各个核都是在自己的缓存旧副本上修改同一条Cache Line中的不同字段，所以实际可能结果是：

Case 1：Core 0先写回，Core 1后写回，最终GM Cache Line = [A=0, B=2, ...]

Case 2：Core 1先写回，Core 0后写回，最终GM Cache Line = [A=1, B=0, ...]

这个例子由于没有保证缓存一致性，最终没有得到期望的GM Cache Line = [A=1, B=2, ...]。

### 缓存一致性​

### DCache与GM的缓存一致性​

DCache与GM的一致性是NPU缓存一致性中最常见的一类问题。Scalar普通读写GM时会经过当前核内的DCache，而每个AIC/AIV核内都有独立的DCache用于缓存GM里的数据。某个核通过DCache修改GM地址对应的数据后，新值可能先停留在本核DCache中，GM和其他核DCache中的副本不会自动更新。

多个核或多条访问路径操作同一段GM数据时，DCache与GM的不一致通常来自两类副本：

写端Dirty副本：生产者已经写入，但新值仍停留在生产者DCache中，GM尚未更新。

读端旧副本：消费者此前已经缓存过同一地址或同一Cache Line，后续读取继续命中本核DCache旧值。

Scalar对GM的普通读写行为可以概括为：

Scalar读GM后，数据会进入DCache；后续读取同一条Cache Line内的数据时，可能直接从DCache读取，不再访问GM。

Scalar写GM时，只会修改DCache中的Cache Line，并把该Cache Line标记为Dirty。Dirty表示DCache中的Cache Line数据与GM中的数据不一致，但是并不会立刻将Dirty数据写回GM，而是需要用户手动刷新或者该数据段被替换时才会写回GM。

所以，生产侧要关心Dirty数据是否已经写回GM；消费侧要关心本核DCache里是否还保留旧副本。DCCI负责处理“副本是否写回或失效”，DSB和流水同步只负责处理“后续访问是否等到前序访问完成”。

### Scalar访问GM带来的缓存一致性​

Scalar普通读写GM时，Scalar.STORE已经执行不等于GM已经被改写。下面用两个核访问同一GM地址A说明生产侧Dirty副本和消费侧旧副本问题，示例中的T0、T1等时间点只表示代码逻辑先后顺序。

Text

初始状态：
GM[A] = 0
Core 0 DCache中没有缓存A
Core 1 DCache中没有缓存A

时间点    Core 0                                  Core 1
======    ====================================    ====================================
T0        Scalar.STORE A = 1
// Core 0 DCache[A] = 1, Dirty
// GM[A]仍可能是0

T1                                                // 如果此时Scalar.LOAD A -> 0
// Core 1可能从GM加载旧值0

T2        Scalar.DCCI A
T3        Scalar.DSB
// Dirty Cache Line写回完成，GM[A] = 1

T4                                                Scalar.LOAD A -> 1
// 如果Core 1没有旧DCache副本，可以读到新值1

如果Core 1在T3之前没有缓存过A，T4再执行Scalar.LOAD A可以从GM获取新值1；如果Core 1缓存了旧值，则需要继续处理读端旧副本。

如果读端本核DCache中已经缓存过同一地址，还需要考虑旧副本。下面示例中，消费侧使用DCCI读取数据前刷新指定GM地址所在Cache Line：

Text

初始状态：
GM[A] = 0
Core 0 DCache中没有缓存A
Core 1 DCache中没有缓存A

时间点    Core 0                                  Core 1
======    ====================================    ====================================
T0        Scalar.LOAD A -> 0                      Scalar.LOAD A -> 0
// Core 0 DCache[A] = 0                 // Core 1 DCache[A] = 0

T1        Scalar.STORE A = 1
// Core 0 DCache[A] = 1, Dirty
// GM[A]仍可能是0

T2        Scalar.DCCI A

T3        Scalar.DSB
// Dirty Cache Line写回完成，GM[A] = 1

T4                                                Scalar.LOAD A -> 0
// Core 1 DCache[A] = 0，Core 1可能继续命中本核旧DCache副本

T5                                                Scalar.DCCI A
// Core 1旧Cache副本失效

T6                                                Scalar.LOAD A -> 1
// Core 1 DCache[A] = 1，Core 1重新从GM获取新值

上面两个例子分别对应写端Dirty副本和读端旧副本。Scalar路径的Cache处理原则是：

生产者通过Scalar写GM后，如果消费者需要从GM或其他路径读取该数据，生产者需要执行DCCI（例如asc_dcci_single）把Dirty数据写回GM。

消费者如果担心本核DCache中有旧副本，读取前需要执行DCCI使旧副本失效。

如果算法本身不需要DCache缓存，可以使用ReadGmBypassDCache和WriteGmBypassDCache直接读写GM。

还有一类问题来自Cache Line粒度。普通Scalar写后续以64B Cache Line为单位写回；多个核即使写的是同一条Cache Line中的不同字段，也可能在各自写回整条Cache Line时覆盖对方结果。此类场景可以通过让不同核操作的地址至少相隔Cache Line大小，或使用ReadGmBypassDCache和WriteGmBypassDCache降低风险。

### Atomic访问导致的DCache缓存一致性​

NPU架构版本3510支持Scalar原子操作，Scalar原子操作可以看成对某个GM地址执行的读-改-写事件，但是Scalar原子操作会绕过DCache，因此会导致缓存一致性。例如C API asc_atomic_add，该接口在指定GM地址上执行原子加操作。原子操作保证的是这一次读-改-写不会被其他原子更新打断，但它不会自动清理同一地址在DCache中的旧副本或Dirty副本。

因此，如果同一GM地址此前被普通Scalar路径访问过，就需要分析原子事件和DCache事件的一致性问题：

若普通Scalar写先发生，数据可能仍停留在DCache中。后续原子操作如果依赖这个写入，需要先把相关DCache Line写回GM。

若原子操作先发生，本核DCache中可能仍保留旧副本。后续再用普通Scalar读同一地址时，需要先使旧副本失效，避免读到原子操作之前的值。

下面用地址A和asc_atomic_add说明Atomic与DCache混用时的问题：

Text

Case 1：普通Scalar写在前，Atomic在后
初始状态：
GM[A] = 0
Core 0 DCache中没有缓存A

时间点    Core 0
======    ====================================
T0        Scalar.STORE A = 1
// Core 0 DCache[A] = 1, Dirty
// GM[A]仍可能是0

T1        asc_atomic_add(A, 1)
// Atomic绕过DCache访问GM
// 如果Atomic从GM旧值0开始累加，GM[A]可能变成1，而不是期望的2

上面的关键不是asc_atomic_add本身不原子，而是原子操作看到的GM值没有包含此前停留在DCache中的普通Scalar写入。

反过来，如果asc_atomic_add(A, 1)已经把GM更新为1，而本核DCache里还保留A == 0的旧副本，后续普通Scalar读也可能继续读到旧副本。

Text

Case 2：Atomic在前，普通Scalar读在后
初始状态：
GM[A] = 0
Core 0 DCache中没有缓存A

时间点    Core 0
======    ====================================
T0        Scalar.LOAD A -> 0
// Core 0 DCache[A] = 0, Clean

T1        asc_atomic_add(A, 1)
// Atomic绕过DCache更新GM
// GM[A] = 1
// Core 0 DCache[A]仍可能是0

T2        Scalar.LOAD A -> 0
// 普通Scalar读可能继续命中本核旧DCache副本

为了避免上述问题，Atomic访问通常需要处理缓存一致性问题。正确的处理流程应该是：

Case 1：普通Scalar写在前，Atomic在后，需要先把DCache写回GM，再执行Atomic操作。

Case 2：Atomic在前，普通Scalar读在后，需要先使DCache旧副本失效，再执行普通Scalar读。

以下是正确流程示例：

Text

Case 1：普通Scalar写在前，Atomic在后的正确处理流程
初始状态：
GM[A] = 0
Core 0 DCache中没有缓存A

时间点    Core 0
======    ====================================
T0        Scalar.STORE A = 1
// Core 0 DCache[A] = 1, Dirty
// GM[A]仍可能是0

T1        Scalar.DCCI A
T2        Scalar.DSB
// Dirty Cache Line写回完成，GM[A] = 1

T3        asc_atomic_add(A, 1)
// Atomic从GM[A] = 1开始累加
// GM[A] = 2

T4        Scalar.DSB
// 等待Atomic产生的GM访问完成

Text

Case 2：Atomic在前，普通Scalar读在后的正确处理流程
初始状态：
GM[A] = 0
Core 0 DCache中没有缓存A

时间点    Core 0
======    ====================================
T0        Scalar.LOAD A -> 0
// Core 0 DCache[A] = 0, Clean

T1        asc_atomic_add(A, 1)
T2        Scalar.DSB
// 等待Atomic产生的GM访问完成，GM[A] = 1

T3        Scalar.DCCI A
// Core 0旧Cache副本失效

T4        Scalar.LOAD A -> 1
// 重新从GM获取Atomic更新后的新值

### 软件维护Scalar访问GM的DCache一致性​

本节介绍如何从软件层面维护Scalar访问GM时导致的DCache一致性。硬件仅维护消费者和生产者都来自同核Scalar DCache操作的场景下的缓存一致性，不会主动维护各核DCache之间的一致性，也不会主动处理Scalar DCache与其他GM访问路径之间的旧副本或Dirty副本问题。因此，当同一GM地址或同一条Cache Line被不同访问路径先后访问时，需要开发者使用DCCI、DSB、流水同步等方式从软件层面维护缓存一致性。

Scalar写GM可以分为两类：

cache: 普通Scalar写通过DCache修改GM，例如直接写__gm__地址；

no cache: 不经过DCache，直接向GM发起写操作，例如WriteGmBypassDCache等。

按照访问顺序，常见需要软件维护一致性的场景可以分为RAW、WAW和WAR三类：

### RAW：先写后读​

RAW（Read After Write）场景要求后序读操作读取到前序写操作产生的新值。维护RAW场景下的缓存一致性时，需要同时处理写端和读端：

写端如果通过Scalar DCache写GM，可能只形成Dirty副本，GM中的数据尚未更新，因此通常需要在消费者读取前执行写端DCCI和DSB。

读端如果通过可能持有旧副本的带Cache路径读取，需要在真正读取前按表执行读端DCCI。

如果写端和读端位于不同流水，还需要插入对应流水同步，保证读操作不会早于写后处理执行。

下表列出同一AI Core内Scalar写GM之后、消费者读取GM完整的软件维护操作序列。假设生产者和消费者访问同一地址，每个单元格都是从写后处理到读前处理的完整操作序列。

表中的cache表示经过cache的访问，no cache表示不经过cache的访问,VF_SIMT表示经过cache的SIMT线程函数，MTE2表示经过MTE2的读取操作，MTE2（cache）指多维数据搬运NDDMA，MTE2.DCI指NDDMA Cache的失效操作或asc_ndim_copy_dci。

后续读取方式（列，后发生）
╲
前序写入方式（行，先发生） | Scalar（cache） | Scalar（no cache） | VF_SIMT（cache） | VF_SIMT（no cache） | MTE2（cache） | MTE2（no cache） |

Scalar（cache） | 先写Scalar.STORE
后读Scalar.LOAD | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
后读Scalar.LOAD | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
sync S→V/VF
后读SIMT.DCCI
后读VF_SIMT.LOAD | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
sync S→V/VF
后读VF_SIMT.LOAD | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
sync S→MTE2
后读MTE2.DCI
后读MTE2.LOAD | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
sync S→MTE2
后读MTE2.LOAD |

Scalar（no cache） | 先写Scalar.STORE
先写Scalar.DSB
后读Scalar.DCCI
后读Scalar.LOAD | 先写Scalar.STORE
先写Scalar.DSB
后读Scalar.LOAD | 先写Scalar.STORE
先写Scalar.DSB
sync S→V/VF
后读SIMT.DCCI
后读VF_SIMT.LOAD | 先写Scalar.STORE
先写Scalar.DSB
sync S→V/VF
后读VF_SIMT.LOAD | 先写Scalar.STORE
先写Scalar.DSB
sync S→MTE2
后读MTE2.DCI
后读MTE2.LOAD | 先写Scalar.STORE
先写Scalar.DSB
sync S→MTE2
后读MTE2.LOAD |

例如，前序写入方式是Scalar（cache），后续读取方式是VF_SIMT（cache）时，对应单元格要求执行：

Text

Scalar.STORE -> Scalar.DCCI -> Scalar.DSB -> sync S→V/VF -> SIMT.DCCI -> VF_SIMT.LOAD

含义是：先通过Scalar.STORE完成写入；由于该写入可能停留在Scalar DCache中，需要执行Scalar.DCCI -> Scalar.DSB将Dirty数据写回GM并等待写回完成；随后通过sync S→V/VF保证VF_SIMT读取发生在Scalar侧处理之后；最后，VF_SIMT读前执行SIMT.DCCI，避免SIMT DCache命中旧副本，再执行VF_SIMT.LOAD读取新值。

如果前序是Scalar（no cache）写，写端不会留下Dirty副本，但写入仍需要通过Scalar.DSB等待完成；如果后续读端走带Cache路径，仍要按表执行读端DCCI处理旧副本。

### WAW：先写后写​

WAW（Write After Write）场景要求最终写入GM的是后一次写操作的数据。因为DCache写回的原因，程序顺序上的后写不一定是内存序上的最后一次写。下面示例中，A和B位于同一条64B DCache Cache Line内。前一次写通过Scalar DCache修改A，后一次写通过MTE3修改B。虽然MTE3写发生在后，但如果Scalar DCache中的Dirty Cache Line最后才写回GM，就可能把B恢复成Scalar缓存副本中的旧值：

Text

初始状态：
GM Cache Line = [A=0, B=0, ...]
A和B位于同一条64B DCache Cache Line内。

时间点    Scalar流水                              MTE3流水
======    ====================================    ====================================
T0        Scalar.STORE A = 1
// Scalar DCache中形成Dirty Cache Line：
// [A=1, B=0, ...]
// GM Cache Line仍可能是[A=0, B=0, ...]

T1                                                MTE3.STORE B = 2
// MTE3直接写GM：
// GM Cache Line = [A=0, B=2, ...]

T2        Scalar.DCCI A
T3        Scalar.DSB
// Scalar Dirty Cache Line按64B写回GM：
// GM Cache Line = [A=1, B=0, ...]
// MTE3对B的后写被前序Scalar Dirty副本覆盖

因此，WAW场景的维护重点是避免前序Scalar Dirty副本在后写之后才写回GM。软件应先按表处理前一次写留下的Dirty副本或前序GM写等待，再执行后一次写；后一次写是否需要进一步等待完成或发布，由后一次写的访问路径决定。上面的例子应改成：

Text

初始状态：
GM Cache Line = [A=0, B=0, ...]
A和B位于同一条64B DCache Cache Line内。

时间点    Scalar流水                              MTE3流水
======    ====================================    ====================================
T0        Scalar.STORE A = 1
// Scalar DCache中形成Dirty Cache Line：
// [A=1, B=0, ...]

T1        Scalar.DCCI A
T2        Scalar.DSB
// 前序Scalar Dirty Cache Line写回完成：
// GM Cache Line = [A=1, B=0, ...]

T3        sync S→MTE3
// 建立Scalar流水到MTE3流水的执行顺序

T4                                                MTE3.STORE B = 2
// GM Cache Line = [A=1, B=2, ...]

下表列出从前序写开始、完成必要的前序写处理和流水顺序、到后续写发起为止的操作序列。维护WAW场景下缓存一致性时先按行找到前一次写路径，再按列找到后一次写路径，按单元格顺序执行即可。表中的后写xxx.STORE只表示后一次写操作已经发起，如果后一次写需要立刻对GM或其他访问路径可见，仍需要在后写完成后按对应写路径执行必要的DCCI操作。

后续写入方式（列，后发生）
╲
前序写入方式（行，先发生） | Scalar（cache） | Scalar（no cache） | VF_SIMT（cache） | VF_SIMT（no cache） | MTE3 | FixPipe |

Scalar
（cache） | 先写Scalar.STORE
后写Scalar.STORE | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
后写Scalar.STORE | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
sync S→V/VF
后写VF_SIMT.STORE | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
sync S→V/VF
后写VF_SIMT.STORE | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
sync S→MTE3
后写MTE3.STORE | 先写Scalar.STORE
先写Scalar.DCCI
先写Scalar.DSB
sync S→FixPipe
后写FixPipe.STORE |

Scalar
（no cache） | 先写Scalar.STORE
先写Scalar.DSB
后写Scalar.STORE | 先写Scalar.STORE
先写Scalar.DSB
后写Scalar.STORE | 先写Scalar.STORE
先写Scalar.DSB
sync S→V/VF
后写VF_SIMT.STORE | 先写Scalar.STORE
先写Scalar.DSB
sync S→V/VF
后写VF_SIMT.STORE | 先写Scalar.STORE
先写Scalar.DSB
sync S→MTE3
后写MTE3.STORE | 先写Scalar.STORE
先写Scalar.DSB
sync S→FixPipe
后写FixPipe.STORE |

如果用户无法保证多个写者之间的先后顺序，应避免多个核通过DCache写同一地址或同一Cache Line。

### WAR：先读后写​

WAR（Write After Read）场景要求前序读操作读取到后续写操作发生之前的旧值。此类场景的关键不是写入结果，而是先保证读操作已经完成，再允许后续写发起。否则，后写可能过早改变GM中的值，导致前序读观察到本不应该看到的新值。

下表列出前序读取方式为Scalar（cache）或Scalar（no cache）时，从前序读开始、完成必要的等待和流水顺序、到后续写发起为止的操作序列。如果是跨核场景需要用核间通知替代表中的核内sync。

后续写入方式（列，后发生）
╲
前序读取方式（行，先发生） | Scalar（cache） | Scalar（no cache） | VF_SIMT（cache） | VF_SIMT（no cache） | MTE3 | FixPipe |

Scalar（cache） | 前读Scalar.LOAD
后写Scalar.STORE | 前读Scalar.LOAD
前读Scalar.DSB
后写Scalar.STORE | 前读Scalar.LOAD
sync S→V/VF
后写VF_SIMT.STORE | 前读Scalar.LOAD
sync S→V/VF
后写VF_SIMT.STORE | 前读Scalar.LOAD
sync S→MTE3
后写MTE3.STORE | 前读Scalar.LOAD
sync S→FixPipe
后写FixPipe.STORE |

Scalar（no cache） | 前读Scalar.LOAD
前读Scalar.DSB
后写Scalar.STORE | 前读Scalar.LOAD
前读Scalar.DSB
后写Scalar.STORE | 前读Scalar.LOAD
sync S→V/VF
后写VF_SIMT.STORE | 前读Scalar.LOAD
sync S→V/VF
后写VF_SIMT.STORE | 前读Scalar.LOAD
sync S→MTE3
后写MTE3.STORE | 前读Scalar.LOAD
sync S→FixPipe
后写FixPipe.STORE |

WAR场景下维护缓存一致性的目标是读取写前旧值，维护重点是确保这个读已经完成。表中的DSB只负责等待前序GM访问完成，不能替代流水间或核间同步。sync S→V/VF、sync S→MTE3和sync S→FixPipe只用于同一AI Core内建立流水顺序。如果前序访问和后续访问位于不同AI Core，表中的核内流水同步不能跨核生效，还需要按以下顺序组织跨核通信：

Text

前序访问核：完成前序访问侧Cache处理和必要DSB -> 发送核间通知
后续访问核：等待核间通知 -> Cache副本处理 -> 访问数据

### ICache与GM的缓存一致性​

ICache用于缓存Scalar单元最近使用或频繁使用的指令。Scalar读取GM指令时，优先从ICache读取；ICache未命中时，再从L2 Cache读取；L2 Cache也未命中时，再从GM加载指令并更新到L2 Cache和ICache。相关说明见系统缓存概述。

从缓存一致性角度看，ICache缓存的是指令而不是普通数据，并且是只读Cache，因此一般不涉及普通数据路径上的Cache副本不一致问题。

### SIMT DCache与GM的缓存一致性​

### SIMT DCache访问路径​

NPU架构版本3510新增SIMT相关硬件单元。SIMT访存会涉及SIMT DCache：SIMT DCache复用UB中的一部分空间，用于缓存SIMT线程访问GM时的数据。SIMT DCache空间可配置范围为32KB到128KB，实际大小由UB总大小扣除静态内存、动态内存和预留空间后确定，若剩余空间超过128KB，SIMT DCache大小固定为128KB。SIMT也提供asc_ldcg、asc_stcg等接口表达不占用或绕过SIMT DCache的访问意图，因此本节按带Cache路径和no cache路径分别分析。

SIMT读GM的底层实现：先从SIMT DCache读取，未命中再访问L2 Cache或GM，并把数据缓存到L2 Cache和SIMT DCache中。L2 Cache与GM之间的数据一致性由硬件保证，但SIMT DCache与GM之间不保证一致性。因此，如果同一GM地址已经被其他核、Scalar路径或MTE路径更新，而当前SIMT DCache中仍有旧副本，后续SIMT读可能继续命中旧值。

SIMT DCache采用直写（write-through）策略：

SIMT写GM时，写入会进入SIMT DCache，同时写出到GM，不形成需要后续Clean回GM的Dirty副本；

no cache写入不经过SIMT DCache，也不会在SIMT DCache中缓存写入数据。no cache写GM时会检查该地址的数据是否已缓存在SIMT DCache里，如果命中，硬件会自己执行DCCI。

判断SIMT写GM后是否需要Cache处理，可以按两步看：

先看写端是否需要等待GM写完成：SIMT DCache是直写Cache，写端不需要执行DCCI把Dirty副本写回GM。

再看读端是否可能命中旧副本：如果消费者通过Scalar或MTE2等其他带Cache路径读取，读端路径中可能已有旧副本，需要在读前按表执行对应的DCCI。

如果线程或核之间还通过GM上的标志位通信，需要另行使用asc_threadfence等内存栅栏处理写入顺序关系。内存栅栏属于内存顺序控制，不会自动写回生产者侧Cache，也不会自动失效消费者侧Cache。

### 软件维护VF_SIMT访问GM的缓存一致性​

本节从软件维护角度说明VF_SIMT访问GM时如何维护SIMT DCache一致性。硬件只维护单核内SIMT流水的缓存一致性，不维护SIMT和其他流水的一致性。由于SIMT DCache采用直写策略，VF_SIMT写GM不会形成需要后续Clean回GM的Dirty副本，但写操作是否已经完成、读端是否可能命中旧副本、不同流水之间是否已经建立顺序，仍需要由软件按访问依赖关系处理。

表中的两类VF_SIMT访问路径含义如下：

VF_SIMT（cache）：写入经过SIMT DCache直写路径，数据会缓存至SIMT DCache和L2 Cache，同时写出到GM。

VF_SIMT（no cache）：写GM不经过SIMT DCache，不在SIMT DCache中缓存写入数据。

由于SIMT DCache为直写Cache，且no cache路径也不会产生Dirty副本，二者在RAW、WAW、WAR场景下需要的软件一致性维护动作基本相同，下文表格将VF_SIMT（cache）和VF_SIMT（no cache）合并为VF_SIMT（cache/no cache）。

下面按RAW、WAW和WAR三类依赖关系分析。本节表格同样按"先看行、再看列"阅读：行表示先发生的访问，列表示后发生的访问；单元格中的操作按从上到下的顺序执行。表中的xxx.STORE表示对应路径的写操作，xxx.LOAD表示对应路径的读操作。SIMT.DCCI用于使SIMT DCache中的旧副本失效，sync X→Y用于同一AI Core内建立流水X到流水Y的执行顺序，Scalar.DCCI和MTE2.DCI表示读端在读取前处理本路径可能存在的旧Cache副本。DSB和sync不改变Cache副本状态，不能替代DCCI。

### RAW：先写后读​

RAW场景要求后续读操作读取到VF_SIMT前序写操作产生的新值。由于SIMT DCache是直写Cache，写端不需要执行DCCI写回Dirty数据；如果后续读端是Scalar或MTE2等可能持有旧副本的带Cache路径，还需要在读前执行对应DCCI，避免命中旧副本。

下表描述核内VF_SIMT作为生产者写GM之后、消费者真正读取之前，生产者和消费者需要按顺序执行哪些Cache处理以及必要的等待/流水顺序操作。

后续读取方式（列，后发生）
╲
前序写入方式（行，先发生） | Scalar（cache） | Scalar（no cache） | VF_SIMT（cache） | VF_SIMT（no cache） | MTE2（cache） | MTE2（no cache） |

VF_SIMT
（cache/no cache） | 先写VF_SIMT.STORE
sync V/VF→S
后读Scalar.DCCI
后读Scalar.LOAD | 先写VF_SIMT.STORE
sync V/VF→S
后读Scalar.LOAD | 先写VF_SIMT.STORE
sync SIMT->SIMT（同线程内无需同步）
后读VF_SIMT.LOAD | 先写VF_SIMT.STORE
sync SIMT->SIMT（同线程内无需同步）
后读VF_SIMT.LOAD | 先写VF_SIMT.STORE
sync V/VF→MTE2
后读MTE2.DCI
后读MTE2.LOAD | 先写VF_SIMT.STORE
sync V/VF→MTE2
后读MTE2.LOAD |

例如，前序写入方式是VF_SIMT（cache/no cache），后续读取方式是普通Scalar（cache）时，需要执行：

Text

VF_SIMT.STORE -> sync V/VF→S -> Scalar.DCCI -> Scalar.LOAD

含义是：SIMT DCache是直写Cache，SIMT写GM时新值会写出到GM；消费者读前再通过sync V/VF→S保证同一AI Core内Scalar读取发生在VF_SIMT写入之后，并执行Scalar.DCCI避免Scalar DCache中已有旧副本；之后Scalar读取才能从正确位置获得新值。

其他单元格按同样方式阅读：先完成行中描述的VF_SIMT写，再根据列中后续读取路径补充等待、流水同步以及读端Cache失效操作。是否需要读端DCCI，取决于后续读取路径是否可能命中旧Cache副本。

其中，表格里的sync SIMT->SIMT（同线程内无需同步）指如果是线程间操作同地址，则需要通过asc_syncthreads建立阶段同步；如果是SIMT的某个单线程内VF_SIMT写入和读取同地址无需同步。

### WAW：先写后写​

WAW场景要求最终写入GM的是后一次写操作的数据。由于SIMT DCache是直写Cache，前一次VF_SIMT写不会留下待写回GM的Dirty副本；软件维护的重点是按表保证前后两次写的执行顺序，避免前一次写延迟到达GM后影响后一次写的结果。

下表列出从前序写开始、完成必要的前序写等待和流水顺序、到后续写发起为止的操作序列。使用时先按行选择前一次VF_SIMT写路径，再按列选择后一次写路径，按单元格顺序执行。表中的后写xxx.STORE只表示后一次写操作已经发起，不表示后一次写已经完成，也不表示后一次写结果已经对GM或其他访问路径可见；后一次写完成后的等待和发布，需要按后写路径另行处理。

后续写入方式（列，后发生）
╲
前序写入方式（行，先发生） | Scalar（cache） | Scalar（no cache） | VF_SIMT（cache） | VF_SIMT（no cache） | MTE3 | FixPipe |

VF_SIMT
（cache/no cache） | 先写VF_SIMT.STORE
sync V/VF→S
后写Scalar.STORE | 先写VF_SIMT.STORE
sync V/VF→S
后写Scalar.STORE | 先写VF_SIMT.STORE
sync SIMT->SIMT（同线程内无需同步）
后写VF_SIMT.STORE | 先写VF_SIMT.STORE
sync SIMT->SIMT（同线程内无需同步）
后写VF_SIMT.STORE | 先写VF_SIMT.STORE
sync V/VF→MTE3
后写MTE3.STORE | 先写VF_SIMT.STORE
sync V/VF→FixPipe
后写FixPipe.STORE |

如果后一次写需要立刻对GM或其他访问路径可见，仍需要在后写完成后按对应写路径执行必要的等待或发布操作。例如，后写为Scalar（cache）时，需要执行后写Scalar.DCCI -> 后写Scalar.DSB；后写为VF_SIMT、MTE3或FixPipe时，需要等待对应写GM操作完成。

### WAR：先读后写​

WAR场景要求前序读操作读取到后续写操作发生之前的旧值。此类场景不需要把新值尽快发布给读端，而是需要先确认VF_SIMT侧读操作已经完成，再允许后续写端执行写GM操作或发送写完成通知。

下表列出前序读取方式为VF_SIMT（cache/no cache）时，从前序读开始、完成必要等待和流水顺序、到后续写发起为止的操作序列。使用时先按行选择前序VF_SIMT读路径，再按列选择后续写路径；单元格中的前读VF_SIMT.LOAD表示旧值读取点，后写xxx.STORE表示新值写入点。跨核场景需要用核间通知替代表中的核内sync。

后续写入方式（列，后发生）
╲
前序读取方式（行，先发生） | Scalar（cache） | Scalar（no cache） | VF_SIMT（cache） | VF_SIMT（no cache） | MTE3 | FixPipe |

VF_SIMT
（cache/no cache） | 前读VF_SIMT.LOAD
sync V/VF→S
后写Scalar.STORE | 前读VF_SIMT.LOAD
sync V/VF→S
后写Scalar.STORE | 前读VF_SIMT.LOAD
sync SIMT->SIMT（同线程内无需同步）
后写VF_SIMT.STORE | 前读VF_SIMT.LOAD
sync SIMT->SIMT（同线程内无需同步）
后写VF_SIMT.STORE | 前读VF_SIMT.LOAD
sync V/VF→MTE3
后写MTE3.STORE | 前读VF_SIMT.LOAD
sync V/VF→FixPipe
后写FixPipe.STORE |

WAR表不要求读端失效Cache副本，因为目标是读取写前旧值；维护重点是等待前序读完成并建立到后续写路径的顺序。若前序访问和后续访问位于不同AI Core，表中的sync V/VF→S、sync V/VF→MTE3和sync V/VF→FixPipe不能跨核生效，应先在前序访问所在核完成读取之后发送核间通知；后序访问所在核收到通知后，再处理本核读端或写端可能存在的Cache副本。

### UB与DCache的缓存一致性​

UB与DCache的一致性问题主要出现在Scalar开启DCache模式读取UB的场景：Scalar访问UB路径时可以CTRL[49]控制Scalar读取UB时是否通过DCache读取；CTRL[49] = 1'b0时，Scalar直接从UB读取；CTRL[49] = 1'b1时，Scalar读取UB数据会经过DCache。相关说明见asc_dcci和asc_set_ctrl()。

### Scalar以DCache模式访问UB带来的缓存一致性​

Scalar读取UB时默认是直接读取UB，但是如果用户显式设置CTRL[49] = 1'b1，Scalar以DCache模式读取UB时，此时需要处理Scalar以DCache模式访问UB带来的缓存一致性问题。

Scalar读UB后，数据进入DCache；后续Scalar再次读取同一段UB地址时，可能继续命中DCache副本，而不是直接读取UB。下面用同一核内的一个UB地址A说明旧副本问题。

Text

初始状态：
UB[A] = 0
Scalar DCache中没有缓存A
CTRL[49] = 1'b1    // Scalar读UB时从DCache读取

时间点    Vector流水                              Scalar流水
======    ====================================    ====================================
T0                                                Scalar.LOAD UB[A] -> 0
// Scalar DCache缓存A所在Cache Line

T1        Vector.STORE UB[A] = 1
// UB[A] = 1

T2        sync V→S
// 保证Scalar后续读取发生在Vector写UB之后

T3                                                Scalar.LOAD UB[A] -> 0
// Scalar可能继续命中旧DCache副本
// 即使UB[A]已经是1，也可能读到0

这个场景包含两个独立问题：一是Vector写UB是否已经完成，二是Scalar DCache中是否已经缓存过旧UB副本。即使Vector写UB已经完成，Scalar也可能继续命中旧DCache副本。因此，消费者通过Scalar DCache模式读取UB前，需要同时处理流水顺序和DCache旧副本：

Text

初始状态：
UB[A] = 0
Scalar DCache中没有缓存A
CTRL[49] = 1'b1    // Scalar读UB时从DCache读取

时间点    Vector流水                              Scalar流水
======    ====================================    ====================================
T0                                                Scalar.LOAD UB[A] -> 0
// Scalar DCache缓存A所在Cache Line

T1        Vector.STORE UB[A] = 1
// UB[A] = 1

T2        sync V→S
// 保证Scalar后续读取发生在Vector写UB之后

T3                                                Scalar.DCCI
// Scalar DCache中缓存的UB[A]旧副本失效

T4                                                Scalar.DSB
// 等待DCCI相关操作完成

T5                                                Scalar.LOAD UB[A] -> 1
// Scalar重新从UB获取新值

其中，sync V->S用于保证同一AI Core内Scalar读取发生在Vector写UB之后；Scalar.DCCI用于处理DCache与UB的一致性，使Scalar后续读取不再使用旧DCache副本；Scalar.DSB用于等待DCCI相关操作完成。若生产者是MTE2把GM数据搬入UB，也需要先建立MTE2 -> S的流水顺序，再处理Scalar DCache中的旧UB副本。

### Cache处理原则​

C API中提供了asc_ub_dcci_single、asc_dcci_entire_ub等面向UB地址空间的DCCI接口。实际使用时，应根据当前产品和CANN版本选择支持的UB DCCI接口；DCCI发出后，还需要使用DSB等待相关操作完成，再执行后续Scalar读取。

实际编程时，应先判断消费者从哪里读取，再选择对应的一致性目标：

如果后续消费者以Scalar DCache模式直接读取UB，关注的是DCache与UB的一致性，应使用DCCI处理旧副本，并在读取前建立生产流水到Scalar流水的同步。

如果Scalar读取UB时没有开启DCache模式，即CTRL[49] = 1'b0，Scalar直接从UB读取，此时通常只需要处理Scalar流水与Vector/MTE流水之间的执行顺序，不需要通过DCCI处理旧DCache副本。

### 原子操作与一致性​

原子操作用于多个执行单元、多个线程或多个核并发更新同一地址时，将一次读-改-写过程作为不可被其他更新打断的事件。原子操作可以避免同一地址上的更新丢失，但不等价于缓存一致性处理，也不等价于同步屏障。判断atomic操作下的缓存一致性，需要同时看两个问题：atomic操作实际更新的是GM还是UB，以及同一地址是否还被普通DCache、SIMT DCache或MTE等其他路径访问。

下面按atomic事件发起路径分类说明：Scalar原子、MTE3随路原子、FixPipe随路原子和SIMT原子。本节只说明这些atomic事件放入一致性分析时需要关注的Cache行为，接口细节可参考对应API文档。

### Scalar原子操作与DCache一致性​

NPU架构版本3510支持Scalar原子操作，Scalar原子操作和普通Scalar读写GM混用时，需要按DCache与GM的一致性来分析，详细处理方式参考Atomic访问导致的DCache缓存一致性。

### MTE3随路原子操作与DCache一致性​

随路原子操作不是单独读取某个地址的Scalar事件，而是改变后续目的地址为GM的数据搬运行为。MTE3负责将UB等核内数据搬出到GM，因此UB -> GM是常见的随路原子使用路径。接口说明可参考C API asc_set_atomic_add和asc_disable_dma_atomic。

从一致性角度看，MTE3随路原子的目标是GM，不会自动处理Scalar DCache中已有的副本，因此需要关注以下关系：

开启随路原子前，如果GM目标地址此前被普通Scalar写过，需要先确保Scalar DCache中的Dirty数据已经写回GM，否则原子操作读取到的"GM已有内容"可能不是程序期望的初值。

随路原子完成后，如果后续通过普通Scalar读取同一GM地址，需要处理读端DCache旧副本，并等待MTE3写GM完成。

### FixPipe随路原子操作与DCache一致性​

FixPipe常用于L0C Buffer -> GM或相关矩阵搬出路径。对一致性分析来说，FixPipe随路原子可以看成另一类"向GM写入的随路原子事件"：数据来源不同于MTE3，但最终更新的仍是GM目标地址。

因此，FixPipe随路原子的缓存一致性行为与MTE3随路原子类似：它不会自动清理Scalar DCache旧副本，也不会自动把此前停留在DCache中的Dirty数据合入GM。

如果GM目标地址的初值来自普通Scalar写GM，需要先按DCache缓存一致性把DCache中的Dirty数据写回GM，再执行FixPipe随路原子。

如果FixPipe随路原子后，消费者通过Scalar普通路径读取GM，需要等待FixPipe写GM完成，并在读端处理DCache旧副本。

如果多个流水或多个核都对同一GM地址做随路原子，原子操作只保证每次GM读-改-写不被打断；不同流水、不同核之间的启动顺序仍应由同步或算法协议保证。

### SIMT原子操作与缓存一致性​

NPU架构版本3510支持SIMT原子操作，SIMT原子操作可能作用于UB，也可能作用于GM；接口例子可参考asc_atomic_add。它保证的是同一地址上这一次读-改-写具有原子性，不保证多个线程之间的执行顺序。

分析缓存一致性时，需要先区分目标地址：

目标是UB时，主要关注同一Thread Block内线程之间对共享UB数据的可见性和执行顺序；如果同一段UB数据还被Scalar以DCache模式读取，需要结合处理DCache与UB的一致性。

目标是GM时，需要把SIMT原子更新GM、SIMT普通读写GM、Scalar普通读写GM、MTE搬运GM放在同一张可见性图里分析。原子操作本身不能替代SIMT DCache或Scalar DCache的旧副本处理；若后续通过默认SIMT读或Scalar普通读读取同一GM地址，仍需要按SIMT缓存一致性或DCache缓存一致性处理读端可能存在的旧Cache副本。

如果不同Thread Block或不同AI Core通过GM标志位配合SIMT原子操作通信，仍需要结合内存栅栏、原子操作、轮询协议以及必要的Cache处理建立完整同步关系，不能只依赖原子操作本身。

### 预取​

预取用于提前把后续可能访问的数据或指令放入Cache，目的是降低访问延迟。相关说明见系统缓存概述。从一致性角度看，预取会改变数据或指令进入Cache的时机，可能会导致一致性问题。

### L2 Cache预取​

L2 Cache和GM之间没有一致性问题，因此L2 Cache的预取不会导致多核间数据不一致问题。但这不表示L2 Cache预取获得的数据就是期望的：如果生产者写入的新值还停留在DCache中、MTE3写GM尚未完成，或者消费者读取顺序早于生产者写入完成，那么L2 Cache中提前加载到的仍可能是GM可见的旧值。此时仍应回到对应章节处理生产者写回、访问完成等待和读端旧副本失效。

### DCache预取​

asc_datacache_preload用于从指定GM地址预加载数据到DCache中，每次调用只能预加载一个Cache Line大小的数据。它的作用是让后续Scalar读取更可能命中DCache，而不是建立新的同步或一致性关系。

DCache是每个核内独立的缓存，因此DCache预取会在当前核DCache中提前形成GM数据副本。如果预取之后，同一GM地址被其他核、MTE、原子操作或SIMT路径更新，当前核后续普通Scalar读可能继续命中预取形成的旧副本。需要读取最新GM数据时，仍要按DCache缓存一致性执行DCCI失效旧副本，或使用绕过DCache的读接口。

### ICache预取​

asc_icache_preload用于从指令所在GM地址预加载指令到ICache中。ICache预取主要影响取指性能，不参与普通数据的生产者-消费者可见性判断，也没有一致性问题。

### 总结​

缓存一致性问题的核心是同一份数据可能同时存在于GM、DCache、SIMT DCache、UB等不同位置。软件需要根据访问路径和RAW、WAW、WAR依赖关系，显式处理Cache副本和执行顺序。

实际编程时重点关注以下规则：

写端可能存在Dirty副本时，需要通过DCCI写回，并配合DSB等待完成。

读端可能存在旧Cache副本时，需要在读取前通过DCCI失效旧副本。

DCCI处理Cache副本状态，DSB等待前序内存访问完成，流水同步或核间通知建立执行顺序，三者不能互相替代。

版本 master 提交 5f50fc4c6ada构建于 2026/10/20 11:47:29包含未提交修改

在 Gitcode 上打开此页

源文档更新于:

Pager
上一页2.7.2. 内存模型

下一页2.7.2.2. 内存一致性

免责声明：本站内容由 asc-devkit 仓 master 分支自动编译生成，属于持续开发版本，可能存在缺陷，仅供预览与参考。如需稳定及商用资料，请查阅官方 昇腾社区。

Copyright Huawei Technologies Co., Ltd.
```

### item 6

https://asc.gitcode.com/guide/programming_guide/advanced_programming/memory_model/memory_consistency.html
Huawei / cann/asc-devkit master docs (auto-built site, 版本 master 提交 5f50fc4c6ada, 构建于 2026/10/20 11:47:29); source md: https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/memory_model/memory_consistency.md
THE authoritative Ascend memory-consistency chapter. Quoted here: the flag/data producer-consumer example, the 缓存一致性 vs 内存一致性 split table, the combined T0..T7 sequence (DCCI+DSB on writer, WAIT flag, DCCI+DSB on reader, then LOAD), §4 核间内存一致性 (hardware does NOT establish global order between AI Cores; RAW/WAW/WAR risks; 核间同步原语 and the Producer/Consumer T0..T7 table), §7 快速决策参考, §8 总结 (NPU内存一致性更接近弱一致性模型).
```
### 内存一致性

本文介绍NPU编程中的内存一致性问题，说明不同执行单元、不同AI Core、不同Stream以及Host侧和Device侧之间如何建立可预期的读写顺序。本文重点讨论“操作顺序和可见性”，缓存副本写回、失效等问题请参考缓存一致性。

相关示例涉及Global Memory（GM）、Unified Buffer（UB）和Data Cache（DCache）等内存。

### 内存一致性基本概念​

### 什么是内存一致性​

内存一致性关注的是多个访问者对同一片内存的读写顺序是否符合程序预期。典型内存一致性例子是：生产者先写数据再写标志，消费者看到标志后，是否一定能读取到生产者写入的数据：

Text

初始状态：
GM[data] = 0
GM[flag] = 0

时间点    生产者                                  消费者
======    ====================================    ====================================
T0        STORE data = 1
T1        STORE flag = 1
T2  LOAD flag -> 1
T3  LOAD data -> ?

用户期望内存序和代码顺序保持一致，在flag赋值之前，data已经赋值完成，期望消费者在T2看到flag = 1后，T3一定读取到data = 1。但是在NPU上，生产者和消费者可能运行在不同流水、不同AI Core、不同Stream，甚至位于Host侧和Device侧之间。不同执行域之间通常是乱序并行的，不会自动建立全局顺序，用户需要根据访问范围插入同步、MEMBAR或通信机制，自行管理内存一致性。

### 缓存一致性与内存一致性的区别​

缓存一致性和内存一致性是两类不同问题：

类型 | 关注点 | 典型问题 | 常用处理方式 |

缓存一致性 | Cache副本和下一级存储中的值是否一致。 | DCache中保留旧值，或Dirty副本尚未写回GM。 | DCCI |

内存一致性 | 多个读写操作之间的先后关系是否符合预期。 | 消费者看到flag = 1后，是否一定能读取到生产者写入的data。 | 同步、DSB等 |

缓存一致性是“副本与最新值是否一致”，内存一致性是“指令实际执行顺序是否和代码顺序一致”。实际编程时，两类问题经常同时出现，例如生产者写入数据后，通过flag发布同步标志，消费者等待flag后再读取数据：

Text

初始状态：
GM[data] = 0
GM[flag] = 0
消费者DCache[data] = 0

时间点    生产者                                  消费者
======    ====================================    ====================================
T0        Scalar.STORE data = 1
T1        DCCI data
T2        Scalar.DSB
// 保证data写回GM完成

T3        STORE flag = 1
// 发布标志，flag本身就是同步通知

T4  WAIT flag == 1
// 等待flag变为1

T5  DCCI data
T6  Scalar.DSB
// 处理消费者侧存在的旧DCache副本

T7  Scalar.LOAD data -> 1

上例中，flag是同步标志。生产者写flag = 1表示发布通知，消费者等待flag == 1表示接收通知。DCCI用于维护data对应的Cache副本，属于缓存一致性处理；Scalar.DSB用于等待写回或失效相关操作完成；STORE flag和WAIT flag用于建立生产者和消费者之间的执行先后关系，属于内存一致性处理。

只做同步但不处理DCache副本，可能仍读到旧值；

只做DCCI但不建立执行顺序，消费者也可能过早读取。

缓存一致性或内存一致性任意一类问题没有处理好，都可能导致最终读取到的数据不符合预期。


[... §2/§3 omitted — see page ...]

### Case 4：DCache和不经过DCache的访问混合同地址访问​

同一GM地址混用DCache路径和不经过DCache路径时，硬件不保证两条路径之间自动维护顺序和Cache副本。例如前序DCache写形成Dirty副本，后序不经过DCache的写虽然直接更新GM，但后续DCache Dirty副本写回时会覆盖不经过DCache的写入：

Text

时间点    Scalar流水
======    ====================================
T0        Scalar.STORE_GM_DCache(addr) = 1
// DCache[addr] = 1，Dirty；GM[addr]仍可能是旧值

T1        Scalar.STORE_GM_no_cache(addr) = 2
// GM[addr] = 2

T2        DCache Dirty副本后续写回
// GM[addr]被覆盖为1

正确处理方式是在不经过DCache的写之前，先把同地址所在Cache Line中已经存在的DCache Dirty副本写回GM，并等待写回完成：

Text

时间点    Scalar流水
======    ====================================
T0        Scalar.STORE_GM_DCache(addr) = 1
T1        Scalar.DCCI(addr)
T2        Scalar.DSB
// 等待DCache Dirty副本写回GM完成

T3        Scalar.STORE_GM_no_cache(addr) = 2

反过来，如果不经过DCache的访问已经更新GM，而当前核DCache中已经缓存过同一地址所在Cache Line，后续要通过DCache路径读取该地址时，需要先处理DCache中的旧副本，处理方式请参考缓存一致性中DCache与GM一致性的读端维护方法。如果当前核DCache中已有同一地址所在Cache Line的Dirty副本，应在不经过DCache写之前先按前一个流程执行DCCI和Scalar.DSB，避免旧Dirty副本覆盖不经过DCache的写入。

### Scalar和其他硬件单元的内存一致性​

当Scalar和其他硬件单元之间存在数据依赖时，需要用户自行插入同步保证内存一致性。例如Scalar先写GM，后续MTE2、Vector、Fixpipe或SIMT再读取该数据，不能只依赖代码书写顺序推断后续硬件单元一定能看到Scalar写入。

下面以Scalar写GM后，MTE2读取GM为例：

Text

时间点    Scalar流水                              MTE2流水
======    ====================================    ====================================
T0        Scalar.STORE_GM(data) = 1
T1        Scalar.DCCI
// Scalar写GM经过DCache时添加DCCI
T2        Scalar.DSB
// 等待Scalar写GM完成

T3        sync S->MTE2
// 通知MTE2可以读取

T3  WAIT S->MTE2
T4  MTE2.LOAD_GM(data)

其中，Scalar.DSB可通过asc_sync_data_barrier实现，用于等待Scalar侧前序GM访问完成；sync S->MTE2可通过asc_sync_notify/asc_sync_wait建立Scalar流水到MTE2流水的先后关系：

DSB和流水同步只处理访问完成顺序和执行先后关系，保证内存一致性，不负责写回或失效DCache副本。如果Scalar写GM经过DCache，或者后续读取路径可能命中旧Cache副本，还需要参考缓存一致性处理DCache相关问题。


[... omitted ...]

### 核间内存一致性​

### 核间一致性问题与多核访问GM的顺序风险​

不同AI Core之间通常通过GM共享数据。硬件不会自动为多个AI Core建立全局读写顺序：一个核写入GM、另一个核读取GM时，如果没有显式同步，读写操作的实际完成顺序或对外可见顺序可能与代码期望不一致。如果访问路径经过DCache，还会叠加写端Dirty副本、读端旧副本等缓存一致性问题。

核间访问GM时，常见风险可以按RAW、WAW、WAR三类依赖理解。

RAW（Read After Write）要求后续读操作读取前序写操作的新值。如果消费者没有等待生产者发布数据，可能提前读到旧值；如果读端DCache中已有旧副本，也可能在同步后仍读到旧值。

Text

时间点    Core 0                                  Core 1
======    ====================================    ====================================
T0        STORE data = 1
T1  LOAD data -> 旧值或不确定值

WAW（Write After Write）要求最终保留在GM中的是后一次写操作的数据。如果两个核对同一地址或同一Cache Line内的数据写入顺序没有建立，最终结果可能不符合“后写覆盖前写”的期望；如果前序写经过DCache形成Dirty副本，后续写还可能被前序Dirty写回覆盖。

Text

时间点    Core 0                                  Core 1
======    ====================================    ====================================
T0        STORE data = 1
T1  STORE data = 2
T2        前序写入或Dirty副本晚于Core 1写入对GM可见
结果      GM中可能不是Core 1期望保留的值

WAR（Write After Read）要求前序读操作先读取到后续写操作之前的旧值。如果后写核过早发布新值，前读核可能读取到新值，破坏“先读旧值、再写新值”的阶段关系。

Text

时间点    Core 0                                  Core 1
======    ====================================    ====================================
T0                                                STORE data = 1
T1        LOAD data -> 新值
结果      Core 0没有读取到后写前的旧值

这些场景下的核间内存一致性全部需要用户手动维护。

### 核间同步原语​

核间同步原语用于在多个AI Core之间建立到达、等待或屏障关系。选择接口时，首先要确认同步范围：是所有参与核都需要对齐，还是只需要某些group之间、同一group内部的block/subblock之间建立顺序。

在分离模式下，AIC（Cube Core）与AIV（Vector Core）按group划分。一个group由1个block（主核）和N个subblock（从核）组成，比例为1:N。因此，组间同步指不同group之间的block同步，组内同步指同一个group内block和subblock之间的同步。

接口 | 同步范围 | 作用边界 |

asc_sync_inter_arrive/asc_sync_inter_wait | 组间block同步，即不同group之间的block同步。 | 通过flag_id传递同步信号，适合group之间的生产者-消费者同步。 |

asc_sync_block_arrive/asc_sync_block_wait | 组内block和subblock同步，即同一个group内AIC和AIV之间同步。 | 通过flag_id传递同步信号，适合同一group内部Cube侧和Vector侧建立顺序。 |

arrive/wait类接口只传递同步信号；flag_id需要配对使用，并避免计数器溢出和ID冲突。使用这些接口后，仍需要根据数据访问路径判断是否需要DCCI和DSB。

以核间生产者-消费者场景为例，展示生产者负责写入GM数据并发布同步信号，消费者等待同步信号后再读取GM数据的抽象流程：

Text

时间点    Producer Core                          Consumer Core
======    ====================================   ====================================
T0        STORE data = value
T1        DCCI data
T2        Scalar.DSB
T3        ARRIVE sync_point
T4  WAIT sync_point
T5  DCCI data
T6  Scalar.DSB
T7  LOAD data -> value

其中，T1和T2时刻处理生产者通过DCache写入后可能形成的Dirty副本，确保数据已经写回到GM；T3和T4时刻建立核间发布/获取顺序；T5和T6处理消费者本核可能存在的旧DCache副本。若生产者和消费者的数据访问路径均不经过DCache，可以按实际路径省略对应DCCI，但仍需要用同步原语建立跨核执行顺序。

上例中的ARRIVE sync_point和WAIT sync_point表示抽象的核间同步动作。组间block同步可以使用asc_sync_inter_arrive/asc_sync_inter_wait实现；同一group内block和subblock之间的同步可以使用asc_sync_block_arrive/asc_sync_block_wait实现。


[... omitted ...]

### 快速决策参考​

判断是否需要用户维护内存一致性时，可以先确认访问跨过了哪一类一致性边界：

一致性边界 | 首选机制 | 责任方 |

同流水内 | 同流水同步，例如asc_sync_pipe(PIPE_xxx)；Scalar访问完成使用asc_sync_data_barrier。 | 用户在同地址、重叠地址或前后访问存在依赖的位置等待前序访问完成。 |

核内跨流水 | 流水间同步，例如asc_sync_notify/asc_sync_wait、asc_lock/asc_unlock。 | 用户在生产者流水发布同步信号，在消费者流水等待后再访问数据。 |

Scalar访问GM | asc_sync_data_barrier配合必要的DCCI。 | 用户等待Scalar侧GM访问完成；若访问经过DCache，还需要按缓存一致性处理Dirty副本或旧副本。 |

多核间 | 核间同步、原子操作、必要时配合DCCI和DSB。 | 用户建立生产者Core和消费者Core之间的顺序；多核同地址读-改-写使用原子操作，普通数据传递仍需处理可见性。 |

跨Stream | Event任务级依赖。 | 用户需要通过Event处理Stream间同步。 |

Host等Device | Host侧显式同步接口，例如aclrtSynchronizeStream、aclrtSynchronizeDevice、aclrtSynchronizeEvent。 | 用户需要添加同步接口保证Host侧在读取Device结果或复用相关资源前等待对应Device任务完成。 |

### 总结​

NPU内存一致性更接近弱一致性模型。普通内存访问默认只具备有限的局部顺序，硬件不会为跨流水、跨AI Core、跨Stream以及Host-Device之间的访问自动建立全局一致的读写顺序。

因此，用户需要根据数据依赖主动维护内存一致性：在生产者和消费者之间建立同步关系，在需要时等待前序内存访问完成，并在涉及Cache副本时配合缓存一致性维护。
```

### item 7

https://asc.gitcode.com/api/SIMD-API/c_api/cache_ctrl/asc_dcci.html
Huawei / cann/asc-devkit master docs (auto-built site, 版本 master 提交 5f50fc4c6ada, 构建于 2026/10/20 11:47:29); source md: https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/c_api/cache_ctrl/asc_dcci.md
The next-gen intrinsic-level doc for the same instruction: asc_dcci_single / asc_ub_dcci_single / asc_dcci_entire_all / _ub / _out / _atomic, the Clean+Invalid decomposition, 流水类型 PIPE_S, the note that asc_load_dev/asc_store_dev bypass DCache entirely, CTRL[49] for UB-through-DCache, and the runnable example pairing asc_dcci_entire_out with asc_sync_data_barrier(mem_dsb_t::DSB_DDR).
```
### asc_dcci

### 产品支持情况​

Ascend 950PR&950DT系列产品：支持

Atlas A3系列产品：支持

Atlas A2系列产品：支持

Atlas 200I/500 A2推理产品：不支持

Atlas推理系列产品AI Core：不支持

Atlas推理系列产品Vector Core：不支持

Atlas训练系列产品：不支持

### 功能说明​

头文件路径为："c_api/cache_ctrl/cache_ctrl.h"。

如图1所示，在AI Core内部，访问Global Memory（GM）与Unified Buffer（UB）的方式分为以下两种：

DMA搬运单元读写GM，数据通过asc_copy_ub2gm和asc_copy_gm2ub等接口在UB等Local Memory和GM间交互，不经过Data Cache（DCache），因此不需要考虑Cache一致性问题。

Scalar单元访问GM，若经过DCache，首先会访问每个核内的DCache，因此存在DCache与GM的Cache一致性问题，具体原因请参考缓存一致性。

图1 DCache内存层次示意图

asc_dcci接口用于解决上述一致性问题，其功能可以拆解为两部分：

Clean：将缓存中被修改过的数据（脏数据，Dirty Data）写回到GM中，避免数据丢失。

Invalid：
将指定地址范围的Cache Line标记为“无效”，使其从缓存中移除。

保证下一次访问这些内存地址时，会从GM重新加载数据，而不是使用可能过期的缓存数据。

该接口的典型使用场景如下（以读写GM为例）：

读取GM的数据，但该数据可能在外部被其余核修改，此时需要使用asc_dcci接口，直接访问GM，获取最新数据。

用户通过Scalar单元写GM的数据，希望立刻写出，也需要使用asc_dcci接口。

说明

Scalar单元读写GM时，若调用的是asc_load_dev和asc_store_dev这一对接口，则读写数据不经过DCache，无需调用asc_dcci接口。

Scalar单元读写UB时是否经过DCache，由asc_set_ctrl接口配置的CTRL[49]位控制，该bit位默认值为1'b0，即默认不经过DCache。CTRL[49]的详细说明请参考asc_set_ctrl的常用CTRL寄存器比特位说明。

### 函数原型​

C

// 刷新传入GM地址所在的单个Cache Line
__aicore__ inline void asc_dcci_single(__gm__ void* dst)

// 刷新传入的UB地址所在的单个Cache Line
__aicore__ inline void asc_ub_dcci_single(__ubuf__ void* dst)

// 根据目标域刷新DCache中所有valid的Cache Line，目标域有以下四种取值：

// 刷新整个DCache中所有valid的Cache Line
__aicore__ inline void asc_dcci_entire_all()

// 刷新整个DCache中包含UB地址的所有valid的Cache Line
__aicore__ inline void asc_dcci_entire_ub()

// 刷新整个DCache中包含GM地址的所有valid的Cache Line
__aicore__ inline void asc_dcci_entire_out()

// 刷新整个DCache中包含与标量原子操作相关的GM地址的所有valid的Cache Line
__aicore__ inline void asc_dcci_entire_atomic()

### 参数说明​

表1 参数说明

参数名 | 输入/输出 | 描述 |

dst | 输入 | 待清理与失效的Cache Line所在地址。
硬件按Cache Line边界对齐定位目标行，传入地址无需满足Cache Line对齐。 |

### 返回值说明​

无

### 流水类型​

PIPE_S

### 约束说明​

dst为__ubuf__ void*类型的函数原型仅支持在AIV中调用。

### 调用示例​

将代码保存为example.asc后，可通过bisheng命令编译运行，其中--npu-arch参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考__NPU_ARCH__。

以Ascend 950PR&950DT系列产品（对应NPU架构为dav-3510）为例，编译运行命令如下：

Bash

bisheng example.asc -o main --npu-arch=dav-3510 && ./main

C++

// 本示例验证asc_dcci_entire_out对GM脏数据的回写效果。Scalar先在两个不同Cache Line中
// 写入101，再调用asc_dcci_entire_out将DCache中的脏数据回写到GM。Host回读后应得到101。
#include <array>
#include <cstdint>
#include <iostream>

#include "c_api/asc_simd.h"
#include "acl/acl.h"

namespace {
constexpr int32_t GM_VALUE = 11;
constexpr int32_t DIRTY_VALUE = 101;
constexpr uint32_t SECOND_LINE_OFFSET = 32;
constexpr uint32_t DATA_COUNT = 64;

__global__ __vector__ void AscDcciEntireOutKernel(__gm__ int32_t* data)
{
asc_init();
volatile __gm__ int32_t* data_gm = data;
// 两个元素相隔128B，位于不同Cache Line。普通Scalar写GM会经过DCache。
data_gm[0] = DIRTY_VALUE;
data_gm[SECOND_LINE_OFFSET] = DIRTY_VALUE;

// 刷新DCache中包含GM地址的Cache Line，并等待脏数据写回GM。
asc_dcci_entire_out();
asc_sync_data_barrier(mem_dsb_t::DSB_DDR);
}

bool CheckResult(const std::array<int32_t, DATA_COUNT>& output)
{
return output[0] == DIRTY_VALUE && output[SECOND_LINE_OFFSET] == DIRTY_VALUE;
}
} // namespace

int main()
{
std::array<int32_t, DATA_COUNT> input{};
input.fill(GM_VALUE);
aclInit(nullptr);
aclrtSetDevice(0);

int32_t* input_device = nullptr;
aclrtMalloc(reinterpret_cast<void**>(&input_device), input.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(input_device, input.size() * sizeof(int32_t), input.data(), input.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
AscDcciEntireOutKernel<<<1, 0>>>(input_device);
const aclError ret = aclrtSynchronizeDevice();
aclrtMemcpy(input.data(), input.size() * sizeof(int32_t), input_device, input.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

const bool passed = ret == ACL_SUCCESS && CheckResult(input);
std::cout << (passed ? "[Success] asc_dcci passed." : "[Failed] asc_dcci failed.") << std::endl;
aclrtFree(input_device);
aclrtResetDevice(0);
aclFinalize();
return passed ? 0 : 1;
}

版本 master 提交 5f50fc4c6ada构建于 2026/10/20 11:47:29包含未提交修改

在 Gitcode 上打开此页

源文档更新于:

Pager
上一页2.3.13.2. asc_datacache_preload

下一页2.3.13.4. asc_dci

免责声明：本站内容由 asc-devkit 仓 master 分支自动编译生成，属于持续开发版本，可能存在缺陷，仅供预览与参考。如需稳定及商用资料，请查阅官方 昇腾社区。

Copyright Huawei Technologies Co., Ltd.
```

### item 8

https://gitcode.com/cann/asc-devkit/blob/master/include/basic_api/kernel_operator_cache_intf.h
Huawei / cann/asc-devkit, master branch (raw: https://raw.gitcode.com/cann/asc-devkit/raw/master/include/basic_api/kernel_operator_cache_intf.h); commit date unknown
The three real DataCacheCleanAndInvalid declarations as shipped (GlobalTensor with dcciDst, LocalTensor with dcciDst, GlobalTensor without dcciDst) — the signature the API-reference page documents.
```cpp

template <typename T>
class LocalTensor;

template <typename T>
__aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t>& src, const T cacheOffset);

template <typename T>
__aicore__ inline void MultiDataCachelinePreload(const GlobalTensor<T>& src, const uint32_t cachelineNum);

template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst);

template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst);

template <typename T, CacheLine entireType>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst);

enum class UnitSize {
```

### item 9

https://gitcode.com/cann/asc-devkit/blob/master/include/c_api/cache_ctrl/cache_ctrl.h
Huawei / cann/asc-devkit, master branch (raw: https://raw.gitcode.com/cann/asc-devkit/raw/master/include/c_api/cache_ctrl/cache_ctrl.h); commit date unknown
The intrinsic-level header behind asc_dcci, including the arch gating (__NPU_ARCH__ 2201 memory_base_impl vs 3510 reg_base_impl).
```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
#error "cache_ctrl.h cannot be used with compile flag --enable-simt enabled."
#endif

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_CACHE_CTRL_CACHE_CTRL_H
#endif

#ifndef INCLUDE_C_API_CACHE_CTRL_CACHE_CTRL_H
#define INCLUDE_C_API_CACHE_CTRL_CACHE_CTRL_H
#include "c_api/defs/defs.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#include "impl/c_api/memory_base_impl/cache_ctrl_intf_impl.h"
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "impl/c_api/reg_base_impl/cache_ctrl_intf_impl.h"
#endif

__aicore__ inline void asc_icache_preload(const void* addr, int64_t prefetch_len);

namespace __asc_aicore {
__aicore__ inline void asc_dcci_single(__gm__ void* dst);
}

__aicore__ inline void asc_ub_dcci_single(__ubuf__ void* dst);

__aicore__ inline void asc_dcci_entire_ub();

__aicore__ inline void asc_dcci_entire_out();

__aicore__ inline void asc_dcci_entire_all();

__aicore__ inline void asc_dcci_entire_atomic();

__aicore__ inline int64_t asc_get_icache_preload_status();

__aicore__ inline void asc_datacache_preload(__gm__ uint64_t* address, int64_t offset);

__aicore__ inline void asc_dci();

__aicore__ inline void asc_set_scalar_cache_mode(asc_load_l2_cache_mode l2_cache_mode);

__aicore__ inline void asc_set_scalar_cache_mode(asc_store_l2_cache_mode l2_cache_mode);

__aicore__ inline asc_load_l2_cache_mode asc_get_scalar_load_cache_mode();

__aicore__ inline asc_store_l2_cache_mode asc_get_scalar_store_cache_mode();

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_CACHE_CTRL_CACHE_CTRL_H)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_CACHE_CTRL_CACHE_CTRL_H
#endif
```

### item 10

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_mo.h
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/include/device/gm2gm/shmem_device_mo.h); Copyright 2025, commit date unknown
The complete memory-ordering header: the ONLY normative visibility text for aclshmem_quiet and aclshmem_fence — HCCS-only => globally visible; HCCS+RDMA => only a PE's own memory updates guaranteed visible to that PE; fence is implemented identically to quiet; CPU-issued and NPU-issued quiet/fence cover only their own side.
```cpp
/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */

#ifndef SHMEM_DEVICE_MO_H
#define SHMEM_DEVICE_MO_H

#include "host_device/shmem_common_types.h"
#include "gm2gm/shmem_device_mo.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The aclshmem_quiet routine ensures completion of all operations on symmetric data objects issued by the
 *        calling PE.
 *        On systems with only scale-up network (HCCS), updates are globally visible, whereas on systems with
 *        both scale-up network HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the
 *        memory of a given PE are visible to that PE.
 *        Quiet operations issued on the CPU and the NPU only complete communication operations that were
 *        issued from the CPU and the NPU, respectively. To ensure completion of NPU-side operations from
 *        the CPU, using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
ACLSHMEM_DEVICE void aclshmem_quiet(void);
#define shmem_quiet aclshmem_quiet

/**
 * @brief In OpenACLSHMEM specification, aclshmem_fence assures ordering of delivery of Put, AMOs, and memory store routines
 *        to symmetric data objects, but does not guarantee the completion of these operations.
 *        However, due to hardware capabilities, we implemented aclshmem_fence same as aclshmem_quiet, ensuring both ordering
 *        and completion.
 *        Fence operations issued on the CPU and the NPU only order communication operations that were issued from the
 *        CPU and the NPU, respectively. To ensure completion of NPU-side operations from the CPU,
 *        using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
ACLSHMEM_DEVICE void aclshmem_fence(void);
#define shmem_fence aclshmem_fence

#ifdef __cplusplus
}
#endif

#endif
```

### item 11

https://gitcode.com/cann/shmem/blob/master/docs/api/sync_barrier_api_usage.md
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/docs/api/sync_barrier_api_usage.md); commit date unknown
The project's own Chinese ordering contract: §二 语义 (lines 28-31) separating sync / quiet / barrier / fence, and §三 Device kernel 中的使用顺序 + 约束 5/6 — 'fence 仅用于建立顺序，不作为通用的通信完成保证', 'Cube 标量单元不在这些 API 的同步保证范围内'.
````

**阻塞与流的区别：**

- **阻塞式接口**（`aclshmem_barrier` / `aclshmem_barrier_all` / `aclshmem_sync` / `aclshmem_team_sync` / `aclshmem_sync_all`）：内部会等待流上操作完成（调用 `aclrtSynchronizeStream`），返回时集合同步已经完成。
- **流式接口**（`aclshmemx_barrier_on_stream` / `aclshmemx_barrier_all_on_stream` / `aclshmemx_sync_on_stream` / `aclshmemx_sync_all_on_stream`）：仅把同步操作排队到指定流上，返回不代表同步完成；调用者需自行 `aclrtSynchronizeStream` 或依赖后续流操作保证顺序。传入 `NULL` 时使用库管理的默认流，与 Host RMA/quiet 的 on-stream 接口一致。

**屏障、同步与通信完成：**

- sync 表示指定 team 的 PE 到达同一同步点；对参与核此前的普通对称内存存储提供适用范围内的同步与可见性，但不保证此前发出的非阻塞通信已完成。Host 的 aclshmem_team_sync 与流式 sync 均应按这一最低保证使用；不要以 sync 代替通信完成操作。
- quiet 完成调用 PE 在对应 Host 或 Device 执行范围内先前发出的通信，但不等待其他 PE 到达同步点。
- barrier 先完成相应执行范围内此前发出的通信，再等待指定 team 的所有 PE 到达同步点。所有集体调用必须由 team 内 PE 按相同顺序参与。
- 在 Device kernel 中，NBI 后如需读取远端结果或复用源缓冲区，使用 barrier；或者先 quiet 完成本 PE 通信，再 sync 等待其他 PE。fence 仅用于建立顺序，不作为通用的通信完成保证。

***

## 二、接口说明

[... interface-by-interface entries omitted ...]

```

***

## 三、Device kernel 中的使用顺序

以下两种顺序均要求 team 内所有 PE 和声明参与范围内的核按相同顺序调用集体接口：

    // 方式一：NBI -> barrier
    aclshmem_putmem_nbi(dst, src, bytes, peer);
    aclshmem_barrier_all();
    // barrier 返回后，先前通信完成；可读取对应目标数据、复用源缓冲区。

    // 方式二：NBI -> quiet -> sync
    aclshmem_putmem_nbi(dst, src, bytes, peer);
    aclshmem_quiet();     // 本 PE 的先前通信完成，可复用本地源缓冲区
    aclshmem_sync_all();  // 等待所有 PE 到达；之后再读取其他 PE 的写入结果

普通对称内存 store 的同步/可见性与通信 API 的完成范围不同：sync 同步参与核在受支持的 HCCS/UB 内存域中的普通存储，但不能代替 quiet 完成 NBI；barrier 同时提供通信完成与 PE 到达同步。对混合核，Cube 标量单元不在这些 API 的同步保证范围内；参与核范围见 Device public header。Host 发出的通信应使用 Host 对应接口完成，Device 发出的通信应在 Device 执行范围内完成。

***

## 四、注意事项

1. **初始化检查**：调用前需完成 `aclshmemx_init_attr` 初始化；未初始化时，返回状态码的接口会返回 `ACLSHMEM_NOT_INITED`，`void` 接口会打印错误日志后直接返回。
2. **team 有效性**：非成员传入哨兵值 `ACLSHMEM_TEAM_INVALID` 时，`aclshmem_barrier` / `aclshmem_sync` / `aclshmem_team_sync` / `aclshmemx_barrier_on_stream` / `aclshmemx_sync_on_stream` 为空操作并返回成功。
3. **阻塞式接口内部同步**：阻塞式接口内部会 `aclrtSynchronizeStream`，因此调用返回即代表集合同步完成；流式接口则需要调用者自行同步流。
4. **参与范围**：集合同步要求 team 内所有 PE 都调用相同接口，否则会阻塞等待直至超时或卡死。
5. **完成与缓冲区复用**：sync 仅保证 PE 到达同步点，不能单独保证 NBI 完成。完成本 PE 通信后才能复用该通信的源缓冲区；所有相关 PE 完成通信并到达同步点后，才读取其它 PE 写入的目标数据。Device 使用 barrier，或按 quiet 再 sync 的顺序调用。
6. **Host 流的等待范围**：Host 阻塞接口使用并等待库默认流；流式接口仅排队，须等待该集体操作所在流或建立明确的跨流依赖。对先前在其他流发出的工作，先建立到集体操作所在流的依赖；仅等待另一条流不能证明该集体操作已完成。单独同步启动 Device kernel 的流，也不能替代 kernel 内对异步通信的 quiet 或 barrier。
````

### item 12

https://gitcode.com/cann/shmem/blob/master/docs/api/atomic_api_sync_async_comparison.md
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/docs/api/atomic_api_sync_async_comparison.md); commit date unknown
Engine-by-engine completion semantics in Chinese: MTE-backend atomics are ALL asynchronous with no quiet/fence inside, synchronised by aclshmemx_mte_quiet or kernel-runtime SetFlag/WaitFlag; the UB+MTE3 path needs MTE2_S/MTE3_S UB fences plus the MTE3_MTE2 GM fence; the Scalar AtomicAdd path bypasses UB/MTE3; RDMA fetch variants quiet internally while void variants do not.
````
---

## 二、MTE 后端接口：`aclshmemx_mte_atomic_<op>`

> 头文件：`include/device/gm2gm/engine/shmem_device_mte.h`
> 实现文件：`src/device/gm2gm/engine/shmem_device_mte.hpp`

**核心特征：所有 MTE 后端接口均为异步**，不论是否带返回值。接口仅发起硬件原子操作即返回，无 quiet/fence。

同步原语：`aclshmemx_mte_quiet()` 或 Kernel Runtime 级 `SetFlag/WaitFlag`。

### 2.1 接口清单

| 接口完整名 | 返回值 | 支持类型 | 硬件平台 |
|---|---|---|---|
| `aclshmemx_mte_atomic_fetch` | T | int32, uint32, float, int64, uint64 | 950 |

[... interface tables omitted ...]

| 调用前 | 其他单元（如 MTE2）正在写同一 UB 区域 | `SetFlag/WaitFlag` on `MTE2_S` |
| 调用后 | 后续操作依赖原子操作在 GM 中的结果 | `SetFlag/WaitFlag` on `MTE3_MTE2` |
| 调用后 | 需要复用同一 UB 区域写入新数据 | `SetFlag/WaitFlag` on `MTE3_S` |

### 2.3 `aclshmemx_mte_atomic_add` 的类型差异（重要）

`aclshmemx_mte_atomic_add` 在 Ascend950 上按类型有两种实现路径：

| 类型组 | 支持类型 | 实现方式 | 同步要点 |
|---|---|---|---|
| UB + MTE3 路径 | int8, int16, bf16, half, int32, float | Scalar 写操作数到 UB → MTE3 读 UB → MTE3 atomic add 写 GM | 需关注 UB 相关 fence（MTE2_S、MTE3_S）和 GM fence（MTE3_MTE2） |
| Scalar AtomicAdd 路径 | uint32, uint64, int64 | Scalar 单元直接执行标量 AtomicAdd 写 GM，不经过 UB 和 MTE3 | 不需要 UB/MTE3 相关 fence；仅在 Scalar↔MTE2/MTE3 有 GM 数据依赖时按通用规则同步 |

---

## 三、RDMA 后端接口：`aclshmemx_roce_atomic_<op>`


[... omitted ...]


### 8.1 异步接口 + RDMA（`aclshmem_int32_atomic_add`）

```c
// 异步 atomic add，RDMA 通路
aclshmem_int32_atomic_add(dst, 42, pe);

// ... 其他不依赖 dst 的计算 ...

// 读取结果前必须 quiet
aclshmemx_roce_quiet(pe, ub_buf, sync_id);

// 现在可以安全使用 dst 的值
int32_t result = *dst;
```

### 8.2 同步接口（`aclshmem_int32_atomic_fetch_add`）

```c
// 同步接口，调用返回即可直接使用返回值
int32_t old_val = aclshmem_int32_atomic_fetch_add(dst, 42, pe);
// old_val 是可靠的旧值，无需额外 quiet
```

### 8.3 MTE 路径 + UB 复用（`aclshmem_int32_atomic_add`，UB+MTE3 路径）

```c
aclshmemx_set_mte_config(0, 256, sync_id);

// fence MTE2 写 UB 完成（如果 MTE2 之前写入了同一 UB 区域）
SetFlag<HardEvent::MTE2_S>(sync_id);
WaitFlag<HardEvent::MTE2_S>(sync_id);

// 异步 atomic add（Scalar 写 UB → MTE3 读 UB → MTE3 写 GM）
aclshmem_int32_atomic_add(dst, 1, pe);

// fence MTE3 写 GM 完成——后续需要读 GM 结果
SetFlag<HardEvent::MTE3_MTE2>(sync_id);
WaitFlag<HardEvent::MTE3_MTE2>(sync_id);

// fence MTE3 读 UB 完成——如果要复用 UB
SetFlag<HardEvent::MTE3_S>(sync_id);
WaitFlag<HardEvent::MTE3_S>(sync_id);

// 现在可以安全读 GM 结果和复用 UB
```
````

### item 13

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/engine/shmem_device_mte.h
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/include/device/gm2gm/engine/shmem_device_mte.h); Copyright 2025, commit date unknown
Three ordering-relevant doc comments from the MTE engine header: aclshmem_ptr ('supported access method for the returned address depends on the transport and topology'), aclshmemx_mte_quiet ('Clear instruction pipes and flush data cache to GM') — i.e. the engine quiet IS the dcci on this path — and the four aclshmemx_mte_put_nbi overloads with their UB staging buffer and sync_id.
```cpp
/**
 * @brief Translate a local symmetric address to the corresponding symmetric address on the specified PE.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] Target PE number.
 * @return The corresponding symmetric address on the specified PE.
 * @note The supported access method for the returned address depends on the transport and topology.
 */
ACLSHMEM_DEVICE __gm__ void* aclshmem_ptr(__gm__ void* ptr, int pe);
#define shmem_ptr aclshmem_ptr

[... RMA declarations omitted ...]

ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf,
    const non_contiguous_copy_param& copy_params, int pe, uint32_t sync_id);
#define shmem_mte_put_mem_nbi aclshmemx_mte_put_nbi

/**
 * @brief Asynchronous interface. Clear instruction pipes and flush data cache to GM.
 */
ACLSHMEM_DEVICE void aclshmemx_mte_quiet();

/**
 * @brief Atomic fetch operation. Returns the value at the source address on the specified PE.
 *        Supported hardware platform: Ascend950.
 *        WARNING: Use sync_id in device_state.mte_config for pipeline synchronization.
 *        NOTE: This is an asynchronous interface. Atomic operations involve scalar computation (Scalar).
 *              If there are data dependencies between the scalar computation unit and the move unit (MTE2/MTE3)
 *              when reading/writing GM, developers need to insert synchronization according to actual situations.
 * @note T only supports int32_t/uint32_t/float/int64_t/uint64_t.
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param src               [in] Symmetric address of the source data.
 * @param pe                [in] PE number of the remote PE.
 * @return The value at the source address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch(__gm__ T* src, int32_t pe);


[... omitted ...]

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, const non_contiguous_copy_param& copy_params,
    int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);
```

### item 14

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_p2p_sync.h
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/include/device/gm2gm/shmem_device_p2p_sync.h); Copyright 2025, commit date unknown
The device-side signal doc comments. NOTE WHAT IS ABSENT: neither aclshmemx_signal_op nor aclshmem_signal_wait_until states that a delivered signal implies prior puts to the same PE are visible — the only atomicity claim is 'On Ascend950, RDMA is used as the communication engine, supporting signal_set with guaranteed atomicity'.
```cpp

/**
 * @brief The aclshmemx_signal_op operation updates sig_addr with signal using operation sig_op on the specified PE.
 *        This operation can be used together with aclshmem_signal_wait_until for efficient
 *        point-to-point synchronization.
 *        On Ascend950, RDMA is used as the communication engine, supporting signal_set with guaranteed atomicity.
 *
 * @param sig_addr              [in] Symmetric address of the signal word to be updated.
 * @param signal                [in] The value used to update sig_addr.
 * @param sig_op                [in] Operation used to update sig_addr with signal. Supported operations:
 *                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                    [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);
#define shmemx_signal_op aclshmemx_signal_op

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                                   ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @return Return the contents of the signal data object, sig_addr, at the calling PE that satisfies the wait condition.
 */
ACLSHMEM_DEVICE int32_t aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val);
#define shmem_signal_wait_until aclshmem_signal_wait_until
```

### item 15

https://github.com/immengzi/shmem/blob/master/include/device/gm2gm/shmem_device_cc.h
immengzi/shmem (GitHub mirror of Huawei cann/shmem) — file last committed 2026-01-30
The device-side barrier header's top-of-file WARNING block — the normative barrier visibility statement ('All operations of all pes of a team on excuting stream before the barrier are visiable to all pes of the team after the barrier', HCCS-only), the MIX-kernel/SyncAll restrictions, and item 4: 'The scalar unit of cube core is not affected by aclshmem_barrier_xxx.' Followed by the aclshmem_barrier / aclshmem_barrier_all declarations with their HCCS vs HCCS+RDMA caveat.
```cpp
/*
    WARNING：Restrictions of Barrier APIs.

    1. Barrier APIs can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if
       it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy).
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.

    2. Barrier APIs conflict with SyncAll. Avoid mixing them together.

    3. We provide 2 kinds of barrier:
        a. aclshmem_barrier_xxx
            Barrier of all cores. On systems with only HCCS: All operations of all pes of a team on excuting
            stream before the barrier are visiable to all pes of the team after the barrier.
        b. aclshmemx_barrier_xxx_vec
            Barrier of all VEC cores. On systems with only HCCS: All operations of ALL VEC CORES of all pes of a
            team on excuting stream before the barrier are visiable to ALL VEC CORES of all pes of the team after
            the barrier.

    4. The scalar unit of cube core is not affected by aclshmem_barrier_xxx. Make sure don't use that.
*/

[... includes omitted ...]


/**
 * @brief aclshmem_barrier is a collective synchronization routine over a team. Control returns from aclshmem_barrier
 *        after all PEs in the team have called aclshmem_barrier.
 *        aclshmem_barrier ensures that all previously issued stores and remote memory updates, including AMOs and
 *        RMA operations, done by any of the PEs in the active set are complete before returning. On systems with
 *        only scale-up network (HCCS), updates are globally visible, whereas on systems with both scale-up network
 *        HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the memory of a given PE are
 *        visible to that PE.
 *        Barrier operations issued on the CPU and the NPU only complete communication operations that were issued
 *        from the CPU and the NPU, respectively. To ensure completion of GPU-side operations from the CPU, using
 *        aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 * @param team              [in] team to do barrier
 */
ACLSHMEM_DEVICE void aclshmem_barrier(aclshmem_team_t team);
#define shmem_barrier aclshmem_barrier

/**
 * @brief aclshmem_barrier of all PEs.
 */
ACLSHMEM_DEVICE void aclshmem_barrier_all(void);
#define shmem_barrier_all aclshmem_barrier_all
```

### item 16

https://gitcode.com/cann/shmem/blob/master/include/host/mem/shmem_host_heap.h
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/include/host/mem/shmem_host_heap.h); Copyright 2025, commit date unknown
The symmetric-heap allocation entries: aclshmem_malloc / calloc / align / free and the aclshmemx_* variants with mem_type, plus aclshmemx_get_heap_base whose '@note Remote visibility: The returned address is local-only, not directly accessible by remote PEs.' is the only visibility statement on the allocation side.
```cpp
/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_HOST_HEAP_H
#define SHMEM_HOST_HEAP_H

#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory is not initialized.
 *        If <i>size</i> is 0, then <b>aclshmem_malloc()</b> returns NULL.
 *
 * @param size             [in] bytes to be allocated
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void* aclshmem_malloc(size_t size);
#define shmem_malloc aclshmem_malloc

/**
 * @brief allocate memory for an array of <i>nmemb</i> elements of <i>size</i> bytes each and returns a pointer to the
 *        allocated memory. The memory is set to zero. If <i>nmemb</i> or <i>size</i> is 0, then
 * <b>aclshmem_calloc()</b> returns either NULL.
 *
 * @param nmemb            [in] elements count
 * @param size             [in] each element size in bytes
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void* aclshmem_calloc(size_t nmemb, size_t size);
#define shmem_calloc aclshmem_calloc

/**
 * @brief allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory address will be a
 *        multiple of <i>alignment</i>, which must be a power of two.
 *
 * @param alignment        [in] memory address alignment
 * @param size             [in] bytes allocated
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void* aclshmem_align(size_t alignment, size_t size);
#define shmem_align aclshmem_align

/**
 * @brief Free the memory space pointed to by <i>ptr</i>, which must have been returned by a previous call to
 *       <b>aclshmem_malloc()</b>, <b>aclshmem_calloc()</b> or <b>aclshmem_align()</b>. If <i>ptr</i> is NULL,
 *       no operation is performed.
 * @param ptr              [in] point to memory block to be free.
 */
ACLSHMEM_HOST_API void aclshmem_free(void* ptr);
#define shmem_free aclshmem_free

/**
 * @brief Allocates a block of ACLSHMEM symmetric memory, the data in this memory is uninitialized
 *
 * @param size          [in] Memory allocation size (in bytes)
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the symmetric memory
 *
 */
ACLSHMEM_HOST_API void* aclshmemx_malloc(size_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Allocates a block of SHMEM symmetric memory and initializes all content to zero
 *
 * @param count         [in] Number of elements
 * @param size          [in] Number of bytes occupied by each element
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the symmetric memory
 */
ACLSHMEM_HOST_API void* aclshmemx_calloc(size_t count, size_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Allocates a block of SHMEM symmetric memory with alignment to the specified length
 *
 * @param alignment     [in] Alignment length (in bytes)
 * @param size          [in] Memory allocation size (in bytes)
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the symmetric memory
 */
ACLSHMEM_HOST_API void* aclshmemx_align(size_t alignment, size_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Frees the allocated symmetric memory
 *
 * @param ptr           [in] Pointer to the memory to be freed
 * @param mem_type      [in] Allocation location of the symmetric memory (Host/Device)
 */
ACLSHMEM_HOST_API void aclshmemx_free(void* ptr, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Returns the start address (heap_base) of the local symmetric memory heap.
 *        If <i>mem_type</i> is DEVICE_SIDE, returns the device heap base address.
 *        If <i>mem_type</i> is HOST_SIDE, returns the host heap base address.
 *        Must be called after ACLSHMEM initialization, otherwise returns NULL.
 *
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the start address of the symmetric memory heap, or NULL if not initialized.
 * @note Thread safety: This function is thread-safe for reading heap_base after initialization.
 *       The heap_base value is set during initialization and does not change afterwards.
 * @note Remote visibility: The returned address is local-only, not directly accessible by remote PEs.
 * @note Failure conditions: Returns NULL if ACLSHMEM is not initialized or the heap has not been allocated.
 */
ACLSHMEM_HOST_API void* aclshmemx_get_heap_base(aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Translate a pointer in a caller-provided source buffer to the corresponding local symmetric pointer.
 *
 * The query uses the current active ACLSHMEM instance initialized by aclshmemx_init_attr_with_buffers(). An interior
 * source pointer retains its byte offset within the matching source buffer. The result identifies the same byte in
 * the local symmetric heap and can be used as a local symmetric-address argument to SHMEM APIs.
 *
 * @param local_ptr [in] Pointer in one of the source intervals supplied during initialization.
 * @return The corresponding local symmetric pointer, or NULL if no active user-buffer-heap instance or interval
 *         matches local_ptr.
 * @note Execution domain: Host only. This query is non-collective and does not perform communication.
 * @note Thread safety: Concurrent read-only queries are supported after initialization. Do not race this call with
 *       finalization of the active instance.
 */
ACLSHMEM_HOST_API void* aclshmemx_get_buffer_ptr(const void* local_ptr);

#ifdef __cplusplus
}
#endif
#endif
```

### item 17

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_rma.h
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/include/device/gm2gm/shmem_device_rma.h); Copyright 2025, commit date unknown
The generic (non-engine-specific) device put_nbi doc comment: 'Supports MTE, RDMA, SDMA, or UDMA as the underlying transport', the dst-must-be-symmetric / src-may-be-any-local-GM address rule, and the @warning that concurrent RMA/AMO to the same PE is NOT supported under RDMA and that rdma_config/sdma_config sync_id is used for pipeline synchronization.
```cpp
/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(\_\_gm\_\_ TYPE *dst, \_\_gm\_\_ TYPE *src, uint32_t elem_size,
 * int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 * Supports MTE, RDMA, SDMA, or UDMA as the underlying transport.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local device of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 *
 * @note Address requirements: dst must point to symmetric memory because it is translated to the corresponding
 *       address on pe. src is used as supplied and may be any valid local device GM address.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting ACLSHMEM_DATA_OP_ROCE, complete transfer
 * ranges for both operands must remain within their respective symmetric memory allocations. Runtime dispatch may
 * select RDMA for the target PE; callers do not need to determine the engine selected for an individual operation.
 *
 * @warning Concurrent RMA/AMO operations to the same PE are NOT supported when using RDMA
 *          as the underlying transport. When using RDMA or SDMA, the corresponding
 *          sync_id from device_state's rdma_config or sdma_config is used for pipeline
 *          synchronization.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_NBI(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_NBI);
```

### item 18

https://gitcode.com/cann/shmem/blob/master/docs/coding_style_guide.md
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/docs/coding_style_guide.md); commit date unknown
The project's mandated doc-comment template for every asynchronous API — the normative phrasing of what a caller must do 'before reusing resources or assuming remote visibility', and §10.3's requirement that examples show the completion operation explicitly.
````
推荐模板：

```c
/**
 * @brief Submit a non-blocking put operation to a remote PE.
 *
 * @param[in] dst Destination symmetric address on the remote PE. Must not be NULL.
 * @param[in] src Source buffer on the local PE. Must not be NULL.
 * @param[in] size Number of bytes to transfer.
 * @param[in] pe Remote PE number. Must be in the valid PE range.
 *
 * @note Address requirements: dst points to symmetric memory on pe. src is
 *       used as supplied and may be any valid local device GM buffer.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting
 *       ACLSHMEM_DATA_OP_ROCE, complete transfer ranges for both operands must
 *       remain within their respective symmetric memory allocations. Runtime
 *       dispatch may select RDMA
 *       for the target PE; callers need not determine the per-operation engine.
 * @note Execution domain: Host.
 * @note Blocking semantics: non-blocking. Returning from this void API only
 *       means the submission path has returned; it is not a completion signal.
 * @note Completion semantics: the caller must call aclshmem_quiet(),
 *       aclshmem_barrier(), aclshmem_sync(), an engine-specific quiet API,
 *       or synchronize the associated stream as documented before reusing
 *       resources or assuming remote visibility.
 * @note Failure behavior: this existing API does not return a status code.
 *       Implementations must document diagnostics and error propagation
 *       through the established project mechanism.
 * @note Thread safety: thread-safe after successful initialization unless
 *       otherwise documented.
 */
ACLSHMEM_HOST_API void aclshmem_putmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe);
```

[... omitted ...]


```c
/**
 * @brief Put data from the current device execution context to a remote PE.
 *
 * @param[in] dst Destination symmetric address on the remote PE.
 * @param[in] src Source address in device global memory.
 * @param[in] size Number of bytes to transfer.
 * @param[in] pe Remote PE number.
 *
 * @note Address requirements: dst points to symmetric memory on pe. src is
 *       used as supplied and may be any valid address in device global memory.
 * @note RDMA-enabled configuration: If initialization enables RDMA by setting
 *       ACLSHMEM_DATA_OP_ROCE, complete transfer ranges for both operands must
 *       remain within their respective symmetric memory allocations. Runtime
 *       dispatch may select RDMA
 *       for the target PE; callers need not determine the per-operation engine.
 * @note Execution domain: Device.
 * @note The caller must ensure that dst and src are non-NULL, properly aligned,
 *       and that pe is valid. These conditions may be checked only in debug
 *       builds. Release device code does not perform defensive fallback checks.
 * @note Completion semantics: the caller must use the documented quiet or sync
 *       operation before assuming remote visibility.
 */
/* device-side public API declaration */
```

---

[... omitted ...]

### 10.2 完成性接口

异步接口必须说明需要哪个完成性接口，例如：

- `aclshmem_quiet`
- `aclshmem_fence`
- `aclshmem_barrier`
- `aclshmem_sync`
- stream 同步接口
- device 同步接口
- engine-specific quiet，例如 MTE、SDMA、RDMA、UDMA 对应的 quiet 或 wait 接口

### 10.3 示例代码要求

示例代码中，如果使用异步接口，必须显式展示完成性操作，例如：

```c
aclshmem_putmem_nbi(dst, src, size, pe);

/* Required before reusing src or assuming remote visibility. */
aclshmem_quiet();
```
````

### item 19

https://gitcode.com/cann/shmem/blob/master/docs/example/example.md
Huawei / cann/shmem (ACLSHMEM), master branch (raw: https://raw.gitcode.com/cann/shmem/raw/master/docs/example/example.md); commit date unknown
The official AllGather kernel example showing the exact ordering the library expects: mte_put_nbi -> aclshmem_quiet() -> aclshmemx_signal_op(..., my_rank) -> aclshmem_signal_wait_until(aclshmem_ptr(gva_sync_gm, x) + flag_offset, ...) — quiet BEFORE the signal, and the wait polls through aclshmem_ptr into the peer's symmetric address.
```cpp

    // data move parameters
    const uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t input_offset, output_offset, gva_offset, num_per_core;

    // [AllGather Step 1] local input gm -> symmetric mem.
    num_per_core = elements / aivNum;
    input_offset = aivIndex * num_per_core;
    gva_offset = aivIndex * num_per_core;
    if (aivIndex == aivNum - 1) {
        num_per_core = elements - num_per_core * aivIndex;
    }
    aclshmemx_mte_put_nbi(gva_data_gm + gva_offset, input_gm + input_offset, tmp_buff, ub_size, num_per_core, my_rank, EVENT_ID0);

    const int64_t core_per_rank = aivNum / pe_size;
    const int64_t core_rank_idx = aivIndex % core_per_rank;
    const int64_t x = aivIndex / core_per_rank;

    // Sync Ensure Corresponding Tasks Done.
    aclshmem_quiet();

    aclshmemx_signal_op(gva_sync_gm + flag_offset, magic, ACLSHMEM_SIGNAL_SET, my_rank);
    aclshmem_signal_wait_until((__gm__ int32_t *)aclshmem_ptr(gva_sync_gm, x) + flag_offset, ACLSHMEM_CMP_EQ, magic);

    // [AllGather Step 2] symmetric mem -> local output.
    num_per_core = elements / core_per_rank;
    output_offset = x * elements + core_rank_idx * num_per_core;
    gva_offset = core_rank_idx * num_per_core;
    if (core_rank_idx == core_per_rank - 1) {
        num_per_core = elements - num_per_core * core_rank_idx;
    }
    aclshmemx_mte_get_nbi(output_gm + output_offset, gva_data_gm + gva_offset, tmp_buff, ub_size, num_per_core, x, EVENT_ID0);
}
```

### item 20

https://github.com/mindspore-ai/hyper-parallel/blob/master/hyper_parallel/core/multicore/ops/runtime/worker_kernel.h
mindspore-ai/hyper-parallel — file last committed 2026-10-15
A real megakernel invalidate-then-poll implementation. WaitForDependency re-issues DataCacheCleanAndInvalid<int32_t, SINGLE_CACHE_LINE, CACHELINE_OUT> before EVERY re-read of the GM event counter inside the spin loop (rate-limited by GetSystemCycle), and PublishDeviceReady does dcci -> GetValue -> SetValue -> dcci -> PipeBarrier<PIPE_ALL>() before aclshmemx_signal_op / aclshmem_signal_wait_until; note DATA_CACHE_LINE_SIZE/INT32_T_SIZE is used as the per-slot stride.
```cpp
  __aicore__ inline void PublishDeviceReady(const ReadyHandshakeMeta &meta, bool completion = false) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t rank = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready = (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + runtime_event_capacity;
    if (completion) {
      ready += (ep + 1) * ready_stride;
    }

    GlobalTensor<int32_t> ready_state;
    ready_state.SetGlobalBuffer(ready, static_cast<uint32_t>((ep + 1) * ready_stride));
    uint32_t generation_index = static_cast<uint32_t>(ep) * ready_stride;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    int32_t previous_generation = ready_state.GetValue(generation_index);
    if (previous_generation == INT32_MAX) {
      AscendC::Trap();
    }
    int32_t generation = previous_generation + 1;
    ready_state.SetValue(generation_index, generation);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    PipeBarrier<PIPE_ALL>();

    uint32_t round = 0;
    for (int64_t distance = 1; distance < ep; distance *= READY_SIGNAL_RADIX, ++round) {
      int64_t target = (rank + distance) % ep;
      __gm__ int32_t *round_ready = ready + round * ready_stride;
      aclshmemx_signal_op(round_ready, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(target));
      aclshmem_signal_wait_until(round_ready, ACLSHMEM_CMP_GE, generation);
    }
    if (!completion) {
      TriggerEvent(meta.ready_event);
    }
#endif
  }

[... omitted ...]

  __aicore__ inline void WaitForDependency(uint32_t event_index, uint32_t poll_interval_us = 0) {
    // Pull producers signal locally after MTE3 completion. Observe short-lived
    // GMM/SwiGLU dependencies promptly while retaining a bounded refresh rate.
#ifdef __DAV_C220_CUBE__
    const uint32_t default_poll_interval_us = pull_protocol_ ? 10 : CUBE_EVENT_REFRESH_INTERVAL_UNITS;
#else
    const uint32_t default_poll_interval_us = pull_protocol_ ? 30 : VECTOR_EVENT_REFRESH_INTERVAL_UNITS;
#endif
    if (poll_interval_us == 0) {
      poll_interval_us = default_poll_interval_us;
    }
    constexpr int64_t cycles_per_us = EVENT_REFRESH_TIME_UNIT_CYCLES;
    int32_t needed = all_event_num_triggers.GetValue(event_index);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t previous_cycle = AscendC::GetSystemCycle();
    while (current < needed) {
      int64_t elapsed_cycles = AscendC::GetSystemCycle() - previous_cycle;
      if (elapsed_cycles / cycles_per_us > poll_interval_us) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        previous_cycle = AscendC::GetSystemCycle();
      }
    }
  }
```

### item 21

https://github.com/mindspore-ai/hyper-parallel/blob/master/hyper_parallel/core/multicore/docs/shmem.md
mindspore-ai/hyper-parallel — file last committed 2026-10-15
A detailed Chinese analysis of the CANN shmem data plane's protocol constraints: Push vs Pull one-sided modes, 'Signal热点的64B隔离' (concurrently-updated independent signals on one target PE must not share a 64B cacheline; give each signal an alignment=64 allocation or 64B stride), and '跨 rank 初始化收敛' (a late local zero_()/fill_() will overwrite an already-delivered remote write), plus the warning that the device barrier spins without timeout and trips aicore 507015 under interconnect contention.
```
## 单边通信模式

单边接口组合成两种数据面模式，`signal`/`wait_signal` 提供跨 PE 的完成通知：

- **Push（推）**：发送方 `put` 把数据直接写进接收方的对称槽，再 `signal` 通知；接收方
  `wait_signal` 后读自己的本地对称槽。数据落点在接收方，适合接收方持有消费缓冲的场景。
- **Pull（拉）**：发送方把数据写进自己的对称槽（本地写或收到的 put），`signal` 通知接收方；
  接收方 `wait_signal` 后 `get` 从发送方的对称槽拉取。数据落点在发送方，适合发送方
  复用同一缓冲多轮发布的场景。

两种模式都必须遵守两条协议约束，均来自 CANN 数据面的既定行为：

1. **Signal热点的64B隔离**：当多个来源PE或设备任务会并发更新同一目标PE上的多个独立Signal位置时，
   这些位置不能共享64B cacheline。热点消费者应让每个Signal独占一个`alignment=64`的分配，或在同一
   分配内按64B步长排布。没有并发写入邻居的单个Signal不因通用接口本身而要求64B基址对齐。
2. **跨 rank 初始化收敛**：任何远端可见写（`signal`、`put` 到对端槽位）之前，所有 rank
   必须完成对相应槽位的本地初始化（`zero_()`/`fill_()`）并经 `host_barrier()` 收敛；否则一个
   迟到的本地初始化会覆盖已送达的远端写，表现为信号永远等不到或数据读回全零。
   设备 barrier（`shmem.barrier()`）不得用作跨 rank 收敛点：其远端等待为无超时自旋，
   互连竞争下会触发 aicore 507015（见 bug_issue：
   `shmem_device_barrier_unbounded_spin_interconnect_contention.md`）。
```

### item 22

https://github.com/huawei-csl/pto-kernels/blob/master/examples/aicpu/hello-world/README.md
huawei-csl/pto-kernels — file last committed 2026-08-24
Reproduces the official CANN example (from hiascend atlasascendc_api_07_00188.html) of an AI Core spinning on an AI-CPU-written GM lock with dcci inside the loop, and independently reports that AscendC::DataStoreBarrier() is only `dsb st` on the device.
````
# AICPU Hello world example

Copied from

### item 23

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00050.html

## `--cce-aicpu-no-firmware` is required

The compile commands in the documentation linked above omit this flag, and without
it the kernel never executes. Measured on Ascend 910B2, CANN 9.0.0, driver
26.0.rc1, 10 runs per build:

| build | runs that print |
|---|---|
| exactly as documented | 0/10 |
| as documented, plus `-fPIC` | 0/10 |
| plus `--cce-aicpu-no-firmware` | 9/10 |

`-fPIC` and `-g` make no difference; only the flag does.

When the flag is missing, `aclrtSynchronizeStream` returns `507018`
(`aicpu exception`) and the device reports `errCode=11003` ("get kernel failed") in
`~/ascend/atrace/*/schedule_event_*/schedule_tracer_*.txt`. A kernel with an empty
body fails identically, so this is unrelated to `printf`.

**Why the flag is needed is not established.** Both builds embed an `ET_DYN`
AArch64 shared object with identical symbols and no `SONAME`, differing only in
size, so the compiler's own help text is the only available explanation:

```
--cce-aicpu-no-firmware
      Compile AICpu code with operating system support instead
      of for firmware-only device
```

The flag is undocumented, and the documentation section that would cover AI CPU
compilation -- 2.3.3 "AI CPU算子编译", referenced by 2.2.5 -- is absent from the
CANN 9.0.0 documentation set. Maintainer input welcome.

### Note on printf reliability

With the flag applied, roughly 6% of runs still lose the output (47/50 printed over
50 runs). `AscendC::printf` writes into a dump buffer drained by a host-side thread,
and `DataStoreBarrier()` is only `dsb st` on the device, so teardown can race the
drain. Inserting a wait before `aclrtDestroyStream` makes it 50/50, but the
documentation prescribes no such wait, so this looks like a CANN-side issue rather
than a missing step here.




### Second example

Interesting example from https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_00188.html



```
struct TilingInfo {
    uint64_t lock; // AI CPU/AI Core之间同步的锁
    int8_t type;
    int8_t mode;
    int8_t len;
};
struct KernelArgs {
    uint32_t *xDevice;
    uint32_t *yDevice;
    uint32_t *zDevice;
    TilingInfo *ti; // 与AI Core共享的参数，用于同步tiling选择
};

template<typename T, int8_t mode, int8_t len>
__aicore__ void hello_world_impl(GM_ADDR m)
{
    if constexpr (std::is_same_v<T, float>) {
       AscendC::printf("Hello World: float mode %u len %u.\n", mode, len);
    } else if constexpr (std::is_same_v<T, int>) {
       AscendC::printf("Hello World: int mode %u len %u.\n", mode, len);
    }
}

// AI Core算子总入口
// tilingInfo: 和AI CPU算子共同传递的参数，用于数据共享
template<typename T, int8_t mode, int8_t len>
__mix__(1,2) __global__ __aicore__ void hello_world(GM_ADDR m, GM_ADDR TilingPtr)
{
     __gm__ struct KernelInfo::TilingInfo *ti = (__gm__ struct KernelInfo::TilingInfo *)TilingPtr;
    AscendC::GlobalTensor<uint64_t> lock;
    lock.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(&ti->lock));
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            while (*reinterpret_cast<volatile __gm__ uint64_t*>(lock.GetPhyAddr(0)) == 0) {   // 下沉模式，AI Core等待AICPU tiling计算完成
                AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(lock);    //直接访问Global Memory，获取最新数据
            }
        }
    }
    // 上面是1个核等待AI CPU tiling计算完成，这里进行核间同步
    AscendC::SyncAll<false>();
    // 根据tiling参数值选择不同模板
    if (ti->type ==0 && ti->mode == 1 && ti->len == 2) {
        hello_world_impl<float, 1, 2>(m);
    } else if (ti->type == 1 && ti->mode == 2 && ti->len == 4) {
        hello_world_impl<int, 2, 4>(m);
    }
    // 执行完留一个核释放lock
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            lock.SetValue(0, 0);  // 刷新 lock
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(lock);    //刷新Dcache，同步与GM之间的数据
        }
    }
}

extern "C" __global__ __aicpu__ uint32_t MyAicpuKernel(void *arg)
{
    KernelArgs* cfg = (KernelArgs*)arg;
    AscendC::printf("MyAicpuKernel inited!\n");
    cfg->ti->lock = 1;
    cfg->ti->type = 1;
    cfg->ti->mode = 2;
    cfg->ti->len = 4;
    AscendC::DataStoreBarrier(); // 对tilingInfo进行写同步
    AscendC::printf("MyAicpuKernel inited type %u mode %u len %u end!\n", cfg->ti->type, cfg->ti->mode, cfg->ti->len);
    return 0;
}
```
````

### item 24

https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops/op_kernel/moe_distribute_base.h
sgl-project/sgl-kernel-npu — file last committed 2026-08-28
Production helper `cacheWriteThrough(__gm__ uint8_t*, uint64_t)` that walks an arbitrary GM range at AscendC::CACHE_LINE_SIZE granularity issuing one SINGLE_CACHE_LINE / CACHELINE_OUT dcci per line — the idiom for flushing/invalidating a multi-cacheline remotely-written buffer.
```cpp
__aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, uint64_t length)
{
    __gm__ uint8_t *start =
        (__gm__ uint8_t *)((uint64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    if (length == 0) {
        return;
    }
    __gm__ uint8_t *end =
        (__gm__ uint8_t *)(((uint64_t)sourceAddr + length - 1) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(global[i]);
    }
}
```

### item 25

https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops/op_kernel/moe_distribute_v2_base.h
sgl-project/sgl-kernel-npu — commit date unknown (same tree as above, fetched 2026-10-20)
The DATA_FULSH macro: the full belt-and-braces flush sequence actually used in shipped code — AscendC::Barrier(), then ENTIRE_DATA_CACHE / CACHELINE_OUT dcci, then a NOP, then dsb(DSB_ALL).
```cpp
#define DATA_FULSH(_gm_tensor, _type)                                                                  \
    AscendC::Barrier();                                                                                \
    DataCacheCleanAndInvalid<_type, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(_gm_tensor); \
    __asm__("NOP");                                                                                    \
    dsb(DSB_ALL);
```

### item 26

https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops/op_kernel/notify_dispatch.h
sgl-project/sgl-kernel-npu — file last committed 2026-08-28
GetMagicValue: dcci the per-block status slot, GetValue it, then write back the incremented magic — the read-modify-write-on-a-remotely-visible-flag pattern, with UB_ALIGN_SIZE used as the per-block stride to keep blocks off each other's cache line.
```cpp
// Assign values to winContext_[COMM_EP_IDX] and blockIdx before calling
template <typename T>
__aicore__ inline uint64_t NotifyDispatch<T>::GetMagicValue(void)
{
    uint64_t magic = 0;
    GlobalTensor<uint64_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_[COMM_EP_IDX]->localWindowsExp);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ uint64_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[blockIdx * UB_ALIGN_SIZE]);
    magic = selfDataStatusTensor(blockIdx * UB_ALIGN_SIZE);
    if (magic <= 0) {
        magic = 1;
    }
    selfDataStatusTensor(blockIdx * UB_ALIGN_SIZE) = magic + 1;
    return magic;
}
```

### item 27

https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops/op_kernel/cam_moe_dispatch_normal.h
sgl-project/sgl-kernel-npu — commit date unknown (fetched 2026-10-20)
The double-buffer state flip at kernel entry: dcci the self data-status flag, read it, flip it, dcci again, then PipeBarrier<PIPE_ALL>() — the dcci/dcci/PipeBarrier sandwich around a scalar RMW on a GM flag that peers also write.
```cpp
    stateOffset = STATE_OFFSET;
    srcRankOffset = startStatusId / moeExpertNumPerRank;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState = selfDataStatusTensor(0);
    if (dataState == 0) {
        selfDataStatusTensor(0) = 1;
    } else {
        selfDataStatusTensor(0) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    PipeBarrier<PIPE_ALL>();
```

### TOPIC A visibility sweep — negative findings

NOTHING: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/... — CANN 9.2.0 is not published on the hiascend doc portal under any probed version code. /doc_center/source/ returns HTTP 403 for 920, 92RC1, 920alpha001, 92RC1alpha001, 910, 901, 900beta3, `9.2.0` and `latest`; only 900 (CANN 9.0.0), 900beta1, 900beta2, 850 and 82RC1 resolve, for both CANNCommunityEdition and canncommercial. The 900 pages quoted above are the newest published text and already list "Atlas 350 加速卡" alongside A3, so they cover the dav-2201 A3 target.

NOTHING: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/<book>/ for any aclshmem/ACLSHMEM book — there is NO aclshmem API-reference book on the hiascend doc portal. Probed book names shmemapi, aclshmemapi, shmemug, shmemdevg, shmem, aclshmem, symmemapi all return HTTP 403 under /doc_center/source/zh/CANNCommunityEdition/900/API/. Repeated WebSearch ("aclshmem API 参考", "aclshmemx_signal_op 接口", "昇腾 对称内存 shmem 接口 参考", site:hiascend.com) returns only gitcode.com/cann/shmem source mirrors. The authoritative ACLSHMEM documentation is the in-repo headers and docs/ quoted above; hiascend.com carries only the HCCL-side symmetric-memory APIs (HcclCommSymWinRegister etc.).

NOTHING: aclshmemx_quiet / aclshmemx_fence — no such symbols exist. The memory-ordering header declares only `aclshmem_quiet` and `aclshmem_fence` (no `aclshmemx_` spelling). The `aclshmemx_` prefix is used for the per-engine variants: aclshmemx_mte_quiet, aclshmemx_sdma_quiet, aclshmemx_udma_quiet, aclshmemx_roce_quiet.

NOTHING: aclshmemx_signal_wait_until — no device-side symbol with this name. The device header declares `aclshmem_signal_wait_until` (int32_t return) and the host header declares `aclshmem_signal_wait_until` plus `aclshmemx_signal_wait_until_on_stream`. Only the `_on_stream` host variant carries the `x`.

NOTHING: AscendC::DcciDst::CACHE_LINE_SINGLE — the enumerator does not exist. The CacheLine enum values are `SINGLE_CACHE_LINE` and `ENTIRE_DATA_CACHE`; the DcciDst enum values are `CACHELINE_ALL`, `CACHELINE_UB`, `CACHELINE_OUT`, `CACHELINE_ATOMIC`.

NOTHING: zhihu.com / csdn.net / cnblogs.com / bbs.huaweicloud.com / juejin.cn — four separate WebSearches ("aclshmem 单边 put 可见性", "昇腾 远端写 cache 一致性 dcci", "DataCacheCleanAndInvalid 用法", "昇腾 GM 标志位 轮询 cache", plus zhihu- and CSDN-targeted phrasings) surfaced no third-party Chinese blog post on this topic. Every result resolved back to hiascend.com official docs, asc.gitcode.com, or gitcode.com/cann/*. The Chinese-language technical writing on remote-write visibility and dcci that does exist is vendor-authored and is captured above (asc.gitcode.com 缓存一致性 / 内存一致性, cann/shmem docs/api/*.md) plus the third-party mindspore-ai/hyper-parallel docs/shmem.md.

NOTHING: openeuler-mirror/umdk src/cam/ — not searched for dcci/visibility material; the UMDK/CAM tree is a host-side URMA transport library and `gh search code "DataCacheCleanAndInvalid"` / `"DcciDst::CACHELINE_OUT"` returned zero hits in it, so it carries no AI-core-side cache-maintenance code.

NOTHING: tile-ai/tilelang-ascend, Ascend/AscendNPU-IR, ByteDance-Seed/Triton-distributed (ascend/) — these appear in `gh search code "aclshmem_quiet"` only as thin FFI/MLIR wrappers (e.g. AscendNPU-IR bishengir/lib/Template/lib/Shmem/Shmem.cpp `_mlir_ciface_aclshmem_quiet` calling `aclshmem_quiet()`, Triton-distributed ascend/python/triton_dist/language/extra/ascend/libaclshmem_device.py mapping `(): ("aclshmem_quiet", ())`). They restate no ordering or visibility guarantee, so nothing verbatim is worth quoting.



## Part 8. ★★ The vendor's own MC2 tuning case — 32.7% measured on MatmulAllReduce, and the three reasons none of it is reachable at batch=1

Source: `/home/shanshan/repo/cann/asc-devkit/docs/zh/guide/operator_practice/best_practices/mc_operator_tuning.md`,
「MC²算子性能调优案例」, in the shipped devkit on this machine. Read 2026-10-20. The sample operators it points
at are `https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce`,
`.../mc2/matmul_reduce_scatter_v2` and `.../mc2/all_gather_matmul_v2`.

This is the complete, measured methodology for the fused matmul+allreduce we keep considering for the
5.35 ms, 100%-serial TP reduce. It is worth reading in full precisely because it rules the approach out at
batch=1 on three independent numeric grounds, each stated by the vendor rather than inferred.

### 8.1 Where the gain comes from, and the exact benefit formula

案例介绍, verbatim:

> MC2通算融合算子的性能收益主要来自于通信、计算的并行执行，即将输入数据切分为多个子块，子块的计算和通信任务形成两条流水线，通过两条流水线上任务的并行执行，实现流水掩盖，从而提升算子性能。…输入矩阵沿M轴被切分为两块，第二块数据的Matmul计算和第一块数据的通信可以并行执行，从而达到计算和通信时间相互掩盖的目的。

分析主要瓶颈点, verbatim:

> MC2算子性能收益公式为：
> 融合前算子串行耗时 = 融合前计算算子耗时 + 融合前通信算子耗时
> MC2算子收益 = （融合前算子串行耗时 - 融合后MC2算子耗时） / 融合前算子串行耗时
>
> - 因素一：计算和通信的执行时间差异
>   若计算和通信任务的执行时间相差不大，则融合后，MC2算子的计算和通信并行执行，实现流水掩盖效果，性能收益较大。若计算和通信任务的执行时间差异较大，则融合后…能够掩盖的时间较少，算子整体执行耗时与未切分串行时的算子执行耗时接近，此时无法获得较大性能收益。
> - 因素二：数据切分导致的计算或通信的执行时间膨胀
>   当对输入数据进行切分后，原本的整块数据被切分成若干小数据块…此时相比切分前，计算或者通信任务的执行时间可能发生膨胀（即执行时间变长）。该膨胀产生的原因包括：**切分后的数据块过小导致计算或通信的效率降低、切分的数据块过多导致增加额外的调度开销、并行执行后计算和通信对L2 Cache或device存储内存的访问冲突等。**

and the three worked inflation regimes, verbatim — note that the third is a *regression*:

> - 未发生膨胀：数据切分前，Matmul执行时间为200us，将Matmul的输入均匀切分为两块，假设切分后，每块数据的Matmul执行时间都是100us，通过计算的并行执行，下图实际性能收益为100us。
> - 发生一般程度的膨胀：…每块数据的Matmul执行时间都是150us…实际性能收益为50us。
> - 发生严重程度的膨胀：…每块数据的Matmul执行时间都是200us…**实际性能收益为劣化50us。**

### 8.2 ★★★ Only M may be split — HCCL requires contiguity

设计优化方案, verbatim. This is the first and hardest of the three blockers:

> 该算子的数据切分策略应满足如下要求：
> - **只切分M轴。因为通信任务调用的Hccl API要求分块数据内存连续，若按N轴切分，则每行数据都被切断，导致通信数据的内存不连续，不满足通信要求；若按M轴切分，则每行数据都是内存连续的，满足通信要求。**
> - 若A表示长块、B表示短块，只能切出A或B连续排布的形式，例如AAAB、BAAA等情况。

and the two bound regimes, verbatim:

> - 计算bound：对于同一个切分后的数据块，计算执行耗时大于通信执行耗时，此时，**计算连续**，且**通信的尾块要短**。
> - 通信bound：对于同一个切分后的数据块，通信执行耗时大于计算执行耗时，此时，**通信连续**，且**计算的头块要短**。

**At decode batch=1, M is 1 — or 8 rows under the MSD stack. There is no M axis to split.** Every bit of
MC2's mechanism is M-axis pipelining, and N is explicitly unavailable because HCCL needs each block
contiguous.

### 8.3 ★★ The empirical short-block floor is 384 rows

切分算法步骤, step 1, verbatim — the third inequality is the one that matters:

> 1. 根据输入矩阵的shape：M、K、N，按照经验值设置合适的M轴方向切分的短块长度。如下表达式中，a、b、c表示根据经验值给出的短块长度的备选，将K、N带入如下三个表达式后，取a、b、c的最小值m0为选定的短块长度。
>    - `a * K * N >= 4 * 1024 * 1024 * 1024`，a取不等式的最小值
>    - `b * K * N / 1024 + b * N >= 6 * 1024 * 1024`，b取不等式的最小值
>    - **`c >= 3 * 128`，c取不等式的最小值**
>    - `m0 = min(a, b, c)`

`c >= 3 * 128` is 384. So **the vendor's own heuristic never produces a short block shorter than 384 rows**,
and `m0 = min(a, b, c)` makes 384 the ceiling on the floor rather than a soft preference. A decode M of 1 or 8
is two orders of magnitude below it.

Steps 2-4, verbatim, including the memory-contention fudge factor:

> 2. 根据短块长度m0和拟合公式，分别得到计算执行时间t0 = CostMM(m0) * 1.15，通信执行时间t1 = CostComm(m0) * 1.15。
>    注意：**通信和计算并行执行时，可能出现抢占内存带宽的情况，导致执行时间增加，一般按经验在拟合公式中乘以1.15的系数**，用户可以根据实测情况调整该系数。
> 3. 根据短块的长度…配平通算，得到长块的长度，**长块长度尽量对齐128个元素，以保证计算亲和**。…将短块的通信时间和长块的计算时间匹配相等：将t1作为长块的计算执行时间，带入t1 = CostMM(m1)公式，计算得到m1，即为长块的长度。
> 4. 根据短块长度m0、长块长度m1、原始M轴长度M，得出长块的切块个数count = (M - m0) / m1。该式一般不能整除，此时需要做如下处理：
>    - 将结果的小数部分舍弃，保留整数部分作为切块个数。
>    - 由于舍弃了小数部分，M轴长度有剩余，因此需要调整长块长度m1 = (M - m0) / count。
>    - 为保持计算亲和，将长块长度m1调整至128对齐，即向下取128倍数的整数，更新长块长度m1。
>    - 由于调整m1后，M轴长度有剩余，因此调整短块长度m0 = M - (m1 * count)。
>    - 最终得到短块长度m0，长块长度m1，长块个数count。

### 8.4 ★ The fitted AllReduce cost model, with a ~14.8 us fixed floor

步骤 3, verbatim — the calibration recipe and the resulting formula:

> 3. 将步骤2得到的数据块分别做AllReduce通信，利用msOpProf工具采集执行时间，得到每个数据块的执行时间t1, t2, ..., tn。然后，作图分析数据量与对应的执行时间的关系，拟合得到公式 t = CostComm(m)…该映射关系一般为线性，**若不满足线性关系，可以采用分段拟合的方式。**
>
>    数据量 x = m * N * sizeof(dataType)，单位是Bytes。该拟合公式表示为：
>    - **x小于8MB：t = -0.9698202 * x * x + 27.0622573 * x + 14.769，单位是us。**
>    - **x大于等于8MB：t = 13.58491263 * x + 61.508333，单位是us。**

(The coefficients only make dimensional sense with `x` in MB despite the 单位是Bytes note; the worked example
in §8.5 confirms it, substituting x = 6 MB.) The piece to keep is the **14.769 us constant term**: an AllReduce
on this platform has a fixed cost of roughly 15 us that no amount of message-size reduction removes. Step 2
gives the calibration grid, verbatim: 「将输入数据按M轴切分，分别切成M轴为256、512、768、1024、2048的若干数据块」.

### 8.5 The worked case: 1874 → 1262 us, 32.7%

验证优化方案性能收益, verbatim:

> 本MatmulAllReduce案例中，给定的输入矩阵Shape为M=4096，K=3072，N=8192，数据类型为half，分核数为8，通过融合前msOpProf工具采集，得到该输入的**Matmul计算执行时间为803us，AllReduce通信的执行时间为1071us，总耗时1874us，属于通信bound场景**。按照上述的切分算法，本案例的具体切分情况如下：
>
> 1. 根据经验值选定短块（bound场景中短块为头块，即切分后的第一个数据块）M方向长度m0为384，则通信数据量x为：384 * 8192 * 2 / 1024 / 1024 = 6MB。按通信拟合公式估算通信的执行时间为143us。考虑可能会发生内存带宽冲突，因此再乘以1.15的系数，得出通信执行时间164us。
> 2. 计算长块M方向长度。根据短块通信时间，配平长块的计算执行时间同样为164us，按计算拟合公式估算出该长度m1为768。
> 3. 根据M=4096、m0=384、m1=768，计算长块个数：(4096 - 384) / 768 = 4.83，向下取整为4。
> 4. 根据短块长度m0=384，长块个数4，调整长块m1长度：(4096 - 384) / 4 = 928，向下按128对齐，调整m1为896。
> 5. 根据长块长度m1=896，长块个数4，调整短块m0长度：4096 - 896 * 4 = 512。
> 6. 最终得到将原始输入矩阵切分为5个数据块，长度分别为：**{512，896，896，896，896}**。
>
> …按该切分策略测试，**融合后该算子的执行时间为1262us，则融合算子的性能收益为(1874 - 1262) / 1874 = 32.7%**。

the tiling that expresses it, verbatim:

```cpp
MatmulAllReduceCustomTilingData *tiling = context->GetTilingData();
tiling->param.rankDim = 8;
tiling->param.tileM = 512; // 短块大小
tiling->param.tileNum = 1; // 短块个数
tiling->param.tailM = 896; // 长块大小
tiling->param.tailNum = 4; // 长块个数
tiling->param.rankM = 4096;
tiling->param.rankN = 8192;
tiling->param.rankK = 4096;
tiling->param.isTransposeA = 0;
tiling->param.isTransposeB = 0;
tiling->param.cToFloatLen = 0;
tiling->param.nd2NzWorkLen = true;
tiling->param.dataType = static_cast(HCCL_DATA_TYPE_MAP.at(aType));
```

and the closing caution, verbatim:

> 上文提到，切分数据会引起计算或通信执行时间的膨胀，使实测结果与理论值有偏差。**比如切块数量较多时，执行时间的膨胀对性能影响较大，可能导致性能收益变小或者出现性能劣化**，因此最终需要根据上述理论切分策略，结合实测，对切分策略做调整。
>
> 总结：MC2算子通过数据切分后计算和通信的并行执行，获得性能收益，但受数据切分后执行时间膨胀的影响。

## Relevance

This closes the MC2 question for our regime with the vendor's own numbers rather than by argument, and it does
so three times over. **One: only the M axis may be split, because 「通信任务调用的Hccl API要求分块数据内存连续」
and an N split cuts every row.** At decode batch=1, M is 1 — 8 rows under the MSD stack — so there is no axis to
pipeline on. **Two: the vendor's own short-block heuristic has a hard floor of `c >= 3 * 128` = 384 rows, taken
as `m0 = min(a, b, c)`,** which puts the smallest block the method contemplates two orders of magnitude above
our M. **Three: the entire gain is 流水掩盖 between a compute pipeline and a comm pipeline, and the worked case
is comm-bound with Matmul 803 us against AllReduce 1071 us** — a ratio achievable only because K=3072 and
M=4096 give the Matmul real work to hide behind. At M=1 the matmul is trivial and there is nothing to overlap
with. So the earlier MonoMoE finding that fusing the TP all-reduce does not pay (README finding 4) now has a
mechanism and a numeric threshold attached, from Huawei's own tuning guide.

Two numbers from it are useful regardless of whether we ever use MC2. The fitted AllReduce cost has a
**14.769 us constant term** below 8 MB — a fixed per-call floor that no message-size reduction removes, which
bounds from below any design that issues one all-reduce per layer: at 78 layers that intercept alone is about
1.15 ms, and our measured 5.35 ms TP reduce should be read against it. And the **1.15 inflation coefficient**
is the vendor's own empirical allowance for compute and communication contending for memory bandwidth when run
in parallel — 「通信和计算并行执行时，可能出现抢占内存带宽的情况」 — which is the same contention MegaMoE's
authors measured as ~50% compute inflation and which §50.4's 512-byte serialization rule explains the mechanism
of. Any in-kernel one-sided reduce we write will pay some version of it, and 15% is the vendor's starting
estimate for a well-behaved case.


## Gaps

Looked for and could not find:

- **`aclshmem` measured latency numbers.** The `cann/shmem` repo, its full host/device API surface, the v1.7.0 release notes and the install path are all public and quoted above, but no published microbenchmark gives an `aclshmem_*_put_nbi` or `aclshmem_barrier` latency on A3. The repo ships `multi-QP performance tests` and `UDMA AllReduce examples` per the v1.7.0 notes; those scripts exist in-tree but no results are published. The SHMEM documentation site (https://shmem-doc.pages.dev/) was not reachable for extraction.
- **`LCCL` / `LCOC`.** Searched in English and Chinese, including against the CANN GitCode org listing, the HCCL repo tree, and the ATB/MindIE documentation. No public repository, API reference or measurement for an Ascend "low-latency collective communication library" under that name was found. The nearest real artefacts are `HCCL_OP_EXPANSION_MODE=AIV` (quoted above) and `cann/asc-comm`. The one hit referencing an "lccl backend" was an ATB API page (`ascendtb_01_0051`) whose content could not be extracted. Do not treat LCCL as a usable target without an internal source.
- **Any measured A3 TP16 small-message AllReduce latency.** HCCL's selector constants give the *thresholds* Huawei tuned to (128 KB AIV one-shot/two-shot crossover for ≤8 ranks; `OMNI_UBX_AR_RANK_SIZE = 16`), and the MC2 tuning case gives an 8-card A2 fit with a ~14.769 us zero-size intercept, but no published number exists for a 16-die A3 AllReduce at a few KB.
- **`HCCL_BUFFSIZE` default value and its exact relation to the AIV CCL buffer.** The `aiv_defines.h` layout map (16K/512K/1M/5M/9M/10M/18M/34M offsets) and `HcclGetHcclBuffer` are quoted, and vllm-ascend's GLM-5.2 reference sets `HCCL_BUFFSIZE: "400"`, but the documented default and the unit's precise semantics for the AIV path were not found in an extractable page (hiascend.com renders via JavaScript; the `doc_center/source` mirror worked for `hcclug_000134` and the MC2 case but not for the env-var reference pages).
- **`HCCL_INTRA_ROCE_ENABLE` semantics beyond one sentence.** Only vllm-ascend's one-line description ("Uses RDMA instead of SDMA links between two 8P nodes for mesh interconnect") and the HCCL source note that it is not parsed on A5-and-later (`if (!EqualToOrGreaterThan950(deviceType))`) were retrievable.
- **The TensorRT-LLM `AllReduceBestStrategyTable` contents.** `customAllReduceUtils.h` was downloaded (15,848 bytes) but the generated LUT body is machine-produced and is not quotable in a useful form; the indexing scheme `[sm][tp][fusionOp][log2(hidden)-7][log2(tokens)]` and the `SelectStrategyLP` rule are quoted from the casebook instead.
- **A multimem/NVLS equivalent on Ascend.** No Ascend hardware multicast or in-switch reduction primitive was found; SiFAR's `redundant pull` therefore has no A3 counterpart. `WriteReduceNbi` is the nearest remote-reduce primitive and is explicitly UB_CTP/URMA-only, i.e. Ascend 950 and later.
- **`DistributedGEMM` as a distinct project.** Searches returned TensorRT-LLM's `cutlass_kernels/allreduce_gemm` tree (file list quoted in Part 1's source enumeration: `gemm_universal_allreduce.hpp`, `sm90_allreduce_nvls_warpspecialized.hpp`, `sm100_visitor_allreduce_tma_warpspecialized.hpp`) rather than a separate library; no standalone documentation or measurement was retrieved.
- **Chinese-language first-party analysis of小包 AllReduce on Ascend.** Searched `昇腾 allreduce 小包 延迟`, `昇腾 AllReduce 小数据量 延迟 优化 零拷贝 AIV`, `MC2 融合通信 昇腾`, `通信计算融合`. The returns were either the official hiascend pages already quoted or third-party reposts with no measurements. No Zhihu article with primary numbers was found.

### Part 7 (in-kernel collective) negative findings, reported as found
- https://github.com/sgl-project/sglang — sglang's own tree has no in-kernel Ascend collective; the only hits for `lccl` are docs telling MindSpore users to `export MS_ENABLE_LCCL=off`. The hand-written AscendC ops live in the separate `sgl-project/sgl-kernel-npu` repo (covered above).
- https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/csrc/utils/inc/kernel/moe_distribute_base.h — contains HCCL window/context struct definitions (`flag`, `hostStateInfo`, `aicpuStateInfo`) but no magic/epoch protocol.
- https://raw.gitcode.com/cann/ops-hccl/raw/master/README.md — returns `{"code":"CH.00402000","title":"Project not found","message":"Project not found:cann/ops-hccl"}`; that repo path does not exist on gitcode.
- https://raw.githubusercontent.com/hicann/cann-recipes-infer/master/ops/shmem/distributed_embedding/op_kernel/distributed_hashtable.h — 30-line forwarding header only; the substance is in the `.cpp` (ITEM 24).
- https://raw.githubusercontent.com/Zhangyq-1999/shmem-doc/main/_sources/api/device_api.rst.txt — 72 lines of Doxygen `doxygenfile::` directives only; no inline API text. The real content is generated from the headers (ITEM 4, ITEM 20).
- Chinese web search for 「昇腾 LCCL one-shot allreduce 低时延 原理 magic 标志位」, 「昇腾 AscendC kernel 内 跨卡 同步 标志位 magic 轮次 对称内存 轮询」, 「LCCL 昇腾 低时延通信库 AllReduce 原理 共享内存 IPC buff 博客」 — no Ascend-specific blog on zhihu/csdn/cnblogs/bbs.huaweicloud.com surfaced; results were GPU-collective papers and an unrelated "LCCL" name collision in arXiv:2512.03644. All substantive material on this question is in source, not blog prose.
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/... — CANN 9.2.0 is not published on the hiascend doc portal under any probed version code. /doc_center/source/ returns HTTP 403 for 920, 92RC1, 920alpha001, 92RC1alpha001, 910, 901, 900beta3, `9.2.0` and `latest`; only 900 (CANN 9.0.0), 900beta1, 900beta2, 850 and 82RC1 resolve, for both CANNCommunityEdition and canncommercial. The 900 pages quoted above are the newest published text and already list "Atlas 350 加速卡" alongside A3, so they cover the dav-2201 A3 target.
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/<book>/ for any aclshmem/ACLSHMEM book — there is NO aclshmem API-reference book on the hiascend doc portal. Probed book names shmemapi, aclshmemapi, shmemug, shmemdevg, shmem, aclshmem, symmemapi all return HTTP 403 under /doc_center/source/zh/CANNCommunityEdition/900/API/. Repeated WebSearch ("aclshmem API 参考", "aclshmemx_signal_op 接口", "昇腾 对称内存 shmem 接口 参考", site:hiascend.com) returns only gitcode.com/cann/shmem source mirrors. The authoritative ACLSHMEM documentation is the in-repo headers and docs/ quoted above; hiascend.com carries only the HCCL-side symmetric-memory APIs (HcclCommSymWinRegister etc.).
- aclshmemx_quiet / aclshmemx_fence — no such symbols exist. The memory-ordering header declares only `aclshmem_quiet` and `aclshmem_fence` (no `aclshmemx_` spelling). The `aclshmemx_` prefix is used for the per-engine variants: aclshmemx_mte_quiet, aclshmemx_sdma_quiet, aclshmemx_udma_quiet, aclshmemx_roce_quiet.
- aclshmemx_signal_wait_until — no device-side symbol with this name. The device header declares `aclshmem_signal_wait_until` (int32_t return) and the host header declares `aclshmem_signal_wait_until` plus `aclshmemx_signal_wait_until_on_stream`. Only the `_on_stream` host variant carries the `x`.
- AscendC::DcciDst::CACHE_LINE_SINGLE — the enumerator does not exist. The CacheLine enum values are `SINGLE_CACHE_LINE` and `ENTIRE_DATA_CACHE`; the DcciDst enum values are `CACHELINE_ALL`, `CACHELINE_UB`, `CACHELINE_OUT`, `CACHELINE_ATOMIC`.
- zhihu.com / csdn.net / cnblogs.com / bbs.huaweicloud.com / juejin.cn — four separate WebSearches ("aclshmem 单边 put 可见性", "昇腾 远端写 cache 一致性 dcci", "DataCacheCleanAndInvalid 用法", "昇腾 GM 标志位 轮询 cache", plus zhihu- and CSDN-targeted phrasings) surfaced no third-party Chinese blog post on this topic. Every result resolved back to hiascend.com official docs, asc.gitcode.com, or gitcode.com/cann/*. The Chinese-language technical writing on remote-write visibility and dcci that does exist is vendor-authored and is captured above (asc.gitcode.com 缓存一致性 / 内存一致性, cann/shmem docs/api/*.md) plus the third-party mindspore-ai/hyper-parallel docs/shmem.md.
- openeuler-mirror/umdk src/cam/ — not searched for dcci/visibility material; the UMDK/CAM tree is a host-side URMA transport library and `gh search code "DataCacheCleanAndInvalid"` / `"DcciDst::CACHELINE_OUT"` returned zero hits in it, so it carries no AI-core-side cache-maintenance code.
- tile-ai/tilelang-ascend, Ascend/AscendNPU-IR, ByteDance-Seed/Triton-distributed (ascend/) — these appear in `gh search code "aclshmem_quiet"` only as thin FFI/MLIR wrappers (e.g. AscendNPU-IR bishengir/lib/Template/lib/Shmem/Shmem.cpp `_mlir_ciface_aclshmem_quiet` calling `aclshmem_quiet()`, Triton-distributed ascend/python/triton_dist/language/extra/ascend/libaclshmem_device.py mapping `(): ("aclshmem_quiet", ())`). They restate no ordering or visibility guarantee, so nothing verbatim is worth quoting.
