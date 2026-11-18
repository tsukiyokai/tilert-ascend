# 沿途下蛋：上游回流与独立成果台账

在冲刺 GLM-5.2 A3 TP16 超低时延解码（2.x ms）的过程中，上游框架常因分层封装的象征界表象而沉溺于虚假的计算对账；对芯片微架构物理实在界的解蔽，在各层裂隙中沉淀出一批具备确定性物理收益、可独立上游化的成果。本台账对全库所获成果进行全景登记与闭环管理。

## 目标仓库与命名空间

|社区/机构|目标上游仓库|工作派生分支/命名空间|管理工具/规程|
|---|---|---|---|
|GitCode CANN|`https://gitcode.com/cann/*`|`https://gitcode.com/fan33/*`|`~/.gemini/antigravity-cli/skills/vibe-pr/`|
|vLLM Ascend|`https://github.com/vllm-project/vllm-ascend`|`https://github.com/tsukiyokai/vllm-ascend`|标准 GitHub PR|
|TileLang Ascend|`https://github.com/tile-ai/tilelang-ascend`|`~/repo/tile-ai/tilelang-ascend`|标准 GitHub PR|
|TileRT|`https://github.com/tile-ai/TileRT`|`https://github.com/tsukiyokai/tilert-ascend`|标准 GitHub PR|

## 1. 成果全景总表

|ID|领域|目标上游|物理机理与消除税项|实测收益 / 状态|
|---|---|---|---|---|
|`EGG-CANN-SHMEM-001`|通信底层|GitCode `cann/shmem`|4字节team后紧跟8字节指针致非对齐，prologue加载越界使屏障失效|根治16-rank静默吞964次507015异常，恢复屏障功能|
|`EGG-CANN-SHMEM-002`|通信底层|GitCode `cann/shmem`|TCP握手遗漏EINTR重试，多信号进程（如vLLM Worker）误报107001|根治Worker初始化偶发失败，对齐兄弟头文件重试逻辑|
|`EGG-CANN-SFA-001`|核心算子|GitCode `cann/ops-transformer`|缺失宽整型模板致使隐式收缩至uint16_t，造成跨64K边界截断|40组边界测试100%通过，AIV省26条指令，时延降2.5%～5.5%|
|`EGG-CANN-SFA-002`|编译器/算子|GitCode `cann/ops-transformer`|双编译流ld.lld弱符号碰撞致Vector误跳Cube，消重调用点代noinline|消除507015异常，AIV指令体积降36.3%，算子降至127.58 µs|
|`EGG-CANN-SFA-003`|核心算子|GitCode `cann/ops-transformer`|邻接双token分核映射与两阶段6块辅助核卸载，激活Core 16..23|M32算子从142.39 µs降至127.58 µs（净省26.4 µs），max_diff=0|
|`EGG-CANN-QLI-001`|核心算子|GitCode `cann/ops-transformer`|16行批量向量化ProcessVD，合并标量发射并消除15/16等待屏障|消除15/16标量发射，单算子向量管线指令大幅压缩|
|`EGG-CANN-QLI-002`|核心算子|GitCode `cann/ops-transformer`|分块对齐单遍就地Top-K，旁路Sort32与双缓冲MrgSort归并树|单算子耗时从90.61 µs降至29.10 µs（-67.9%），单轮省1.13 ms|
|`EGG-CANN-GMM-001`|核心算子|GitCode `cann/grouped-matmul`|Batch=1直线型流水线改造，消除多阶段无谓空转保护与两次SyncAll|单次调用节省6.43 µs，消除两次混合屏障与全结构体拷贝|
|`EGG-CANN-GMM-002`|核心算子|GitCode `cann/grouped-matmul`|闭式计算group-slot span并直接复用Tiling M，消除全表冗余遍历|单次调用省4～6 µs，78层GMM单轮净降1.468 ms（每token快0.326 ms）|
|`EGG-CANN-GMM-003`|核心算子|GitCode `cann/grouped-matmul`|GMM2活跃专家动态发布与自适应Tile（M32/N1024/K128）配置|优化A8W4反量化计算流水，避免空专家浪费计算带宽|
|`EGG-VLLM-MOE-001`|推理框架|GitHub `vllm-project/vllm-ascend`|误对单张量MoE合并调用torch._foreach_add，承担21.28 µs结构性耗时|替换为普通Add（1.82 µs），75层MoE单轮净省1.4 ms，输出严格位等|
|`EGG-VLLM-CACHE-001`|推理框架|GitHub `vllm-project/vllm-ascend`|静态算子缓存键缺失vendor源码标识，导致不同工程树跨轮污染|杜绝不同宽度静态核二进制串扰，保障1024-token输出确定性|
|`EGG-VLLM-LOAD-001`|推理框架|GitHub `vllm-project/vllm-ascend`|宿主起草器场景未构建MTP模块，AutoWeightsLoader遇到rot.weight崩溃|优雅忽略未注册的前缀，避免侵入性篡改权重索引哈希|
|`EGG-VLLM-MC2-001`|推理框架|GitHub `vllm-project/vllm-ascend`|消除forward_context内每步对reserved_mc2_mask的切片与布尔重写|依token宽度预构建并静态缓存掩码，消除主机调度开销|
|`EGG-VLLM-C8-001`|推理框架|GitHub `vllm-project/vllm-ascend`|利用byte view打包FP16 scale以兼容Norm模式下K/V类型严格一致限制|修复C8 Indexer在npu_scatter_pa_kv_cache中的类型校验崩溃|
|`EGG-VLLM-HOOK-001`|推理框架|GitHub `vllm-project/vllm-ascend`|解耦MLA输出投影与门控逻辑，暴露fused_experts外挂钩子|为下游无缝插入一体化融合算子铺平接口路径|
|`EGG-FUSED-MLA-001`|融合算子|`tilelang-ascend` / TileRT|融合Attention后V-up、门控Sigmoid、O-proj与残差Add至单核|78层静态核减少3个，消灭中间GM往返，verify发射降至1.28 ms|
|`EGG-COMM-PUSH-001`|通信组件|GitCode `cann/shmem` / TileRT|单边对称内存Push配合本端按序规约替代HCCL AllReduce握手|规约时延从63.5 µs压至22.3 µs，157次规约单轮可省6.47 ms|
|`EGG-HOST-PACKED-001`|调度工程|TileRT / `vllm-ascend`|预分配固定设备地址与锁页内存，消除token/slot逐步分页H2D与tolist|78层width32的主机verify发射开销从2.136 ms压至1.200～1.370 ms|
|`EGG-TOOL-RELOAD-001`|工具链|昇腾开发者工具基建|直连bisheng编译配合LD_PRELOAD内存热替换内核二进制|算子测试周期从22.5s降至0.096s（提速83倍，热替换仅0.19 ms）|
|`EGG-INFRA-AXIOM-001`|精度基建|昇腾算子精度基建|固化A3 Fixpipe清除scale低13位、Norm-Quant间BF16舍入与HCCL加法树|建立跨软硬件层不可逾越的位等精度基准，终结虚假漂移排查|
|`EGG-CANN-MEGAMOE-001`|核心算子|GitCode `cann/ops-transformer`|A3掩码路由走独立tiling key 5188，规避动态分支回退|专有分发路径稳定入图，已建GitCode PR !4|
|`EGG-TILELANG-CARVER-A3`|编译器基建|`tile-ai/tilelang-ascend`|Carver自动调优求解器适配A3架构规范（核数/缓存/内存预算）|解锁A3自动tiling求解能力，独立单元测试全部通过|
|`EGG-TOOL-FREE-001`|工具链|昇腾集群调度基建|解析卡空闲die strip与实际运行进程，绕过内存缓存虚假占用状态|实现精准单die/少die调度，杜绝无谓等待整机闲置|

## 2. CANN 基础通信与底层算子成果

### `EGG-CANN-SHMEM-001`：参数非8字节对齐致越界加载与屏障失效

- 目标仓库：GitCode `cann/shmem`
- 载体补丁：[`scripts/patches/cann-shmem-1.7.0/two-defects.patch`](../scripts/patches/cann-shmem-1.7.0/two-defects.patch)
- 物理机制：在 `aclshmemi_barrier_on_stream_kernel(aclshmem_team_t team, uint64_t ffts_config)` 中，4字节的 team 置于 8 字节地址前，导致后随地址未能落在 8 字节对齐偏移。设备 prologue 从下一个对齐偏移加载，读取到参数缓冲区之外的高半部基址。导致 `aclshmem_barrier_all()` 在 A3 上从未正确工作，16-rank 运行直接吞下 964 次 `507015` 异常。
- 验证收据：调换形参声明顺序，使 8 字节参数居首，在 A3 16-rank 下成功恢复屏障功能，消除全部异常。

### `EGG-CANN-SHMEM-002`：TCP 握手遗漏 EINTR 导致多信号进程初始化失败

- 目标仓库：GitCode `cann/shmem`
- 载体补丁：[`scripts/patches/cann-shmem-1.7.0/README.md#L22-L39`](../scripts/patches/cann-shmem-1.7.0/README.md#L22-L39)
- 物理机制：`acc_tcp_link_complex_default.h` 仅将 `EAGAIN` 视作重试分支，遗漏了 `EINTR`，直接跌入 `ACC_LINK_ERROR` 并返回 -4，报错 `Set device id to be -1 failed: 107001`。而在其兄弟文件 `acc_tcp_link_default.h` 中已包含 `EINTR`。在拥有 NPU watchdog、Python 信号、SIGCHLD 的 vLLM Worker 中极易触发该中断。
- 验证收据：为 complex 实现补全 5 行 `EINTR` 重试分支，16-rank Worker 进程集群成功建连，消除偶发初始化失败。

### `EGG-CANN-SFA-001`：整除模板缺失致 16 位隐式收缩截断缺陷

- 目标仓库：GitCode `cann/ops-transformer`
- 载体文件：[`scripts/patches/cann-sparse-flash-attention/sparse_flash_attention_common.h`](../scripts/patches/cann-sparse-flash-attention/sparse_flash_attention_common.h)
- 物理机制：SFA 算子公共头文件引入 `using namespace AscendC`，底层 `batch_matmul_utils.h` 仅提供 `uint16_t` 重载。当传入 `int64_t` 实参时隐式向下收缩转换，结果被截断为 16 位整数返回。
- 验证收据：40 组边界测试在修复前有 20 组截断错误，修复后 100% 通过；AIV 指令体积缩减 104 字节（减少 26 条指令，`ZEROEXT.u16` 从 10 处减为 5 处），M=1 时延下降 2.80%～5.53%，M=32 时延下降 2.54%～4.20%。

### `EGG-CANN-SFA-002`：双编译流弱符号碰撞规避与指令体积压缩

- 目标仓库：GitCode `cann/ops-transformer`
- 载体文档：[`docs/probe.md#L62`](file:///home/shanshan/repo/tile-ai/tilert-ascend/docs/probe.md#L62)
- 物理机制：MIX 算子分 `aic.o` 与 `aiv.o` 两次编译后由 `ld.lld -r` 链接合并。未按 `g_coreType` 模板特化的 `__attribute__((noinline))` 产生同名 WEAK 符号，ld.lld 丢弃 `aiv.o` 版本导致 Vector 核跳转至 Cube 指令流引发 507015 异常。通过消除重复调用点替代 noinline，规避符号碰撞并保障标量提升。
- 验证收据：AIV 代码体积从 34,608 B 压缩 36.3% 至 22,048 B，`mtp_m1` 降至 55.22 µs，`target_m32` 降至 127.58 µs。

### `EGG-CANN-SFA-003`：邻接双 Token 分核与两阶段辅助核卸载均衡

- 目标仓库：GitCode `cann/ops-transformer`
- 载体源码：[`docs/probe.md#L62`](file:///home/shanshan/repo/tile-ai/tilert-ascend/docs/probe.md#L62)
- 物理机制：原分块映射将 32 个 query token 连续分配给 Cores 0..15，导致 Cores 16..23 完全闲置。改为邻接双 token 分核映射（`2*c, 2*c+1`），并利用 PreloadPipeline 在 gloop=5 前卸载 4 块、在 gloop=5..8 间再卸载 2 块，实现 16 主核算 5 块 + 8 辅助核算 6 块的精确极值平衡。
- 验证收据：`target_m32` 耗时从 142.39 µs 压至 127.58 µs（净减 26.40 µs），`max_diff = 0.00e+00`。

### `EGG-CANN-QLI-001`：16 行批量向量化 ProcessVD 消除标量开销

- 目标仓库：GitCode `cann/ops-transformer`
- 载体源码：[`scripts/patches/cann-quant-lightning-indexer/arch22/quant_lightning_indexer_service_vector_arch22.h`](../scripts/patches/cann-quant-lightning-indexer/arch22/quant_lightning_indexer_service_vector_arch22.h)
- 物理机制：原实现逐行循环发射 16 次 `Cast/Mul/ReduceSum` 与 `WaitFlag`；重构为 16×64 整块向量运算，单次发射完成 16 行计算，彻底消除 15/16 标量发射与同步等待开销。
- 验证收据：配合下述 Top-K 优化，共同促成 QLI 算子从 90.61 µs 压至 29.10 µs。

### `EGG-CANN-QLI-002`：分块对齐单遍就地 Top-K 旁路归并树

- 目标仓库：GitCode `cann/ops-transformer`
- 载体源码：[`scripts/patches/cann-quant-lightning-indexer/arch22/quant_lightning_indexer_service_vector_arch22.h`](../scripts/patches/cann-quant-lightning-indexer/arch22/quant_lightning_indexer_service_vector_arch22.h)
- 物理机制：在 `cuS2Len == cuS2LenVecAlign`（`S2_BASE_SIZE=2048 == kCount`）时，无需跨块多路归并，直接跳过 `Sort32` 与双缓冲 `MrgSort` 树，在 `reduceOutBuff` 上通过 `ArithProgression` 单遍生成索引并就地写入，消除约 65% Vector 排序指令。
- 验证收据：单算子耗时从 90.61 µs 压至 29.10 µs（-67.9%），在 78 层整模中单轮 `host_tokens` 从 39.104 ms 压至 37.977 ms，TPOT 最小降至 8.3532 ms。

### `EGG-CANN-GMM-001`：W4A8 Expert GMM Batch=1 直线型流水线

- 目标仓库：GitCode `cann/grouped-matmul`
- 载体补丁：[`scripts/patches/cann-gmm-swiglu-v2/gmm1-straight-line.patch`](../scripts/patches/cann-gmm-swiglu-v2/gmm1-straight-line.patch)
- 物理机制：在 `loopCount == 1`（所有 decode 场景）下，三阶段循环调用针对越界索引的 `preProcess(1)` 与 `postProcess(-1)` 纯属空转与同步浪费。将其展开为 `pre(0) -> sync -> mid(0) -> sync -> post(0)` 直线流水线。
- 验证收据：单次调用消除 6.43 µs，消除两次混合屏障、整结构体赋值及无谓的 `pipe->Reset()`，输出完全位等。

### `EGG-CANN-GMM-002`：闭式计算组槽跨度与复用 Tiling 行数

- 目标仓库：GitCode `cann/grouped-matmul`
- 载体补丁：[`scripts/patches/cann-gmm-swiglu-v2/gmm1-chunk-span.patch`](../scripts/patches/cann-gmm-swiglu-v2/gmm1-chunk-span.patch) 与 [`scripts/patches/cann-gmm-swiglu-v2/gmm1-tiling-rowcount.patch`](../scripts/patches/cann-gmm-swiglu-v2/gmm1-tiling-rowcount.patch)
- 物理机制：在 split=1 时累积行数恒低于限制，`UpdateWorkSpaceSplitConfig` 存在闭式解，无需遍历表（每核省 2*groupListLen+1 次全局标量读，省 2.1 µs）；`InitWorkSpaceSplitConfig` 直接复用 tiling 已有的行总数 M，消除重复累加整表（省 2～4 µs）。
- 验证收据：78 层 GMM 实测单轮净减 1.468 ms（每 token 快 0.326 ms），1024-token 输出哈希严格一致。

### `EGG-CANN-GMM-003`：GMM2 动态活跃专家发布与自适应 Tile

- 目标仓库：GitCode `cann/grouped-matmul`
- 载体补丁：[`scripts/patches/cann-grouped-matmul/gmm2-active-groups.patch`](../scripts/patches/cann-grouped-matmul/gmm2-active-groups.patch) 与 [`scripts/patches/cann-grouped-matmul/gmm2-adaptive-tiles.patch`](../scripts/patches/cann-grouped-matmul/gmm2-adaptive-tiles.patch)
- 物理机制：在 AIV Core0 上根据活跃专家计数生成掩码并发布至 workspace，配合 INT4 ND×NZ 自适应 Tile（M32/N1024/K128, baseK64）配置，跳过非活跃专家的矩阵乘计算。
- 验证收据：解决反量化流水线中的空闲专家计算空洞，提升专家并行利用率。

### `EGG-CANN-MEGAMOE-001`：MegaMoE A3 掩码路由专用 Tiling Key 5188

- 目标仓库：GitCode `cann/ops-transformer`
- 载体仓库：[`/home/shanshan/repo/cann/ops-transformer`](file:///home/shanshan/repo/cann/ops-transformer)（分支 `feat/mega-moe-a3-mask-only`）
- 物理机制：在 A3 架构的 MegaMoE 掩码路由场景下，通用动态 shape 分发逻辑易触发低效兜底路径；引入独立的专有 tiling key 5188 绑定专有内联流水。
- 验证收据：PR !4 已在 GitCode 创建，通过静态检查与基础测试。

## 3. 推理框架与调度层缺陷成果

### `EGG-VLLM-MOE-001`：消除单张量 MoE Combine 对 ForeachAdd 的滥用

- 目标仓库：GitHub `vllm-project/vllm-ascend`
- 载体源码：[`tilert/models/glm_5_2_ascend/moe_combine.py#L1-L15`](file:///home/shanshan/repo/tile-ai/tilert-ascend/tilert/models/glm_5_2_ascend/moe_combine.py#L1-L15)
- 物理机制：vLLM-Ascend 为执行 MoE 输出求和，错误调用了 `torch._foreach_add` 并传入仅 1 个张量的列表。该多张量 kernel 具有 21.282 µs 的固定结构性下发开销；而普通的 elementwise `Add` 仅耗时 1.820 µs。
- 验证收据：替换为普通加法后，75 层 MoE 单轮净省 1.4 ms，输出 BF16 张量严格位等。

### `EGG-VLLM-CACHE-001`：静态算子缓存键补全源码标识防串扰

- 目标仓库：GitHub `vllm-project/vllm-ascend`
- 载体补丁：[`scripts/patches/vllm-ascend/static-kernel-cache-key.patch`](../scripts/patches/vllm-ascend/static-kernel-cache-key.patch)
- 物理机制：`_get_static_kernel_cache_key` 仅根据算子形参和选项构建哈希，遗漏了 vendor 源码路径，且索引写入工作目录。导致不同工程树、不同编译版本在运行时互相复用旧二进制，造成不可预期的跨宽度污染。
- 验证收据：加入源码标识与环境隔离后，16-rank width 8 场景下成功稳定输出唯一的基准 token 流（`68f2bc184553`），消除宽度串扰。

### `EGG-VLLM-LOAD-001`：宿主起草器场景容忍未注册的 MTP 权重

- 目标仓库：GitHub `vllm-project/vllm-ascend`
- 载体补丁：[`scripts/patches/vllm-ascend/target-ignores-mtp-rot.patch`](../scripts/patches/vllm-ascend/target-ignores-mtp-rot.patch)
- 物理机制：Checkpoint 索引包含 `rot.weight`，但当使用 host drafter 时 `speculative_config=None`，不构建 MTP 模块，`AutoWeightsLoader` 抛出 `ValueError: There is no module or parameter named 'rot'`。修改为忽略该前缀。
- 验证收据：避免了直接修改权重索引文件所导致的 SHA-256 破坏与捕获图失效问题。

### `EGG-VLLM-MC2-001`：按 Token 宽度预构建 MC2 掩码

- 目标仓库：GitHub `vllm-project/vllm-ascend`
- 载体补丁：[`scripts/patches/vllm-ascend/prebuilt-mc2-mask.patch`](../scripts/patches/vllm-ascend/prebuilt-mc2-mask.patch)
- 物理机制：消除 `ascend_forward_context.py` 中每步 forward 对 `reserved_mc2_mask` 进行的动态张量切片与布尔赋值，根据 token 宽度（1、8、32）预先构建并静态缓存。
- 验证收据：消除重复的 CPU 主机调度与张量构造损耗。

### `EGG-VLLM-C8-001`：C8 Indexer KV Cache 散弹写入类型对齐修复

- 目标仓库：GitHub `vllm-project/vllm-ascend`
- 载体补丁：[`scripts/patches/vllm-ascend/paired-c8-index-cache.patch`](../scripts/patches/vllm-ascend/paired-c8-index-cache.patch)
- 物理机制：`torch_npu.npu_scatter_pa_kv_cache` 在 Norm 模式下强制要求 Key 与 Value 数据类型一致。对于 C8 indexer，利用 byte view 将 FP16 scale 打包成 int8 传入，规避算子底层断言崩溃。
- 验证收据：成功支持 C8 稀疏索引特征与 Scale 向量的协同缓存落地。

### `EGG-VLLM-HOOK-001`：MLA 投影与门控解耦暴露融合钩子

- 目标仓库：GitHub `vllm-project/vllm-ascend`
- 载体补丁：[`scripts/patches/vllm-ascend/mla-tail-hook.patch`](../scripts/patches/vllm-ascend/mla-tail-hook.patch)
- 物理机制：将 Attention 后的 V-up 投影、门控 Sigmoid 计算与最终 O-proj 汇聚拆分为独立可插拔方法 `_project_attention_output`，统一 latent input 与 gate hidden state 的投影流。
- 验证收据：为下游直接接入单核融合算子提供了干净的上游调用契约。

## 4. 极致融合微算子与通信组件

### `EGG-FUSED-MLA-001`：MLA Tail 单核融合算子

- 目标仓库：`tile-ai/tilelang-ascend` / TileRT
- 载体源码：[`tests/glm52/mla_tail/mla_tail.h`](file:///home/shanshan/repo/tile-ai/tilert-ascend/tests/glm52/mla_tail/mla_tail.h) 与 [`tilert/models/glm_5_2_ascend/mla_tail.py`](file:///home/shanshan/repo/tile-ai/tilert-ascend/tilert/models/glm_5_2_ascend/mla_tail.py)
- 物理机制：将 SFA 之后的 V up-projection (4×256)、门控 Sigmoid、O-projection (6144) 与 Residual Add 融合成单一常驻核，消除多算子调度、中间张量 GM 往返与 3 次 kernel 启动。
- 验证收据：78 层 TP16 静态核数减少 3 个（244 -> 241），verify 发射开销降至 1.28 ms，16-rank 输出逐位一致。

### `EGG-COMM-PUSH-001`：单边对称内存 Push 与本端有序规约

- 目标仓库：GitCode `cann/shmem` / TileRT
- 载体源码：[`tilert/models/glm_5_2_ascend/onesided_reduce.py`](file:///home/shanshan/repo/tile-ai/tilert-ascend/tilert/models/glm_5_2_ascend/onesided_reduce.py)
- 物理机制：用单边 push 到对端 symmetric window 并在本端按 rank 顺序求和，替代阻塞式 HCCL AllReduce，消除 rendezvous 握手延迟。
- 验证收据：单次规约耗时从 63.5 µs 压至 22.3 µs，157 次规约单轮可节省 6.47 ms，严格 rank 顺序相加保障浮点加法位等。

### `EGG-HOST-PACKED-001`：宿主零同步静态信封与批量发射

- 目标仓库：TileRT / `vllm-ascend`
- 载体源码：[`tilert/models/glm_5_2_ascend/cann_head.py`](file:///home/shanshan/repo/tile-ai/tilert-ascend/tilert/models/glm_5_2_ascend/cann_head.py) 与 [`docs/probe.md#时间余量的证据边界`](file:///home/shanshan/repo/tile-ai/tilert-ascend/docs/probe.md#时间余量的证据边界)
- 物理机制：将解码轮内的 token、position、slot 映射、有效序列长度槽等全部绑定到预分配固定设备地址与锁页主机内存，消除每步分页 H2D 拷贝、事件创建与 `.tolist()` 阻塞。
- 验证收据：78 层 width32 的 verify 主机发射时间从 2.136 ms 压至 1.200～1.370 ms。

## 5. 开发者基建与硬件物理测绘资产

### `EGG-TOOL-RELOAD-001`：直连 Bisheng 编译与 LD_PRELOAD 0.19 ms 热重载套件

- 目标仓库：昇腾开发者工具基建 / `tilelang-ascend`
- 载体代码：[`evidence/sfa-fast-iter-20261031/fast_build/fast_compile_sfa.sh`](evidence/sfa-fast-iter-20261031/fast_build/fast_compile_sfa.sh) 与 [`evidence/sfa-fast-iter-20261031/hotreload_sfa.c`](evidence/sfa-fast-iter-20261031/hotreload_sfa.c)
- 物理机制：彻底绕过 Python/TBE/opc 包装流，8.6s 直连 bisheng 并行编译 aic/aiv，并通过 `LD_PRELOAD` 拦截 `aclrtBinaryLoadFromData` 与 `aclrtBinaryGetFunctionByEntry`，在常驻 Python 进程内以 0.19 ms 热替换 `.o` 内核。
- 验证收据：单次测试循环从 22.5s 降至 0.096s（提速 83 倍），消除模型重建与权重重新加载开销。

### `EGG-INFRA-AXIOM-001`：A3 硬件数值公理门禁系统

- 目标仓库：昇腾算子精度基建
- 载体文档：[`docs/lesson.md#3-确认硬件实际执行的数值规则`](file:///home/shanshan/repo/tile-ai/tilert-ascend/docs/lesson.md#3-确认硬件实际执行的数值规则) 与 [`tests/experiments/cann-precision/check.py`](../tests/experiments/cann-precision/check.py)
- 物理机制：确立 A3 硬件不可逾越的三大算术公理：1) Fixpipe 反量化前清除 FP32 scale 低 13 位；2) Norm 与 Quant 之间的 BF16 中间舍入；3) 确定性 HCCL `[8,4,2,1]` 规约树加法结合序。
- 验证收据：解释并消除了 CPU 参考实现与真实 NPU 的系统性漂移，建立 16-rank 位等精度门禁标准。

### `EGG-TOOL-FREE-001`：基于 Die Strip 与真实物理进程的精准调度探针

- 目标仓库：集群调度工具 / 开发者基建
- 载体脚本：[`scripts/free_dies.sh`](file:///home/shanshan/repo/tile-ai/tilert-ascend/scripts/free_dies.sh)
- 物理机制：解决 `npu-smi` 状态因系统内存缓存低于 15% 虚假报告“占用”的问题，通过解析卡空闲 die strip 与实际运行进程，精准识别可用 die。
- 验证收据：支持单 die/少 die 实验免等待快速入场，大幅提升集群卡位周转效率。

### `EGG-TILELANG-CARVER-A3`：Carver 硬件求解器适配 A3 架构与资源规约

- 目标仓库：`tile-ai/tilelang-ascend`
- 载体代码：[`tilelang/carver/arch/ascend.py`](file:///home/shanshan/repo/tile-ai/tilelang-ascend/tilelang/carver/arch/ascend.py)、[`tilelang/carver/arch/ascend_specs.py`](file:///home/shanshan/repo/tile-ai/tilelang-ascend/tilelang/carver/arch/ascend_specs.py) 与 [`testing/python/carver/test_ascend_specs.py`](file:///home/shanshan/repo/tile-ai/tilelang-ascend/testing/python/carver/test_ascend_specs.py)
- 物理机制：Carver 自动调优布局求解器此前仅支持 910A/910B/310P，缺乏 A3（9392/9382，架构 2201）的硬件模型。本成果从 SDK INI 规范中解析并注入 A3 的 24 Cube / 48 Vector 核数、192MB L2 缓存、UB/L1/L0 内存预算与 BF16 张量核能力。
- 验证收据：`test_ascend_specs.py` 完整单元测试全部通过，准确建立 A3 硬件空间拓扑约束。
