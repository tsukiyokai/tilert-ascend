# 实施计划

TileRT-Ascend 是构建在 Ascend 上的专用低时延 decode 框架：它从成熟服务栈接收 prefill 交来的历史状态，接管单请求或极低并发下的 token 生成关键路径，并对输出质量、请求状态与完整解码时间负责。项目定位见 [readme]，实验方法见 [lesson]，实测账本与正负证据见 [probe]。

## 当前验收

A3 单 die 已通过 [token staging 对照](evidence/ascend-decode/token-staging-a3-20261028/README.md)，width32 配对节省 p50 为 2.13–2.24 ms/整数夹具轮；[SFA 元数据预搬运](evidence/ascend-decode/sfa-metadata-a3-20261028/README.md)数值通过却使配对时延退化，保留原实现。规约所有权与数据交换原型、Carver 资源修正均已留档；组件结果继续接到真实权重、TP16 和在线门禁。

整模数值基线与独立组件计时分别推进。[修正包TP16复验](evidence/ascend-decode/fixed-hf-launch-20261029/README.md#9392完整复验)已取得576个PT，16rank状态与reset_exact通过，HF及M1/M32仍失败。[第一层边界](evidence/ascend-decode/layer0-boundaries-a3-20261029/README.md)已把首差收窄到SFA及上游。[BF16 score与先展开Value组合](evidence/ascend-decode/attention-rounding-a3-20261029/README.md)的真实TP16 h1误差从0.003595降至0.001961；前三层h3改善未达预注册30%目标。[完整常驻模型](evidence/ascend-decode/resident-attention-a3-20261029/README.md)现已完成两臂各首帧与精确reset，16rank全字段一致；组合把首个hidden越界h7推迟到h8、首个专家选择分歧从layer6推迟到14，最终logits NRMSE为0.0771525→0.0608584，仍未过精度门禁。[prefix Value缓存](evidence/ascend-decode/prefix-values-a3-20261029/README.md)已在单die验证一次准备、decode仅投影当前行的接线；[三臂计时](evidence/ascend-decode/prefix-value-timing-a3-20261029/README.md)中缓存墙钟p50为906.016µs，慢于展开的783.941µs及原生的263.544µs，未达预测，结束此实现的性能接入。[layer68索引顺序干预](evidence/ascend-decode/layer68-boundaries-a3-20261029/README.md)已使选定row13在16rank的M1/M32 h69及logits逐位一致；[完整SFA计时](evidence/ascend-decode/sfa-order-timing-a3-20261029/README.md)显示排序未获得双钟净收益，停止此实现的性能接入。数值侧继续闭合完整HF、其他行、八个冻结8K输入与在线双时钟；性能侧转向改变数据流、通信所有权或执行包络的对照，先验收完整组件成本，再接真实整轮。既有规约和graph原型保留。

[合包输入](evidence/ascend-decode/host-inputs-a3-20261029/README.md)在9382的width32准备窗口取得墙钟18.082%、event29.823%的p50下降，墙钟未达预注册20%，保留可选候选。[真实TP16消费检查](evidence/ascend-decode/host-inputs-model-a3-20261029/README.md)的96份实际PT已通过独立读回，两臂复现旧设备输出及新增KV；仅一个8K输入、eager数值范围。下一步先取得同包真实width32整轮的独立诊断trace与profiler-off窗口，再量通信buffer两端cast融合的完整消费者成本；保留原HCCL结合顺序与BF16舍入，剖析请求排除于TPOT统计。

[多样性门禁](evidence/ascend-decode/diversity-gate-20261029/README.md)已拒绝双钟1ms的常量输出反例，36项CPU检查通过。正式计时须预注册并冻结token/bigram policy；缺policy仍可测量，`passes_2ms=false`。真实CANN收据接入、八输入与完整会话质量继续验收。

CPU 接入、状态、计时修复及正负结果见 [10-28 账本](probe.md)。[同源码原生 lookup](evidence/ascend-decode/lookup-native-20261028/README.md)继续通过[无环节点 GC 对照](evidence/ascend-decode/lookup-gc-20261028/README.md)减少本地构造分配与典型成本，复制收益很弱且长尾仍有退化；默认部署尚未安装扩展。[78 层合成 HF](evidence/ascend-decode/hf-runtime-20261028/README.md)已有 [A/B/A 各 32 行的完整参考帧](evidence/ascend-decode/graph-frame32-20261028/README.md)；[持久观测槽](evidence/ascend-decode/graph-producer-20261028/README.md)已在 CPU AOT 的实际 verify 入口检查字段新鲜度与坏图拒绝。下一步闭合实际 NPU/SuperKernel graph→HF 观测与正式 CANN 门禁，检验请求观测成本；设备验证按[单 die、少 rank、空闲 A3](evidence/ascend-decode/online-lookup-20261028/README.md)推进，先核对构造副作用。整模型精度与在线 TPOT 仍待验收，当前未宣告 CPU 前沿耗尽。

## 框架定位与物理前提

构建专用低时延 decode 框架，首先要越过两重幻象。第一重是自上而下的抽象框架幻象：在尚未打穿单条极限物理路径前先搭通用算子注册表、多模型插件层与调度外壳，只会把错误的内核边界固化成维护负担；逆向拆解上游 ROCm 产物可见（见 [ROCm 融合结构][tilert-rocm]），其逐原语通用 GEMV 库在二进制中完全不可达，真正支撑单用户 469 tok/s 的是整层静态展开内核与携带代次标签的核内点对点写。第二重是 GPU 持久大内核的同构幻象（见 [AMD 机制重读][amd-reread] 与 [Ascend 融合机制][ascend-fusion]）：Ascend A3（`dav-2201`）在物理上切开了 AIC 与 AIV，不存在 `L0C→UB` 通路，跨核交接必经 `L0C → Fixpipe → L2/GM → MTE2 → UB` 并配齐 `DataSyncBarrier<DDR>`；`CrossCoreSetFlag` 仅有 16 个不可清零的 4 位硬件计数器（上限 15 次），且 `CrossCoreWaitFlag` 在 `dav-2201` 上会阻塞等待核全部流水线，强行跨越 78 层串联硬件信号必触发死锁或溢出。

当前可复核的整模型锚点是同一冻结前缀下的内部 TPOT 12.413 ms，正式精度、多输入与在线门禁仍待验收。后续预算以同版本、同轨迹的整轮墙钟和受控替换为据，记录于 [机制档案与余量账本](probe.md)。[A3 流量模型][a3-roofline]的 4.056 GB/rank 与 3.62 ms，以及历史 PIPE_S、宿主周转计数，提供实验先验；各流水线计数的重叠关系和实际访存重用须随实验确认。token 批量准备在真实 A3 上形成组件收益，SFA 元数据预搬运则已退化，表明结构替换仍须由实际消费者的数值和墙钟共同筛选。2.x ms 保持为待实测达成的目标，优先比较宿主准备、索引/KV 数据流、通信所有权与层内融合，再从多条已验证路径中提炼框架接口。

## 首条验证锚点与演进路线图

| 项目 | 范围 |
| -- | -- |
| 模型与精度 | 真实 GLM-5.2-W4A8C8 checkpoint，保留原稀疏 attention、量化、采样与 KV 语义 |
| 解码拓扑 | 一台 A3，16 die（`Ascend910_9392` 或 `9382`），TP16，单请求常驻服务；单算子与少卡实验可直接利用局部空闲 die 推进 |
| 基准负载 | 八个独立 8192-token 输入，每请求连续生成 1024 个 token |
| 终极时延门禁 | 生产端与客户端双时钟 32 请求 p50/p95 TPOT ≤ 2.x ms（`passes_2ms`） |
| 成果沉淀目标 | 沿途下蛋回流全部 GitCode `cann/*` 仓库、`vllm-ascend`、`tilelang-ascend` 与 `TileRT` |
| 扩展工作负载 | 短对话、长短交替会话、EOS 与取消恢复、百万上下文 AgentX 回放 |

已有 [冻结前缀宽验证计时][width32]（内部 TPOT 中位数 12.413 ms，正式精度与多输入门禁未开）、[真实 P/D 冷启动链路][pd-live] 与 [三 rank 原生树规约整层检查][tree-check]。下述五个阶段为可证伪的工作假说地图而非教条禁锢：任何编译器/ISA 栈、规约拓扑（平面直写、两级规约或树规约）或常驻执行结构，只要经真实 NPU 精度门禁确认且能压降墙钟时延，均可超越或合并中间阶段：

| 阶段 | 周期参考 | 核心目标与交付物 | 整轮与 TPOT 门禁 |
| -- | -- | -- | -- |
| P0 | 第 1–2 周 | 收束单一运行时状态机，锁死 8×8K 精度门禁与常驻 16 路 HIXL 宽验证 P/D | 精度与状态全过，客户端 TPOT ≤ 12.0 ms |
| P1 | 第 3–6 周 | 肃清宿主周转空洞与 `PIPE_S` 标量全局读阻塞，打通 `dav-2201` Tile 直通图编译链 | 整轮 35–40 ms，稳态 TPOT ≤ 7.0 ms |
| P2 | 第 7–12 周 | 78 层真实权重双积木具体化（索引注意力＋专家归并）与 `aclshmem` 核内确定性树规约 | 整轮 20–24 ms，稳态 TPOT 3.5–4.5 ms |
| P3 | 第 13–18 周 | `SoftSyncAll` 跨层静态权重预取、L2 冷热隔离与混合草拟树动态宽度闭环控制 | 双时钟通过 `passes_2ms`，AgentX 达标 |
| P4 | 第 19 周起 | 凝结 `compiler` / `comm` / `spec` / `pd_vllm` 四层框架，接入第二模型与新拓扑/芯片 | 双路径共用核心，净收益抵偿维护成本 |

## P0：闭合真相基座与常驻交接契约

当前仓库并存三条未完全合流的执行路径：四层 BF16 功能参考 [DecodeEngine][engine]、78 层算子对账路径 [NativeDecodeEngine][native-engine]，以及复用 vLLM-Ascend 模型加载与图的 [CannModel][cann-model]。P0 的首要任务是将离线基准与在线服务收束到同一套运行时状态机与不可绕过的精度门禁 [precision_gate.py][precision-gate]，终结仅有速度读数而无正式精度凭据的状态。

1. 身份与溯源锁死：固定 checkpoint revision、量化配方、109 个权重文件清单与全量字节哈希，完整加载 78 层主干及第 78 层 MTP 权重。每次运行随结果保存 CANN、驱动、`torch`/`torch_npu`、配套 `vLLM`/`vLLM-Ascend`、编译器参数、算子 overlay、Python 源码与实际加载设备 ELF 的哈希，并用独立缓存目录隔离编译产物。
2. 硬件精确数值公理：按 [输入与精度协议][measurement] 冻结八个独立 8192-token 输入、前缀快照、首 token 与请求标识。将已实证的硬件算术规则固化为参考对账标准：复现 A3 Fixpipe 反量化前清除 FP32 scale 低 13 位的截断行为，保留 Norm 与 Quant 之间的中间 BF16 舍入，锁死确定性 HCCL 的 `[8,4,2,1]` FP32 规约树加法结合顺序。逐宽度、逐起草器比对层输出、logits、路由、main/indexer/MTP KV 与完整贪心序列，阈值不随失败结果放宽。
3. 常驻在线 P/D 贯通：打通 [CANN 服务入口][serve-pd]、[AscendMtpPDEngine][mtp-session]、[AscendTileRTConnector][connector]、[KV 布局协议][layout] 与 [MTP parent 交接][parent-code]，废除先停 prefill 再加载 decode 的分时冷启动流程。规划独立 P/D 节点或覆盖加载预热峰值、多宽度图池、通信窗口及临时缓冲的确定性显存预算，使在线入口直接承载 `width=16/32` 宽验证、编译目标图、SuperKernel 与 FP32 head。
4. 事务级状态与故障协议：遵循 [P/D 状态协议][pd-contract] 与 [parent 交接约定][parent-contract] 验收全生命周期。输入长度为 P 时交接主 KV 的 P 行、MTP 的 P−1 行及未旋转的目标模型 P−1 位置 final-norm hidden，首 token 仅交付一次且 pending token 与已提交 KV 严格分离；草稿零接受、部分接受、全接受、EOS 裁剪与取消均在全部 16 个 rank 同步回滚尾部；活跃 HIXL WRITE 完成前保留源页与接收缓冲，断连失联时隔离槽位，迟到消息不得复活旧请求；TP 广播命令序号失配时立即停止复用并执行整组恢复；不支持的采样或配置在准入前显式拒绝或分流至通用后端。

验收标准：八个冻结 8K 输入、长短交替请求与取消重放全部通过 16 rank 一致性、reset 隔离与串行参考对账；常驻服务无需人工触发文件即可连续响应，分别出冷启动、TTFT、HIXL 传输与稳态解码报告，客户端稳态 TPOT ≤ 12.0 ms。

## P1：肃清宿主空洞、标量阻塞与打通 Tile 编译链

在重写整层算子前，先在 P0 可信基线上拔除宿主调度空洞与设备标量管线阻塞，并建立绕开厂商 OPP 工程束缚的直接编译底座（对照见 [低时延宿主杠杆目录][latency-baselines] 与 [自定义算子包装机制][op-packaging]）。

1. 零同步静态宿主信封：将解码轮内的 token、position、slot 映射、[RoPE 刷新][cann-model]、有效序列长度槽、跨层稀疏选择代次与拒绝采样 EOS 掩码全部绑定到预分配固定设备地址与锁页主机内存，消除每步分页 H2D 拷贝、事件反复创建销毁、`.tolist()` 阻塞以及退化为 `NonZero` 强制同步的布尔掩码索引；将 MoE 权重预取激活阈值降至 1 token；实现词表并行局部 argmax，仅跨 rank 收集 `(value, index)` 二元组并直接送入设备侧验证器，消除全词表 FP32 logits 聚合。
2. 索引与 KV 数据流：SFA 的 top-k/页表预搬入 UB 已在 width32 退化，结束该轮预算并保留原实现。下一种索引方案须改变实际重复访问、布局或消费者同步，提出新的定量预测，再做隔离缓存、变化输入的配对差分。MoE 分组元数据与跨查询索引共享分别验收，收益取自对应完整阶段及紧邻消费者。
3. `dav-2201` Tile 编译与图直通底座：修正 `tilelang-ascend` 中 `carver/arch/ascend.py` 将 `Ascend910_9392` 误判为 `Ascend910A` 的缺陷，补齐 24 AIC / 48 AIV 核配比、双缓冲 64 KB L0C、真实 UB/L1 容量阶梯，以及裸 `bisheng -xcce --npu-arch=dav-2201` 编译下缺失的 `MTE2_S`、`S_MTE3` 与 `V_MTE2` 显式同步推导；统一通过 `torch.library` 的 `PrivateUse1` 分派键与 `fake_impl` 将 TileLang 生成内核及受控 AscendC/PTO 内核直接注册进 ACLGraph 捕获流，停止修补受限于固定函数原型的厂商二进制算子包。

验收标准：无 profiler 干扰下整轮设备墙钟从 61–71 ms 压降至 35–40 ms，5.26 ms 宿主周转空洞消失，标量管线固定开销大幅回落，八个冻结输入在线稳态 TPOT ≤ 7.0 ms 且数值完全符合 P0 门禁。

## P2：层内双积木具体化与核内单边通信内生化

P2 将 [WholeLayerManager][whole-layer] 从单/双/三 rank 合成权重的原型推进为承载 78 层真实 W4A8C8 权重与 16 个物理 die 的默认执行引擎。顺应 A3 无片上 `L0C→UB` 通路但共享大容量 L2 的物理结构，层内融合的目标是让跨核交接命中 L2、常量化静态 shape 以削去 tiling 标量税、避开无谓的 13.58 µs `MIX` 启动税（纯计算段优先采用 2.30 µs `AIC_ONLY` 与 0.70 µs `AIV_ONLY` 或受控 `__schedmode__(1)`），并将每层收敛为两块边界清晰的标准执行积木：

1. 稀疏索引与 MLA 注意力积木：按 [indexed attention 契约][attention] 组合前置残差归一化量化、分页 INT8/BF16 Lightning Indexer top-2048 选择、MLA 前处理、稀疏 KV 读取及输出投影尾段。覆盖负 slot、越界掩码与全部声明验证宽度，固定单请求与分桶宽度下的 `MatmulApiStaticTiling` 常量参数，并对多核广播读取施加 `(mBlockIdx + blockIdx) % numBlocks` 错位分核以消除 512 字节同地址窗口串行化。
2. W4A8 路由/共享专家与核内归并积木：按 [W4A8 路由规范][routed] 与 [完整 FFN 契约][ffn] 组合路由元数据准备、A8W4 MSD 高低半字节矩阵乘、SwiGLU、Fixpipe `REQ8`/`VREQ8` 重量化（在满足舍入等价处直接省去向量侧 Cast）、down 投影及原始 top-8 顺序归并。针对 `moe_intermediate=1792` 在 INT4 下为 896 字节（非 256 字节对齐）导致的随路转换损耗，采用离线对齐排布或向量侧预转换；利用侧流（捕获期记录串行默认流、重放期开启并发以规避图池流注册限制）或核组交错让共享专家计算及归约与路由专家重叠。
3. 核内单边对称内存规约：按 [低时延 TP 规约研究][tp-reduce]、[TP 层链][tp-layer]、[三 rank 规约顺序审计][reduce-diagnostic] 与 [树规约整层检查][tree-check]，用基于 `aclshmem_ptr` 的 MTE3 远端直写或 `aclshmemx_mte_atomic_add` 配合绕过 DCache 的单调递增窗口代次（epoch）轮询，接管每层注意力后与 FFN 后的两次跨卡规约，彻底移除每轮 156 次独立 HCCL 内核下发。实现须主动利用 A3 模组内 0.31 µs 对跨模组 0.84 µs 的物理拓扑，严格复现 TP16 下的 `[8,4,2,1]` 确定性 FP32 加法树括号，并保证捕获前按宽度与 stream 预留对称窗口、重放期设备侧自推进代次、销毁时严格遵循「停提交→同步设备→销毁引用图→跨 rank 会合→释放窗口→finalize」的生命周期。
4. 分桶静态图与 SuperKernel 编排：在整模型 78 层上按验证宽度分桶（`M ∈ {1, 4, 12, 16, 32}`）捕获完整 ACLGraph 并开启 CANN `SuperKernel`（含 `early_start`），用同一真实 shape 与精度横向实测 TileLang-Ascend 生成代码与手写 AscendC/PTO 实现，择优固化进主干。

验收标准：78 层真实 checkpoint 在 16 个物理 die 上完成局部对比、变化输入图重放、整模型精度门禁与 P0 常驻 P/D 回归，覆盖跨轮变化 payload、立即消费者与特殊浮点边界值；整轮墙钟降至 20–24 ms，稳态 TPOT 达到 3.5–4.5 ms。

## P3：跨层持久流水与自适应投机闭环

P3 消除层与层之间残余的权重加载气泡，并将投机草拟与底层硬件执行斜率耦合为闭环控制器，冲刺 2.x ms 物理极限。

1. 零硬件信号消耗的跨层静态权重预取：避开 4 位硬件 `CrossCoreSetFlag` 的 15 次饱和上限与 `CrossCoreWaitFlag` 全流水阻塞，在相邻稀疏索引层与共享索引层之间采用基于全局内存与 UB 软件记分牌（`SoftSyncAll` 机制）或 `SuperKernel` 预取描述符的细粒度依赖推进。当第 `l` 层尾部执行专家归并与单边跨卡发布时，空闲 MTE2 通道提前将第 `l+1` 层共享专家及注意力静态投影权重预取入 L2；同时对单请求仅流式读取一次的冷路由专家权重施加 `SetL2CacheHint(CACHE_MODE_DISABLE)`，防止冷分片驱逐热激活与预取权重。
2. 投机策略与硬件斜率的联合闭环：打通 [原生链式 MTP][native-mtp]、[后缀索引起草][lookup-draft]、[自动机起草][automaton-draft] 与 [树掩码生成][tree-mask]。在固定确定性目标串上逐宽度核查轮起点、树因果掩码与接受位置，定位并消除如 width24 接受数异常回落等非单调效应；运行时依据实时后缀匹配率与实测宽度成本斜率 `T_round(W) / E[accept(W)]` 动态切换起草器与验证宽度分桶（低重复语境切回链式 MTP，高重复智能体轨迹切入后缀/自动机宽树），将策略切换本身的开销计入总墙钟。

验收标准：在八个独立 8K 输入、连续 32 次在线 P/D 请求（3 次预热后无 profiler 测量）下，生产端回调时钟与客户端接收时钟的请求 TPOT p50/p95 均不超过 2.0 ms（`bench_pd_decode.py::passes_target` 的实际门槛），正式点亮 `passes_2ms`；同步交付百万上下文 AgentX 会话回放报告，分别列出排队、prefill、HIXL 传输、稳态 decode、网络与工具耗时以及任务完成质量。

## P4：从单路径具体化凝结为专用解码框架

严格遵循奥卡姆剃刀：唯有在 GLM-5.2 / A3 TP16 达成 2.x ms 之后，且由真实的第二模型（如 DeepSeek-V3.2 DSA 或 Qwen3.5 / GLM-5.3 系列 MoE）及第二硬件或并行形态（同机 TP8+EP8 混合并行与热点专家复制驻留，或具备原生 `L0C→UB` 通路与完整 MTE 原子操作但须重新设计 W4 权重路径的下一代 Ascend `950PR/950DT`）驱动时，才将两条路径共同依赖的不变量沉淀为正式的四层框架接口：

1. `tilert.ascend.compiler`（声明式层图与 Tile 编译层）：以统一阶段图表达层内双积木与跨层预取依赖，后端对接校准好芯片画像的 TileLang-Ascend 与受控 AscendC/PTO 模板，自动注入跨核 `DataSyncBarrier<DDR>`、`PIPE_S` 显式事件配对、静态常量 tiling 与内容寻址 ELF 缓存校验。
2. `tilert.ascend.comm`（拓扑感知设备通信与传输层）：封装单节点 HCCS 模组层级与跨机 RoCE 下的 `aclshmem` 对称内存窗口、代次推进协议、确定性规约树生成，以及面向 P/D 分离的多路 HIXL 零拷贝页映射传输。
3. `tilert.ascend.spec`（事务型投机与缓存状态引擎）：统一管理主缓存、混合精度分页索引缓存与草稿层缓存的分配、增量前缀复用、多分支树掩码验证、原子提交与拒绝回滚，以及词表并行贪心、分布式 top-p 与结构化输出采样。
4. `tilert.pd_vllm`（分级服务接入与共生池化层）：通过标准 `KVConnector` 与 `MultiConnector` 无侵入接入主流推理服务栈，在统一 OpenAI 路由入口下实现 SLO 分级调度——将延迟敏感的实时交互、高频决策与长程智能体会话导入 TileRT-Ascend 专用低时延解码池，将大吞吐离线批处理流量留在常规解码池，当上游通用后端在同等负载下达到同等质量、成本与时延时及时收敛重复维护。

[a3-roofline]: evidence/a3-roofline.md
[amd-reread]: evidence/tilert-ascend/amd-article-reread-20261018.md
[ascend-fusion]: reference/ascend-fusion-mechanics.md
[attention]: ../tests/glm52/whole_layer/attention_indexed.md
[automaton-draft]: ../tilert/models/glm_5_2_ascend/automaton_draft.py
[cann-model]: ../tilert/models/glm_5_2_ascend/cann_model.py
[connector]: ../tilert/pd_vllm/ascend_connector.py
[engine]: ../tilert/models/glm_5_2_ascend/engine.py
[ffn]: ../tests/glm52/whole_layer/ffn_original_shared_gu_static/README.md
[latency-baselines]: reference/ascend-decode-latency-baselines.md
[layout]: ../tilert/pd_vllm/ascend_layout.py
[lesson]: lesson.md
[lookup-draft]: ../tilert/models/glm_5_2_ascend/lookup_draft.py
[measurement]: evidence/ascend-decode/measurement.md
[mtp-session]: ../tilert/models/glm_5_2_ascend/mtp_session.py
[native-engine]: ../tilert/models/glm_5_2_ascend/native_engine.py
[native-mtp]: ../tilert/models/glm_5_2_ascend/native_mtp.py
[op-packaging]: reference/ascend-custom-op-packaging.md
[parent-code]: ../tilert/pd_vllm/ascend_parent.py
[parent-contract]: evidence/ascend-decode/mtp-parent-handoff.md
[pd-contract]: evidence/ascend-decode/pd-ascend.md
[pd-live]: evidence/ascend-decode/pd-live-20261017/README.md
[precision-gate]: ../tests/glm52/precision_gate.py
[probe]: probe.md
[readme]: readme.md
[reduce-diagnostic]: evidence/ascend-decode/megakernel-device-20261021/integration/communication-order-three-rank-v1-a3-194/AUDIT.md
[routed]: ../tests/glm52/routed_int4/README.md
[serve-pd]: ../tests/glm52/serve_cann_pd.py
[tilert-rocm]: reference/tilert-rocm-glm52-structure.md
[tp-layer]: ../tests/glm52/whole_layer/layer_tp_chain/README.md
[tp-reduce]: reference/low-latency-tp-reduce.md
[tree-check]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-tree-v1-a3-194/AUDIT.md
[tree-mask]: ../tilert/models/glm_5_2_ascend/tree_mask.py
[whole-layer]: ../tilert/models/glm_5_2_ascend/whole_layer.py
[width32]: evidence/ascend-decode/width-ladder-20261023/w32-lookup.json
