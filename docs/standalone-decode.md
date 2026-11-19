# GLM-5.2 TP8+EP8 独立解码终态架构

本文定义 TileRT-Ascend 冲刺 SemiAnalysis InferenceMAX 标准 8K/1K 榜极限交互性端点的唯一终态架构：硬件锁死单台 Ascend A3（Ascend910_9392 或 9382，dav-2201 架构，8 物理 die），模型锁死 GLM-5.2 W4A8C8（TP8+EP8），负载锁死 batch=1、ISL=8192、OSL=1024 标准代码生成用例 [case-01-coding](../tests/glm52/megakernel_tp8_ep8/case-01-coding)（冻结提示 SHA256 为 a93d673b...，首词 ID 为 13534），终极目标为稳态 TPOT < 3.0 ms，且 1024 词元输出严格复现黄金 SHA256（2f751372fc82cf290c8e35d3e7cc3cad0296fde4d8d188fba102ac34f82c2962）并与 8 卡主网及独立串行草稿有效状态位级对齐。TP16、非 coding 场景、Prefill 时延（TTFT）、通用在线 HTTP 网关及离散分页块表兼容均非目标，全部从核心引擎剥离。项目背景见 [readme.md](readme.md)，阶段路线见 [plan.md](plan.md)，投机实测见 [speculative-decoding.md](speculative-decoding.md)，屋顶线与实验账本见 [probe.md](probe.md)，数值纪律见 [lesson.md](lesson.md)，自治准则见 [goal.txt](goal.txt)。

## 🎯 2026-11-19 屋顶线重校与 S1/S2/S3 物理路线

截至提交 `085508a8`（2026-11-19），全自研零 vLLM 整轮图已将 [case-01-coding](../tests/glm52/megakernel_tp8_ep8/case-01-coding) 冷启动正式 2+5 中位 TPOT 由 M1 基线 6.6928 ms 压至 **5.6904 ms**（最低 5.6836 ms，累计降 1.0024 ms / −15.0%，141 轮，轮均接受 7.255 词元，8 卡主网及串行草稿零误差对齐，样本外 `case-02/03/04` 同步刷新至 10.8965 / 6.3214 / 7.8553 ms）。基于最新带周期剖析的 5.7392 ms 硬件账本（整轮 41.3 ms：巨核 35.78 ms + L78 草稿与提交 5.50 ms），必须修正早期两处物理误判：

1. **修正单芯带宽屋顶（原 6.32 GB/芯低估 2.4 倍）**：8 芯跑 14–16 宽树时每轮激活约 600 个路由专家（11.3 GB/芯）加 MLA/共享专家/8K KV 共 **15.0 GB/芯**；按单芯 1.25 TB/s 有效带宽，纯字节搬运下限已达 12 ms。当前路由专家矩阵乘（10.35 ms）已运行在峰值七成四，硬抠路由 GEMM 微调收益已近天花板。
2. **证伪纯后缀匹配、锁定 29.15 ms 低效主战场**：实测 `case-01..04` 真实 1024 输出的纯 prompt/suffix 匹配轮均仅 1.25–1.62 词元（`case-01` 仅 1.43），远低于当前 DSpine L78 的 7.255，停止在 PLD/SuffixDecoding 上投入；真正距屋顶线 5 倍以上、硬件效率低于二成的肥肉集中在注意力（14.17 ms 对比 2.5 ms 字节下限）、非矩阵乘同步与跨卡倾斜等待（9.48 ms）以及 10 步串行 L78 草稿（5.50 ms）。

|阶段|目标TPOT|核心破壁动作|目标单轮与接受长|
|---|---:|---|---|
|S1 收割低效区|4.0–4.6 ms|重写注意力(14.17→6.5 ms)+压同步与非GEMM气泡(9.48→4.0 ms)+双端自适应树(W=4..24)|32 ms / 7.5–8.2|
|S2 破结构双墙|2.7–3.3 ms|专家零倾斜重排(10–12槽镜像或带260B极值同步的TP8切分)+单步并行块草稿/跨轮重叠替10步串行L78|23–26 ms / 8.5–10.0|
|S3 释放整机与收敛|<2.0 ms|打通单台A3满配16芯(TP16单芯搬运减半)+清退70+历史版本宏沉淀四层窄腰|14–16 ms / 8.5+|

四项立刻生效的工程决断：
- **算子重仓转向注意力与同步气泡**：减少对已达带宽七成四的路由矩阵乘微调，集中算力砍注意力 14.17 ms（消除 1.40 ms `MergeKv` 中转、块粒度破离散访存 16 并发墙、尾投影与一段归约核内融合）与 9.48 ms 同步气泡（清理 `OVERLAP_SH_P4` 引发的 `slot19` +0.72 ms 竞争、扩大静态镜像槽位至 10–12 槽或评估 TP8 专家切分彻底归零剩余 3.04 ms 跨卡倾斜）。
- **双端自适应树 + 并行块草稿**：依托已打通的 `W=2,4,6` 零误差路径与 32 宽通信底座，短期落地高熵轮 4–6 宽早停、高置信轮单束延展至 14–24 步的双端自适应树；中期以 DFlash 类单步并行草稿或跨轮草稿验证流水重叠替代 10 步串行 L78（5.50→≤1.5 ms）。
- **明确半机 8 芯与满机 16 芯物理边界**：当前 `TP8+EP8` 仅占单台 A3 一半物理芯数（8/16 die），在当前 8 芯契约下靠 S1+S2 冲击 `<3.0 ms` 物理极限；后续 S3 启用整台 A3 16 芯将单芯搬运直接腰斩，稳态守住 `<2.0 ms`。
- **终结历史条件宏债务**：`build_megakernel.py` 已堆叠逾 70 个 `-DK11xxx_*` 历史宏并屡次诱发静默空操作与跨阶段竞争；即日起每落库一项已验证收益即删除失效旧分支，维持单一主干实现，待破 3 ms 后再面向下一代 950 架构抽象框架层。

## 🧩 四层零妥协极简窄腰

摒弃混合状态对象、框架猴子补丁及一切面向混部分页的妥协参数，全引擎自底向上收敛为四道不可变值对象与 C-ABI 窄腰：

|窄腰层级|组件命名|物理承载与唯一职责|关键设计决断|
|---|---|---|---|
|S1：前缀竞技场|FrozenPrefixArena|从偏移0起步的单块连续HBM缓存池(容量对齐至9280或10240行)：0–78层共79对BF16 MLA latent[512]与RoPE[64]、22层INT8/BF16 Indexer Key[128]、16层FP16 Indexer Scale[1]、末尾未旋转父隐藏态 $\tilde{h}_{\text{main}}[P-1]$ 及首词13534|砍除一切离散block_tables页表寻址寄存器开销，22层Lightning Indexer直接以连续MTE2突发读取扫满全前缀；离线直灌快照，外部P/D仅在[ascend_layout.py](../tilert/pd_vllm/ascend_layout.py)传输侧线速聚拢|
|S2：自持权重池|ModelArena|零框架流式读取ModelSlim量化分片与rot.safetensors，按TP8+EP8切分并经torch_npu.npu_format_cast固化Fractal NZ排布与256 B对齐填充，冻结为只读设备指针表|升级[weights.py](../tilert/models/glm_5_2_ascend/weights.py)：embed_tokens、lm_head与第78层独立shared_head.head按TP8切为19360×6144；0–2层Dense合并gate/up(W8A8)；3–78层每卡驻留TP8共享专家与32个完整EP8路由专家(W4A8高低半字节打包，16组scale_bias加载期预求和)，彻底废除NPUWorker|
|S3：全层Megakernel链|FullChainABI|单份C++17标准布局头文件([contract.h](../tests/glm52/whole_layer/layer_local_chain/contract.h))贯通编译期static_assert断言，用五类融合积木覆盖0–78全层与双输出头，零Python算子派发|补齐[whole_layer_abi.py](../tilert/models/glm_5_2_ascend/whole_layer_abi.py)缺失的Prolog、0–2层Dense与双头积木；层间通过GM软件记分牌实现零硬件标志消耗的跨层静态权重L2预取|
|S4：自治整轮超图|ShowHandsLoop|单张aclmdlRI+SuperKernel整轮静态图 $G_{\text{round}}(W)$ 包揽「元数据自备→草稿树推演→主网验证→条件草稿重算→片上路径裁决与游标推进」，向锁页环单向刷出64 B已确认词元|采纳二次设计胜出方案：将跨轮状态全量推入设备GM，宿主退化为约60行无状态深度2流水投喂器(in_flight=2)，实现0 µs轮间气泡且免除庞大宿主C++状态机|

### 1. S3：0–78 全层五积木与跨层持久流水（Megakernel & Persistent Prefetch）

顺应 dav-2201 物理切分 24 AIC / 48 AIV、无片上 L0C→UB 直通但共享 192 MB 大容量 L2 的微架构特征，摒弃虚耗 13.58 µs 启动税的泛化 MIX 包装，按纯计算段 AIC_ONLY（2.30 µs）/ AIV_ONLY（0.70 µs）与受控 schedmode(1) 调度模式将 79 层收束为五类高内聚积木（工具链按实测墙钟在手写 AscendC、PTO-ISA 与 TileLang-Ascend 间择优固化）：

|积木命名|覆盖层段|核内融合闭环与单边通信原语|微架构极致压榨要点|
|---|---|---|---|
|PrologBlock|主网入口与第78层入口|按设备词元槽查表；按设备游标直刷RoPE cos/sin切片；第78层融合enorm、TP8切分rot、hnorm、eh_proj及aclshmem跨卡归约|零宿主H2D拷贝，直接消费GM游标槽与父隐藏态缓冲区|
|AttentionBlock|0–78层全部注意力|残差RMSNorm+精确BF16舍入+向量Brcb动态INT8量化+分页归零连续INT8/BF16 Lightning Indexer Top-2048选择(共享索引层直读主缓冲，树验证叠加祖先位掩码)+稀疏MLA+o_proj+aclshmem确定性[4,2,1]树规约|静态常量化分桶行数(M∈{1,2,4,8,12,16})消除标量Tiling税；施加(mBlockIdx+blockIdx)%numBlocks错位分核消除512 B同地址窗口串行化；核内DCCI直读GM长度([attention_indexed_native.h](../tests/glm52/whole_layer/attention_indexed_native.h#L76-L81))|
|DenseFfnBlock|0–2层稠密前馈|残差RMSNorm+动态量化+W8A8合并Gate/Up矩阵乘+SwiGLU+Fixpipe重量化+W8A8 Down投影+aclshmem核内确定性树规约|彻底拔除前3层向vllm_ascend回退的历史断层|
|MoeFfnBlock|3–78层混合专家|残差RMSNorm+Top-8 Sigmoid路由(缩放2.5)+TP8共享专家并发计算+EP8分发+W4A8 N256/K256列块矩阵乘+aclshmem核内确定性归并([tp_push_kernel.cpp](../tests/glm52/shmem/tp_push_kernel.cpp))|冷路由专家权重施加SetL2CacheHint(CACHE_MODE_DISABLE)防污染；前层尾部归并期间空闲MTE2通道按GM软件记分牌(SoftSyncAll)预取下层共享专家与注意力静态权重入L2|
|HeadArgmaxBlock|主网头与第78层自有头|Final RMSNorm+本地19360维词表分片FP32累加投影+片上局部最大值与全局词元ID换算+aclshmem 64 B(val,idx)八卡归约|主网严用lm_head，第78层严用shared_head.head；设备槽直出贪心词元ID，彻底废除154880维全词表Logits跨卡聚合|

### 2. S4：设备自治整轮超图与双缓冲投喂（ShowHandsLoop）

针对 [mtp_round.py](../tilert/models/glm_5_2_ascend/mtp_round.py) 与 [cann_mtp_graph.py](../tilert/models/glm_5_2_ascend/cann_mtp_graph.py) 的二次正交设计表明：既然核内已通过 DCCI 直读 GM 元数据，则固定树拓扑下的整轮投机闭环不存在任何宿主数据分支。因此废除逐轮三次图下发与重型宿主状态机，直接构建对齐上游 [end2end.py](../tilert/models/glm_5_2_rocm/end2end.py) mtp_n 语义的设备自治超图：
1. 单张整轮图闭合：将整轮捕获为单一 aclmdlRI 静态图 $G_{\text{round}}(W)$ 并开启 SuperKernel early_start。图首置一颗极轻 AIV 核 PrologPrepareMeta，读取设备 GM 游标 $P$ 与完成标志 done，原地生成全层 positions、连续写槽与 SFA/Indexer 长度描述符（当 done = 1 时将有效查询行数 q_len 置 0，后续全层积木读到 q_len = 0 立即微秒级空退，且因 8 卡贪心输出逐位相同必在同一轮同步空退，保持 aclshmem 窗口代次严格平衡）；图中串接共享种子索引草稿推演、0–77 层主网宽验证及第 78 层目标条件重算（从本轮起点覆写临时草稿行）；图尾置一颗单核 AIV 核 ResolveAndAdvance，在设备侧执行无递归最长接受路径扫描（对应 [tree_resolve.py](../tilert/models/glm_5_2_ascend/tree_resolve.py)），核对 EOS 集合与 1024 词元剩余预算，原地更新 GM 游标 $P \leftarrow P + m$、草稿有效游标 $P_{\text{mtp}} \leftarrow P + m - 1$，将接受末节点的条件草稿 $(token, recycle\_hidden, indices)$ 拷入下轮固定种子槽，并向 aclrtMallocHost 锁页环槽单次写出 64 B 回执 $(seq, m, done, tokens[0..m-1])$。
2. 深度 2 流水消隐宿主：启动解码时宿主向 RTS 硬件流预入队 2 个 $G_{\text{round}}(W)$。第 $t$ 轮尾核写毕 GM 新游标的瞬间，RTS 硬件调度器以 0 µs 轮间空洞直接拉起已在队列中的第 $t+1$ 轮。在第 $t+1$ 轮长达 15 ms 以上的设备执行窗口内，约 60 行无状态宿主投喂器从容读取锁页环第 $t$ 轮回执并在 done = 0 时补入队第 $t+2$ 轮；命中终止条件时至多仅空退 1 个在途图，零 KV 污染且完全免疫任何宿主线程抖动。唯有后续实验证实逐轮跨分桶变宽收益超过约 8 µs 轮间往返税时，才按需将投喂深度切回 1 并挂载轻量 C++ 选桶器。

## ⚖️ 硬件物理铁律与数值真值公理

一切 Megakernel 融合、预取与投机树改造必须无条件服从以下五条硬件与数值公理，任一违背直接否决（历史反例见 [lesson.md](lesson.md) 与 [probe.md](probe.md)）：
1. 归一化与动态量化逐位等价：RMSNorm 输出必须保留精确 BF16 舍入后再执行动态 INT8 量化；向量量化统一采用标准 Brcb 广播乘法与 int16 CAST_RINT 路径，严禁混入标量 Muls 加 int32 中转造成 1-ULP 翻转及后续专家路由级联发散。
2. Fixpipe 硬件截断一致：全部 W8A8 与 W4A8 矩阵乘的反量化计算及 [reference.py](../tilert/models/glm_5_2_ascend/reference.py) 真值参考严格复现 A3 Fixpipe 清除 FP32 scale 低 13 位的硬件算术行为。
3. 跨卡规约树括号锁死：8 卡 aclshmem 单边规约严格固定确定性 [4,2,1] FP32 加法结合顺序，确保任意轮数下 8 个 rank 的隐藏态、专家路由、稀疏 TopK 同分决胜序（固定查询分组分核）与贪心词元 100% 逐比特一致。
4. 硬件同步计数器防饱和：严禁跨层串联 dav-2201 仅有 4 位计数上限（15 次不可清零）且阻塞全流水的 CrossCoreSetFlag / CrossCoreWaitFlag；核内 AIC↔AIV 交接必经 L2/GM 并配齐 DataSyncBarrier<DDR>，跨卡与跨层同步统一走绕过 DCache 的 GM 原子代次（epoch）。
5. 草稿状态与输出头纯净性：自滚草稿链写入的第 78 层临时行绝不转正为已提交 KV，所有保留行必须取自主网真实隐藏态驱动的目标条件重算；第 78 层草稿严禁混用主网 lm_head 权重。

## 🗂️ 工作区洁癖、命名品味与代码库大收敛

对比 TileRT 上游根目录（[project.py](../scripts/project.py) ROOT_ENTRIES）与兄弟实现 [glm_5_2_rocm](../tilert/models/glm_5_2_rocm) 的极简 6 文件结构，当前仓库必须执行激进外科手术式清理，根除根目录杂物、扁平文件膨胀与构建垃圾混排：

|治理域|当前积弊诊断|终态目录与命名规范|硬性护栏与清理动作|
|---|---|---|---|
|仓库根目录 /|散落临时会话大文件与未跟踪杂项tmp目录|严格锁定上游TileRT根目录白名单加docs、scripts、tests，零多余文件或目录|启用scripts/hooks/pre-commit钩子阻断任何白名单外根目录条目及二进制产物入库|
|模型主包 tilert/models/glm_5_2_ascend/|50个扁平Python文件混杂cann前缀寄生补丁、旧TP16原型与过渡适配器|对标[glm_5_2_rocm](../tilert/models/glm_5_2_rocm)收敛为6个核心模块加1个算子包：config.py、weights.py、abi.py、spec.py、end2end.py(Glm52AscendShowHands)、reference.py及ops/|G2合流后一次性删除cann_model.py、cann_graph.py、cann_mtp.py、cann_mtp_graph.py、cann_mtp_projection.py、cann_head.py、cann_attention.py、whole_layer.py、moe_combine.py、routed_expert.py、routed_shared.py、mla_tail.py、attention_native.py及全部TP16遗留包装(逾35个文件)|
|算子与原生层 ops/|C++源码深埋于tests/glm52/whole_layer下60余个build前缀副本目录中|正式AscendC/TileLang/PTO源码升格至tilert/models/glm_5_2_ascend/ops/{prolog,attention,dense_ffn,moe_ffn,head_argmax,round_ctrl,shmem}|遵循零注释自解释代码与短命名品味；任何.o/.so/.elf编译缓存与快照一律重定向至tests/.artifacts/隔离目录|
|测试与基准 tests/|tests/glm52扁平堆积150余个一次性脚本与内嵌构建产物|黄金性能与真值锚点唯一锁定[case-01-coding](../tests/glm52/megakernel_tp8_ep8/case-01-coding)；单测按窄腰收敛为test_weights.py、test_ops.py、test_spec.py与bench_case01.py|清退tests/glm52下全部build前缀构建目录与失效一次性脚本，历史实验收据归档于tests/.artifacts/runs/并由manifest校验SHA256|
|设计文档 docs/|部分旧段落仍残留TP16与多路路线包袱|维持readme.md、plan.md、本文、speculative-decoding.md、probe.md、lesson.md、eggs.md职责正交、精炼不过时|全目录强制遵守.markdownlint.json的MD060 compact表格风格，零死链、零废话，实验结论必绑定原始收据路径|

## 🏁 见缝插针实验推进与可证伪门禁

依托 kbenv（/home/shanshan/dotfiles/harness/skills/vibe-iris/scripts/experiment/kbenv）按物理 die 空闲条带（· 标记与 npu-smi 无进程核验，严禁退回 A2 硬件或干扰共租户容器）高机动推进：单积木与微基准直接通过 ASCEND_RT_VISIBLE_DEVICES 抢占 1–4 颗空闲 A3 die 即刻验证，整模配对验收锁定任意具备 8 颗空闲 die 的 A3 节点执行隔离编译缓存、关闭 Profiler、3 预热加 5 交替配对的同机 A/B 测试：

|演进门禁|所需A3资源|核心交付动作|硬性真值与性能判据|
|---|---|---|---|
|G1：五积木与单边融合闭环|1–4颗空闲die(单算子/局部TP)，随后8 die对账|在ops/下补齐PrologBlock、0–2层DenseFfnBlock、第78层投影、双HeadArgmaxBlock及首尾游标裁决核；打通aclshmem核内AR+Norm+Quant原位融合|大于192 MB破L2工作集下确认组件物理净收益；M1/M8/M12/M16多宽度下79层隐藏态、路由、KV与双头贪心词元逐位对齐冻结参考，零调用vllm_ascend算子|
|G2：ShowHands超图脱钩与仓库大扫除|8颗空闲die(TP8+EP8)|上线ModelArena自持加载与单张整轮超图 $G_{\text{round}}$ 深度2预投喂，跑通[case-01-coding](../tests/glm52/megakernel_tp8_ep8/case-01-coding)离线全量1024词元生成；彻底删除35+个cann前缀寄生文件并收敛目录|进程内断言sys.modules零含vllm与vllm_ascend；1024词元精确命中黄金SHA256(2f751372...)且8卡主网与串行草稿状态全过；同机配对稳态TPOT压破6.0 ms|
|G3：跨层预取与共享索引树破3 ms|8颗空闲die(TP8+EP8)|开启SoftSyncAll跨层L2静态权重预取与冷专家CACHE_MODE_DISABLE；将共享种子索引下的Top-2/Top-4静态草稿树(W12/W16)合入 $G_{\text{round}}$|遵循Lakatos退化门禁(连续两次无新物理预测修订即截断切换机制格)；[case-01-coding](../tests/glm52/megakernel_tp8_ep8/case-01-coding)五组交替测量稳态中位TPOT < 3.0 ms，黄金SHA256与8卡全状态位级一致|
