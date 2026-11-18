# GLM-5.2 TP8+EP8 投机解码

目标锁定单台 Ascend A3（Ascend910_9392 或 9382，dav-2201 架构，8 物理 die）、官方冻结权重 GLM-5.2 W4A8C8（TP8+EP8）、batch=1、输入 8192 词元、输出 1024 词元，唯一正式验收基准为 [case-01-coding](../tests/glm52/megakernel_tp8_ep8/case-01-coding)（首词 ID 13534 来自预填充，稳态 TPOT 分母为随后生成的 1023 词元，主网有效缓存终态长度 9215）。硬门禁要求稳态中位 TPOT < 3.0 ms，1024 词元输出命中黄金 SHA256（2f751372fc82cf290c8e35d3e7cc3cad0296fde4d8d188fba102ac34f82c2962），且 8 卡主网有效 KV 与独立串行目标条件草稿 KV 逐位对齐。终态架构总纲见 [standalone-decode.md](standalone-decode.md)。

当前最佳已验收实测稳态中位 TPOT 为 7.5988 ms（最快 7.4716 ms，P90 7.7494 ms，尚未达标 3.0 ms）：采用模型自带第 78 层单层 MTP 自回归锥形树 W16（十步层宽调度 4-4-4-4-2-2-2-2-1-1，共 149 轮，单轮平均提交 6.866 词元；W32 树达 7.992 词元）配合 75 层持久 Megakernel K11030，单轮墙钟约 52.45 ms（0–77 层主网验证 44.31 ms，含 75 层持久链 41.40 ms 与 0–2 层及输出头 2.9 ms；第 78 层 10 步树起草 6.54 ms；目标条件重算与种子提交 1.42 ms；跨卡同步 0.18 ms）。后续 K110383（NZ 注意力尾加 GU_B12 加 DN_KB4）在同构 W16 验证下将 75 层链压至 41.399 ms、单轮验证墙钟压至 43.172 ms，8 卡 0-ULP 通过。

## 📊 里程碑实测台账

正式计时包含完整解码循环及设备同步，排除权重加载、预填充、静态图捕获（含补齐预热后的首次输出头图捕获，见 [计时修正](../tests/.artifacts/runs/glm52-width-20261111/head-warmup/README.md)）、诊断插桩与计时后状态摘要，全程关闭 Profiler；表中时延为 rank0 墙钟或八卡最大墙钟，独立组件收益不可线性相加为端到端 TPOT 收益。源码与运行入口见 [TP8 MTP 实验](../tests/experiments/glm52-tp8-mtp/README.md)，黄金状态收据见 [配对测量](../tests/.artifacts/runs/glm52-tp8-mtp-20261111/capacity/mtp-paired/summary.json) · [主网状态对照](../tests/.artifacts/runs/glm52-tp8-mtp-20261111/capacity/baseline/frozen-reference-check.json) · [八卡状态结论](../tests/.artifacts/runs/glm52-width-20261111/state-summary.json) · [恢复基线](../tests/.artifacts/runs/glm52-width-20261111/recovery/full/README.md)。

|方案|样本|平均提交词数|TPOT(ms)|验证结果|
|---|---|---|---|---|
|Recency lookup|单次|1.637(625轮)|22.61|1024词元及8卡主网有效KV匹配冻结对照|
|MTP W4 eager|五次，与graph交替|3.552(288轮)|中位16.20；范围15.88–16.33|十次运行输出、主网KV均匹配；逐卡草稿KV在两方案间逐位一致|
|MTP W4 graph|五次，与eager交替|3.552(288轮)|中位11.16；范围11.14–11.23|同上；另有19项eager/graph字节对照在八卡通过|
|MTP W8 eager，旧TopK分组|单次诊断|5.715|15.52(不纳入有效结果)|第53个输出词元开始分歧，拒绝|
|MTP W8 eager，固定TopK分组|单次验收，7.150.13.62|4.990(205轮)|18.15|1024词元及八卡主网KV匹配黄金；草稿KV匹配独立串行参考|
|MTP W8 graph，固定TopK分组|五次，与W4 graph交替|4.990(205轮)|中位9.69；范围9.67–9.76|80份逐卡运行记录的词元、主网及草稿状态全部通过|
|MTP W8 graph，动态草稿词表(每卡4096候选)|五次，与完整词表交替|4.966(206轮)|稳态中位9.49(对照9.76，-2.72%)|80份逐卡黄金词元、主网及独立串行草稿状态全部通过|
|MTP W8 graph，动态词表加GU N256列块|五次，与原GU交替|4.966(206轮)|稳态中位9.42(对照9.47，-0.569%)|80份逐卡黄金词元、主网及独立串行草稿状态全部通过|
|MTP W8 graph，GU加动态词表加物理主图12行|五次，与同配置32行交替|4.966(206轮)|稳态中位9.178(对照9.333，-1.663%)|80份逐卡黄金词元、主网及独立串行草稿状态全部通过|
|MTP W8 graph，GU加12行主图加提示内校准加完整草稿词表|同批三臂，每臂五次交替|5.193(197轮)|稳态中位9.101(对照9.170，-0.754%；7.156.97.214重验八卡最大耗时中位9.2618)|120份逐卡黄金词元、主网、独立串行草稿状态及图检查全部通过|
|原生第78层MTP自回归锥形树W16(4-4-4-4-2-2-2-2-1-1)加75层持久Megakernel K11030|三次预热加五次正式测量|6.866(149轮)|稳态中位7.5988；最快7.4716；P90 7.7494|黄金SHA256与8卡主网、独立串行草稿KV逐位通过(K110383同构验证墙钟进一步压至43.172 ms)|

## 🧭 投机范式谱系与 DSpine 引入收益评估

外部文献中 DFlash、DSpark、DFlash2、DSpine 四项同属「独立训练多层并行块起草（Block-Diffusion / Parallel Block Drafting）」谱系，前提是额外训练多层非因果或半自回归草稿网络，在单次前向中同时生成整块未来词元；当前仓库走的是正交的「原生单层 MTP 自回归树起草（EAGLE-2 范式，零额外训练）」路线，直接复用官方检查点第 78 层 MoE 权重与自有独立头，沿树深 10 步串行自滚、同层多分支并行扩展，经 [tree_attention.py](../tilert/models/glm_5_2_ascend/tree_attention.py)、[tree_mask.py](../tilert/models/glm_5_2_ascend/tree_mask.py) 与 [tree_resolve.py](../tilert/models/glm_5_2_ascend/tree_resolve.py) 完成单次主网祖先掩码树验证及目标条件草稿 KV 回写（注：部分早期实验记录将 4-4-4-4-2-2-2-2-1-1 锥形束宽调度简记为 DSpine 锥形树，实际执行不含独立训练的多层并行块网络或门控邻接注入层）。

|方案|草稿权重来源|起草时间结构|块内依赖与防后缀退化机制|验证形态|
|---|---|---|---|---|
|DFlash(2602.06036)|独立训练的多层非因果块扩散模型|单次前向并行预测整块掩码位|注入主网多层隐藏态上下文，各位置独立取极大|单链块验证|
|DSpark(2607.05147)|独立训练的半自回归块草稿模型|并行块主干加轻量马尔可夫序列头|低秩序列头显式建模块内相邻词元转移|按存活置信度与负载动态伸缩验证长度|
|DFlash2(Inco AI)|独立训练的块扩散模型加选择器头|单次前向并行产出各位置候选格|局部卷积加轻量成对选择器搜索连贯单路径|单链块验证|
|DSpine(2609.36173)|独立训练的5层并行草稿模型|全位置并行前向，沿5层深度逐步条件化|每层门控邻接因果注入加输出嵌入PCA白化逐层余弦监督加16×16末次写入精化转移缓存|单链块验证|
|当前方案(TileRT-Ascend)|官方检查点第78层单层MTP(9.7B MoE)与独立shared_head.head(零训练)|沿树深10步自回归串行，同层按4-4-4-4-2-2-2-2-1-1束宽并行|共享种子Top-2048稀疏索引；提示内无监督rank64残差映射抑制自滚漂移|16词元祖先掩码树单次主网Megakernel验证(16中6.866)|

锚定端到端时延方程 $\text{TPOT} = (T_{\text{verify}} + T_{\text{draft}} + T_{\text{cond}} + T_{\text{gap}}) / \mathbb{E}[\text{commit}]$，在 GLM-5.2（745B 总参量，检查点 377.08 GiB，8 die 下每 die 权重占约 47.1 GiB）与 A3 硬件约束（CANN 融合解码注意力单轮硬上限 16 词元，见 [speculative-acceptance.md](reference/speculative-acceptance.md)；dav-2201 无片上 L0C→UB 直通）下，引入 DSpine 的预期收益严格取决于权重门禁与参数初始化方式：

|评估情景|起草与条件重算开销 $T_{\text{draft}}+T_{\text{cond}}$|期望接受词数 $\mathbb{E}[\text{commit}]$|当前主网(44.31 ms)预期TPOT|终态主网(13.0 ms)预期TPOT|净收益判定与物理依据|
|---|---|---|---|---|---|
|基线：原生第78层W16锥形树|当前7.96 ms(6.54+1.42)；终态3.5 ms|当前6.866；校准目标7.2|7.60 ms(实测)|2.29–2.40 ms|零训练基准；严格满足冻结权重与第78层串行KV位级对齐契约|
|情景A：从零外挂标准5层稠密DSpine(~1.2B)替换第78层|省去条件重算，5层块并行约1.0–1.5 ms(受HBM/L2搬运限制)|约5.6–6.0(单链退化)|7.88 ms(按5.8计，倒挂+3.7%)|2.41 ms(按5.8计，无净收益)|负收益/无收益：DSpine原论文贪心代码实测仅5.71(HumanEval)/6.14(MBPP，主从比8:1–25:1)，600:1主从悬殊下1.2B稠密单链不敌9.7B预训练MoE树；另需约32卡天离线蒸馏、4个新算子，且破坏官方权重与第78层KV验收契约|
|情景B：解锁外部草稿权重，复用第78层9.7B MoE做EAGLE-3多步自滚微调或嫁接轻量DSpine邻接门控桥|自滚微调维持3.5 ms(零新算子)；嫁接3步块并行压至约1.8 ms|7.5–8.3(消除多步漂移并保住MoE容量与树宽)|6.15–6.70 ms|1.90–2.06 ms|显著正收益(终态净降15%–25%)：唯有保留第78层万亿词元预训练MoE底座，消除自滚暴露偏差或压缩串行步数才能同时做小分子、做大分母；受限于冻结权重门禁不可用于标准榜|
|情景C：严守冻结权重契约，仅零训练迁移DSpark样本外置信度/存活率宽度调度|裁剪低存活分支压至2.8–3.3 ms|7.0–7.3|7.35–7.50 ms|2.18–2.30 ms|正收益且合规：在[case-02-review](../tests/glm52/megakernel_tp8_ep8/case-02-review)、[case-03-refactor](../tests/glm52/megakernel_tp8_ep8/case-03-refactor)、[case-04-math](../tests/glm52/megakernel_tp8_ep8/case-04-math)样本外标定锥形束宽与共享种子Top-512/1024稀疏前缀，抑制无效分支冷专家激活|

## 🧱 原生单层 MTP 约 7 词接受上限的五重物理证据

冻结官方权重前提下，原生第 78 层单层 MTP 的单轮期望接受词数物理封顶于 7 词附近，破 3.0 ms 必须依靠持久 Megakernel 将整轮耗时硬压入 20 ms 以内：
1. 单步马尔可夫覆盖率上限：源码与检查点审计证实第 78 层通过独立的 shared_head.head 计算 BF16 logits，其 8 个 TP 分片权重与主网 lm_head 全异（早期共享头假设已撤回，9382 独立头重放逐位复现 9392 的 36 组完整 logits，[输出头契约](../tests/.artifacts/runs/glm52-width-20261111/mtp-head/contract.json)）。在提示内 512 个位置上，真实第 78 层首步第一候选命中 348 项（67.97%），前两候选并集覆盖 406 项（79.30%，[候选覆盖](../tests/.artifacts/runs/glm52-width-20261111/mtp-tree/README.md)）；按单链步接受率 $a \approx 0.84\text{–}0.86$ 计，几何级数渐近上界 $1/(1-a)$ 锁死于 6.25–7.14（线性 W8 为 5.193，锥形树 W16 达 6.866）。
2. 多步自滚暴露偏差与零训练校准触顶：第 78 层仅受单步真值隐藏态预训练，解码多步自滚时留出集隐藏态 MSE 在 7 步内从 0.518 单调漂移至 0.901（[递推采集与筛选](../tests/.artifacts/runs/glm52-width-20261111/mtp-rollout/README.md)）。提示内前 1536 对样本拟合、后 512 对选强的无监督 rank64 残差映射（仅修正未提交草稿链父状态，不读生成区）将完整词表 W8 从 205 轮压至 197 轮、TPOT 降至 9.101 ms（校准加动态词表为 9.106 ms，对照 9.170 ms，120 份状态通过，[校准证据](../tests/.artifacts/runs/glm52-width-20261111/mtp-recycle/README.md)）。此后五路零改主权重的校准实验全部触顶被拒：七步递推重拟合固定 rank64 基最多仅降总体 MSE 1.087%（未达 2% 门禁）；监督式 primal SVD 低秩残差回归使留出 MSE 由 0.760610 恶化至 0.800252（[监督式低秩筛选](../tests/.artifacts/runs/glm52-width-20261111/mtp-regression/README.md)）；主网 head-map 使 512 位置命中由 348 退化至 339；双候选一步前瞻重评分复现 4096 个逐卡 seed 与 288 份分支状态后命中由 348 退化至 343（改对 15、改错 20，[前瞻筛选](../tests/.artifacts/runs/glm52-width-20261111/mtp-selection-partial/README.md)）；1280 样本子空间上训练 4160 参数的自有头分类校准在内部 256 项上由 170 退化至 164/166/162/149/144，独立读回退回恒等映射（[分类校准](../tests/.artifacts/runs/glm52-width-20261111/mtp-head-calibration/README.md)）。
3. 共享种子 Top-2048 稀疏索引跨步失配：M2/M4 并行草稿在逻辑 RoPE 与物理 KV 槽分离、种子索引不读临时分支行约束下通过 288 个逐卡批次与 M1 逐字节等价门禁（[并行候选门禁](../tests/.artifacts/runs/glm52-width-20261111/mtp-candidates/README.md)），但树深越过 6–8 步后复用首步种子的 Top-2048 稀疏历史集合相对当前查询发生偏移，制约深层分支命中率。
4. 256 专家 MoE 路由并集膨胀吞噬宽树收益：主网 W8 独立诊断在 75 层累计触发 4800 个专家命中，八卡分别调用 403–453 个专家块（[原始计数](../tests/.artifacts/runs/glm52-width-20261111/native-profile-ffn/counts/cycles-rank-0.json)）。验证树宽从 W16（6.866）扩至 W32（7.992）接受词数仅增 16.4%，但多分支在 256 专家上激活近似独立的 Top-8 子集，冷专家 HBM 读取线性膨胀导致单轮验证墙钟斜率陡增（早期图执行 W8/W12/W16 单次 TPOT 分别为 9.74/9.82/11.14 ms，[宽度比较](../tests/.artifacts/runs/glm52-width-20261111/mtp-wide-draft/graph-widths-summary.json)；反之将主图物理行由 32 缩至 12 在逻辑 W8 下使五组中位由 9.634 降至 9.491 ms，叠加 GU 与动态词表由 9.333 降至 9.178 ms，80 份状态全过，[缩窄主图](../tests/.artifacts/runs/glm52-width-20261111/main-graph12/bf16-chain/README.md) · [组合证据](../tests/.artifacts/runs/glm52-width-20261111/main-graph12/gu-vocab/README.md)）。
5. 无模型上下文检索无法互补增益：[case-01-coding](../tests/glm52/megakernel_tp8_ep8/case-01-coding) 唯一 4-gram 比例高达 0.7698，Recency lookup 平均仅提交 1.637 词；同前缀 46 轮严格配对比较中，因果后缀匹配累计提交 184 词，净负于原生 MTP 的 246 词（[混合检索筛选](../tests/.artifacts/runs/glm52-width-20261111/hybrid-lookup/screen.json)）。

## 🔧 确定性数值修复与算子筛选收据

### 1. 执行路径与多宽度位级等价修复

1. 原生接口与单边规约修复：[原生 MTP](../tilert/models/glm_5_2_ascend/cann_mtp.py) 补齐 TP8 每卡 8 查询头及共享输入初始化字段。[单边规约](../tests/glm52/shmem/tp_push_kernel.cpp) 旧 one_hop 分支仅检查有效元素数而漏检分片容量，致 M1/M4 六个本地 worker 仅覆写前 1/8 输出并引发草稿 M4 NaN；修复后 8 卡 M1/M4/M8/M32 全元素精确对齐确定性规约（[修复前](../tests/.artifacts/runs/glm52-tp8-mtp-20261111/bf16-before.log) · [修复后](../tests/.artifacts/runs/glm52-tp8-mtp-20261111/bf16-after.log)）。[共享索引快捷路径](../tilert/models/glm_5_2_ascend/moe_combine.py) 原省略实体 top-k 缓冲写回导致选中第 2 行仍读回缓存第 0 行，现让草稿成对走原始 get/update 并保留主网快捷路径。
2. 主网 W8/W16/W32 索引器同分决胜序对齐：W8 首差定位于位置 8221、layer25（layer24 输入一致，rank5 注意力先变并经跨卡求和扩散，[逐字段首差](../tests/.artifacts/runs/glm52-width-20261111/prefix-diff.json) · [层内对照](../tests/.artifacts/runs/glm52-width-20261111/layer24-eager/comparison.json)），根因为 layer22 八卡相等 FP32 评分在不同分核下换序；固定查询分组分核后 W8 全程回到 W4，同机五组图配对由 W4 的 11.08 ms 降至 W8 的 9.69 ms（-12.5%，eager 对照见 [固定分组完整验收](../tests/.artifacts/runs/glm52-width-20261111/mtp-grouped-eager/state-comparison.json)，[配对回执](../tests/.artifacts/runs/glm52-width-20261111/mtp-grouped-paired/summary.json) · [评分与分区](../tests/.artifacts/runs/glm52-width-20261111/layer22-indexer/score-comparison.json) · [顺序干预](../tests/.artifacts/runs/glm52-width-20261111/layer24-topk/rank-0.json) · [串行对照](../tests/.artifacts/runs/glm52-width-20261111/layer22-baseline/serial-comparison.json)）。W16 首差定位于位置 8410、layer2 BF16 索引器同分换序（否定原 C>8 FFN 布局错误假设，[前提反证](../tests/.artifacts/runs/glm52-width-20261111/checks/premise-check.json)），固定四组分核后 teacher forcing 下 W4/W8/W16/W32 全部逐位通过（[分数与分区](../tests/.artifacts/runs/glm52-width-20261111/layer2-indexer/score-comparison.json) · [完整检查](../tests/.artifacts/runs/glm52-width-20261111/teacher-dense-widths/rank-0.json)）。
3. 第 78 层宽草稿本地投影分块对齐：修复主网后真实 W16 MTP（174 轮）曾出现草稿 KV 偏离独立串行 M1 参考（[逐卡状态](../tests/.artifacts/runs/glm52-width-20261111/mtp-dense-widths/state-comparison.json)），定位出本地 BF16 GEMM 在 M16 下与 M1/M4/M8 存在形状相关舍入差异（首差分别在首轮位置 8192 的 rot、eh_proj 及第 70 轮位置 8593 的索引投影，FP64 误差解释见 [数值分析](../tests/.artifacts/runs/glm52-width-20261111/mtp-wide-draft/projections/fp64.json)）；解码期将这三处投影按最多 8 行分块计算后，完整 W8/W16 黄金词元、8 卡主网与串行草稿缓存全部逐位通过（[完整验收](../tests/.artifacts/runs/glm52-width-20261111/mtp-wide-draft/full-canonical/summary.json)）。

### 2. 草稿词表、矩阵分块与通信重叠正负对照

1. 草稿词表筛选：动态草稿词表由主网前 8 验证行本地 logits 每卡选 4096 候选及权重（注：含切换主网 head 权重因素），补齐预热后五组配对全部快于对照（中位 9.49 对 9.76 ms，206 对 205 轮，[配对回执](../tests/.artifacts/runs/glm52-width-20261111/mtp-adaptive-vocab/paired-warm/summary.json)）；固定提示词表退化至 13.76 ms（[固定词表](../tests/.artifacts/runs/glm52-width-20261111/mtp-shortlist/README.md)），FP32 局部贪心头无一致配对收益（[贪心头](../tests/.artifacts/runs/glm52-width-20261111/mtp-greedy-head/README.md)），仅取有效前缀评分词表使轮数增至 212、TPOT 退至 9.78 ms（[有效前缀词表](../tests/.artifacts/runs/glm52-width-20261111/mtp-valid-vocab/device-result.json)），后三者均拒绝。
2. GU 与 SFA 算子候选：GU 改用 N256/K256 列块与 SDK normal 缓存保留精确整数点积与规约序，72 项边界与双词表五组配对均胜出（中位 9.42 ms，[构建与运行证据](../tests/.artifacts/runs/glm52-width-20261111/gu-tiles/README.md)）。SFA helper 完成同步拆两段在补齐预热后五组中位 9.469 对 9.486 ms，后三组近零差，保留原版（[稳态组合回执](../tests/.artifacts/runs/glm52-width-20261111/mtp-sfa/paired-warm/summary.json)）；SFA PV 邻接两次 BF16 ND2NZ 合并为单次 UINT32 搬运独立配对由 9.78 降至 9.67 ms（-1.134%），但与 GU、动态词表组合为 9.46 对 9.48 ms 且一组略慢，暂不替换（[加载合并证据](../tests/.artifacts/runs/glm52-width-20261111/sfa-mm2-load/README.md) · [组合配对](../tests/.artifacts/runs/glm52-width-20261111/gu-sfa/README.md)）。
3. 专家放置离线评估：仅用提示末尾 256 词（192 训练 / 64 验证）做静态共激活分区使验证集逐层最大专家块和由 4855 降至 4797（-1.19%），每卡每层两份固定副本（占 2.65 GiB）降至 4555（-6.18%），归档未搬动设备权重（[分区证据](../tests/.artifacts/runs/glm52-width-20261111/route-balance/placement/README.md) · [副本证据](../tests/.artifacts/runs/glm52-width-20261111/route-balance/replicas/README.md)）。
4. 已证伪归档的 FFN 归并/重叠候选（80–144 份状态全过但墙钟退化，均不采用）：按列分工归约再广播 BF16（9.79→9.91 ms，+1.23%，[配对回执](../tests/.artifacts/runs/glm52-width-20261111/slot-owner/paired-sync/summary.json) · [实验记录](../tests/.artifacts/runs/glm52-width-20261111/slot-owner/README.md)）；GU 专家列块 24 核连续均分（9.77→9.82 ms，[配对回执](../tests/.artifacts/runs/glm52-width-20261111/balanced-gate/paired/summary.json)）；75 层×48 AIV 逐核轮询诊断后的 8 槽未就绪 owner 去重（非插桩 9.152→9.233 ms，[去重对照](../tests/.artifacts/runs/glm52-width-20261111/native-profile-ffn/poll-dedup/README.md) · [原始记录与边界](../tests/.artifacts/runs/glm52-width-20261111/native-profile-ffn/reduce/README.md)）；共享等待期间预搬已就绪远端槽（9.204585→9.224526 ms，+0.217%；172800 条层/AIV 被动计数中 28.736% 发生预搬，信号探针 52928 条见证含 37872 条旧 SH 与 15056 条当前 SH 证实跨 owner 到达偏斜，但接收端预搬无净收益，[预搬证据](../tests/.artifacts/runs/glm52-width-20261111/native-profile-ffn/overlap/README.md) · [被动计数](../tests/.artifacts/runs/glm52-width-20261111/native-profile-ffn/overlap/counts/README.md) · [信号见证](../tests/.artifacts/runs/glm52-width-20261111/native-profile-ffn/overlap/witness/README.md)）；生产端六专家波次分批 down 与输出重叠（覆盖 639/1800 个组合，三前缀整链 event 反增 1.57%–2.63%，[分批流水筛选](../tests/.artifacts/runs/glm52-width-20261111/native-profile-ffn/down-pipeline/README.md)）。
