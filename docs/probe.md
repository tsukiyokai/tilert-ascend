# 实验记录

本页记录截至 2026-11-01 做过的实验、成立的结论和核查入口。主工作负载是 A3 单机 16 die（TP16 或 TP8+EP8）、GLM-5.2-W4A8C8、单请求、8K 前缀；单 die、局部层、合成权重及其他宽度另行标注。项目定位见 [readme](readme.md)，后续工作见 [plan](plan.md)，实验方法见 [lesson](lesson.md)。

按问题查找：

- [整请求速度有哪些记录与更正？](#%E6%95%B4%E8%AF%B7%E6%B1%82%E9%80%9F%E5%BA%A6%E4%B8%8E%E8%BE%93%E5%87%BA)
- [数值一致性和在线 P/D 做到哪一步？](#%E6%95%B0%E5%80%BC%E4%B8%80%E8%87%B4%E6%80%A7%E4%B8%8E%E5%9C%A8%E7%BA%BF-pd)
- [原生融合验证了多少个 rank？](#%E5%8E%9F%E7%94%9F%E8%9E%8D%E5%90%88%E4%B8%8E%E8%B7%A8-rank-%E8%A7%84%E7%BA%A6)
- [整轮时间花在哪里？](#%E6%95%B4%E8%BD%AE%E8%80%97%E6%97%B6%E4%B8%8E%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90)
- [Attention、缓存和宽验证有哪些正负结果？](#attention%E7%BC%93%E5%AD%98%E4%B8%8E%E5%AE%BD%E9%AA%8C%E8%AF%81)
- [MoE、量化和通信有哪些实验结果？](#moe%E9%87%8F%E5%8C%96%E4%B8%8E%E9%80%9A%E4%BF%A1%E7%BB%84%E4%BB%B6)
- [起草器怎样影响接受数和成本？](#%E8%B5%B7%E8%8D%89%E5%99%A8%E7%9A%84%E6%8E%A5%E5%8F%97%E6%95%B0%E4%B8%8E%E6%88%90%E6%9C%AC)
- [更早的组件实验在哪里？](#%E6%97%A9%E6%9C%9F%E5%AE%9E%E9%AA%8C%E7%B4%A2%E5%BC%95)
- [哪些共同验收尚待完成？](#%E5%BE%85%E5%AE%8C%E6%88%90%E7%9A%84%E5%85%B1%E5%90%8C%E9%AA%8C%E8%AF%81)

各项结论以列出的输入、算术、编译产物和计时窗口为准；多输入、真实 TP16 与在线稳态等共同缺口集中在末节。`evidence/` 保留历史推导，本页注明的混淆、更正和撤回优先于其中的早期结论。脚本、接口检查和源码分析单独列示，设备结果以实际运行记录为据。

## MAP-Elites 正交机制档案与屋顶线账本

按通信拓扑与执行包络登记机制；空格表示尚无对应设备结果，不据架构猜测排除路线。局部、合成或未过正式精度门禁的结果保留为候选。

### MAP-Elites 结构格状态

|通信拓扑 \ 执行包络|多算子 ACLGraph|层级融合双积木|常驻 Megakernel|
|---|---|---|---|
|HCCL|[w80-tp8-ep8-sfa-sk-formal](evidence/ascend-decode/w80-tp8-ep8-sfa-sk-formal.json) 78层TP8+EP8（`9392`）冻结分母（220轮，4.655 tok/轮，SHA `f35b9bd1a857`）内部TPOT最小8.3532 ms（中位数8.3589 ms，`host_tokens=37.977 ms`，单轮总耗时38.842 ms，8-rank一致）；[w73-tp8-ep8-sfa-v2-sk-formal](evidence/ascend-decode/w73-tp8-ep8-sfa-v2-sk-formal.json) 最小8.654 ms（中位数8.671 ms，`host_tokens=39.104 ms`）；[w72-tp8-ep8-sfa-exact-sk-formal](evidence/ascend-decode/w72-tp8-ep8-sfa-exact-sk-formal.json) 中位数8.947 ms；旧TP8+EP8基线12.273 ms（单轮56.51 ms）；TP16对照见[w32-lookup-sfa1-ctrl](evidence/ascend-decode/w32-lookup-sfa1-ctrl.json) 11.024 ms|[w32-unigram-cyc-mlatail-sfa1](evidence/ascend-decode/w32-unigram-cyc-mlatail-sfa1.json) 78层TP16 `--mla-tail`内部TPOT中位数13.265 ms（verify发射1.280 ms，静态核244→241，16-rank一致）|待对应实测|
|离散 SHMEM|[w32-unigram-cyc-onesided-sfa1](evidence/ascend-decode/w32-unigram-cyc-onesided-sfa1.json) 78层TP16 `--onesided-reduce`实测`host_tokens=61.256 ms`（较HCCL AIV慢6.63 ms/轮）且SuperKernel下跨步无屏障致数值退化（输出仅2种token）；[完整 push][execution-gap]为组件对照|待对应实测|待对应实测|
|核内 SHMEM|[RSAG TP2/TP4/TP16](evidence/ascend-decode/rsag-a3-20261028/README.md) 双规约+CUBE 数值通过；首轮计时未准入|[三 rank 整层 tree][native-tp-tree-v1] 数值通过；合成权重、未计时|待对应实测|

### 时间余量的证据边界

各项只有在相同版本、输入轨迹和计时窗口下完成消融，才进入可加和账本。当前没有覆盖整轮的独立分解，余量均未定量确认。

|分量|已有依据|下一项可证伪检查|
|---|---|---|
|HBM|3.62 ms 为流量模型；[旧 round ledger][round-ledger] 不能证明实测下界或余量耗尽|同轨迹记录实际权重/KV 字节、复用与墙钟|
|宿主周转|[轮计时][round-is-device] 已区分发射与设备完成等待；`--hoist-metadata --hoist-forward-context --rope-graph --packed-host-inputs`将78层width32的`verify`主机发射从2.136 ms降至1.200–1.370 ms|单因素替换后测同配置整轮|
|PIPE_S|scalar 管道计数按核重叠，不能解释为可省的 20.4/28.95 ms 串行墙钟|SFA 索引/页表批量进 UB 的独立对照|
|TP 规约|[96 KiB、156 次相依 graph][tp-mode] 的 AIV p50 为 5.044 ms；不等于当前 width32 规约账单|匹配 shape、顺序与窗口协议后同机配对|
|GM 往返与启动|[原生两层对照][native-layer-native-timing-v3] 只覆盖单 bank 合成权重；78层`--mla-tail`实测`verify`发射省0.090 ms但设备执行增1.753 ms|78层`--routed-kernel --routed-shared-kernel`真实权重配对|
|Cube/Vector 算术|活跃计数不能单独换算为串行时间或排除访存改进|用实际关键路径与阶段替换确认余量|

本轮机制族排序为 SFA 元数据 MTE2→UB、分布式 FP32 butterfly、A3 编译器资源描述校正。第一族从预算 1 开始；每族两次无新物理预测的临时修订后停止，保留反例再切换路线。

---

## 2026-10-31 至 2026-11-01：78 层 TP8+EP8 算子级重构（56.51→38.84 ms/轮，TPOT 12.273→8.353 ms）与编译器/微架构意外发现

在 `7.150.11.149` 与 `7.150.10.180`（`Ascend910_9392`）上固定 78 层 `TP8+EP8`（`world_size=8, expert_parallel=True`）、8K 前缀 `case-00`、1024 输出 token、`width=32` 及完全冻结的起草器分母（`first_token=6500`、220 轮、`E[accept]=4.655`、输出 SHA `f35b9bd1a8578fd0e61dfc3823631b47e38bb730f876128ffb29dce70f4e8c45`），严格通过削减每轮 NPU 执行时间 $T_{\text{round}}$ 推进端到端 TPOT：

|实验标识|机型|核心机制|TPOT最小(ms)|TPOT中位数(ms)|轮数|接受/轮|单轮总耗时(ms)|host_tokens(ms)|输出SHA前12位|
|---|---|---|---|---|---|---|---|---|---|
|TP8+EP8 sfa1 基线|9392|TP8+EP8 初始 sfa1 静态核|12.251|12.273|220|4.655|56.510|55.120|`f35b9bd1a857`|
|[w70b-tp8-ep8-gmm12-sk-formal](evidence/ascend-decode/w70b-tp8-ep8-gmm12-sk-formal.json)|9392|Prolog/Quant 向量化 + GMM1/GMM2 W4A8 分块重构 + 静态核重编|9.247|9.259|220|4.655|42.473|42.148|`f35b9bd1a857`|
|[w72-tp8-ep8-sfa-exact-sk-formal](evidence/ascend-decode/w72-tp8-ep8-sfa-exact-sk-formal.json)|9392|同上 + SFA 24 核全量协同 `CoopMergeKv`/`ComputeMm1CoopTile`（`mtp_m1: 112→58.85 µs, target_m32: 196→153.98 µs`）|8.941|8.947|220|4.655|41.038|40.715|`f35b9bd1a857`|
|[w73-tp8-ep8-sfa-v2-sk-formal](evidence/ascend-decode/w73-tp8-ep8-sfa-v2-sk-formal.json)|9392|同上 + SFA 64位 `CeilDiv`/`CeilAlign` 修正 + `PreloadPipeline` 合一 + 两阶段 6 块辅助核卸载（`target_m32: 153.98→127.58 µs`）|8.654|8.671|220|4.655|40.320|39.104|`f35b9bd1a857`|
|[w80-tp8-ep8-sfa-sk-formal](evidence/ascend-decode/w80-tp8-ep8-sfa-sk-formal.json)|9392|同上 + `QuantLightningIndexer` 单遍就地归并 + 16行批量 `ProcessVD` + `MlaPrologV3` 静态核融合进 SuperKernel `sk_14`|8.353|8.359|220|4.655|38.842|37.977|`f35b9bd1a857`|

在 [sfa-fast-iter-20261031](evidence/ascend-decode/sfa-fast-iter-20261031/) 及后续静态核重构中，单算子 `mtp_m1` 从 `112 µs` 压至 `55.22 µs`（`-50.7%`）、`target_m32` 从 `196 µs` 压至 `125.10 µs`（`-36.2%`，`max_diff=0.00e+00`）、`QuantLightningIndexer` 从 `90.61 µs` 压至 `29.10 µs`（`-67.9%`），期间捕获并验证七项关键意外发现：

|发现编号|意外发现与根因机制|定量效应与验证证据|
|---|---|---|
|1. CANN `CeilDiv`/`CeilAlign` 16位隐式截断陷阱|CANN 9.2.0 的 `batch_matmul_utils.h`（被 `using namespace matmul;` 引入全局）仅定义 `uint16_t CeilDiv(uint16_t, uint16_t)` 与 `uint16_t CeilAlign(uint16_t, uint16_t)`。`SFAVectorService` 中全部 11 处 `int64_t` 调用被静默截断为 `uint16_t`（生成 `AND.b64 #65535` 与 32 位有符号除法），诱发寄存器溢出并多生成 41 条指令（164 B）|在 [sparse_flash_attention_common.h](evidence/ascend-decode/sfa-fast-iter-20261031/src/sparse_flash_attention_common.h) 补齐泛型 64 位 `CeilDiv`/`CeilAlign` 后，`target_m32` 从 `153.98 µs` 降至 `142.39 µs`（`-11.59 µs`），`mtp_m1` 从 `58.85 µs` 降至 `56.53 µs`（`-2.32 µs`），`max_diff=0.00e+00`|
|2. `PreloadPipeline` 双重内联膨胀与 L1 I-Cache 越界|`Process()` 同时内联 `ProcessBalanceCoop()->RunSingleToken()->PreloadPipeline()` 与 `ProcessBalance()->PreloadPipeline()`，加之辅助核二次内联 `MergeKv`，使 `mix_aiv` 暴涨至 `71,440 B`（越过 64 KB L1 指令缓存边界）、`mix_aic` 达 `34,636 B`，导致 `mtp_m1` 回退 `+2.2 µs`|在 [sparse_flash_attention_kernel_mla.h](evidence/ascend-decode/sfa-fast-iter-20261031/src/arch22/sparse_flash_attention_kernel_mla.h) 将两套入口合一并复用 `MergeKvCoop`，`mix_aiv` 缩减 `51.9%` 至 `34,376 B`、`mix_aic` 缩减 `36.3%` 至 `22,048 B`，`mtp_m1` 降至 `55.22 µs`，`target_m32` 降至 `127.58 µs`|
|3. 双通道 `bisheng` + `ld.lld -r` 的 `noinline` 弱符号碰撞与标量提升失效|MIX 算子分两次编译 `aic.o`（`dav-c220-cube`）与 `aiv.o`（`dav-c220-vec`）后由 `ld.lld -r` 合并；未按 `g_coreType` 模板特化的 `__attribute__((noinline))` 成员函数在两目标文件中产生同名 `WEAK` 符号，`ld.lld` 丢弃 `aiv.o` 版本导致 Vector 核跳入 Cube 指令流触发 `507015` 异常；即便加 `template<int32_t CT=g_coreType>` 消歧，切断 `Init()` 与 `ProcessBalance()` 内联边界仍阻碍 `this->constInfo` 标量提升|`noinline` 消歧版使 `target_m32` 从 `142.4 µs` 退化至 `168.4 µs`（`+26.0 µs`）；确立规则：MIX 算子内部通过消除重复调用点而非 `noinline` 控制代码体积|
|4. `target_m32` 邻接双 token 分核映射（`[2*c, 2*c+1]`）|`InitCalcParamsEach()` 按 `avgBaseNum = (32+24-1)/24 = 2` 将 32 个 query token 连续分配给 `Cores 0..15`：核 `c` 的 Wave 0 为 `qIdx = 2*c`、Wave 1 为 `qIdx = 2*c + 1`（而非跨步的 `16 + c`），`Cores 16..23` 因 `bN2Start=1 > bN2End=0` 完全闲置|修正前按 `16 + c` 卸载导致 Wave 1 读错 token（`max_diff=6.64e-01`）；修正为 `qIdx = 2*c + 1` 后精确闭合|
|5. `PreloadPipeline` 1 拍延迟真相（`(loop-1)%3`）与两阶段 6 块辅助核卸载（`s2Idx=0,1,3`）|1) `extraInfo2`（驱动 `Vec1L + ComputeMm2`）下标为 `(loop+2)%3 == (loop-1)%3`，即 `MergeKv+Mm1` 到 `Vec1L+Mm2` 实为 **1 拍延迟（`loop-1`）** 而非 2 拍（仅 `Vec2` 为 `(loop-2)%3`）；故 `gloop=4,5,7`（Wave 1 的 `s2=0,1,3`）分别在 `gloop=5,6,8` 被消费；2) `RunInfo extraInfo[3]` 按 `loop%3` 复用且 `CalcParams` 不清零扩展字段，必须显式 `else` 复位 `useCoopMm1`；3) 在 `gloop==5` 之前由 `Cores 16..23` 卸载 4 块（`s2=0,1`），在 `gloop==5..8` 之间趁 `Cores 0..15` 自算 `s2=2`（`gloop=6`）时复用已释放的 `kvSlot=4*c+0`/`mm1Slot=2*c+1` 再卸载 2 块（`s2=3, gloop=7`），实现 `16` 主核各算 `5` 块 + `8` 辅助核各算 `6` 块的精确负载均衡|4 块卸载（`s2=0,1`）将 `target_m32` 从 `142.39 µs` 降至 `137.85 µs`；两阶段 6 块卸载（`s2=0,1,3`）进一步将 `target_m32` 压至 **`127.58 µs`**（`-26.40 µs` vs `w72`），8 块卸载（`149.33 µs`）反证 `6` 块为精确平衡极值点，`max_diff=0.00e+00`|
|6. `8.6s` 直连 `bisheng` 编译与 `0.096s` `LD_PRELOAD` 热重载守护进程|1) 绕过 `opc` Python/TBE 包装，直接并行调用 `bisheng` 编译 `aiv.o`/`aic.o` 并配合 `-DSFA_FORCE_BF16=1` 与轻量 [matmul_intf.h](evidence/ascend-decode/sfa-fast-iter-20261031/fast_build/inc/lib/matmul_intf.h)，将编译耗时从 `14.5s` 压至 `8.6s`、回滚从 `14.5s` 压至 `0.001s`；2) 通过 [hotreload_sfa.c](evidence/ascend-decode/sfa-fast-iter-20261031/hotreload_sfa.c) 拦截 `aclrtBinaryLoadFromData` 与 `aclrtBinaryGetFunctionByEntry`，在常驻进程 [sfa_daemon.py](evidence/ascend-decode/sfa-fast-iter-20261031/sfa_daemon.py) 内以 `0.19 ms` 热替换 `.o` 内核二进制|单次 `NPUGraph` 计时 + `max_diff` 校验从冷启动 `8.0s` 降至 `0.096s`（提速 `83x`），端到端实验周期从 `22.5s` 降至 `8.7s`|
|7. `QuantLightningIndexer` 16行批量 `ProcessVD` 与 `S2_BASE_SIZE=2048` 单遍就地 Top-K|`ProcessVD` 原先逐行循环发射 16 次 `Cast/Mul/ReduceSum` 与 `WaitFlag`；合并为 16×64 整块向量运算后消除 15/16 标量发射。进一步在 `cuS2Len == cuS2LenVecAlign`（`S2_BASE_SIZE=2048 == kCount`）时跳过 `Sort32` + 双缓冲 `MrgSort` 树，直接在 `reduceOutBuff` 上执行单遍 `ArithProgression` 索引生成与就地写入，消除约 65% Vector 排序指令并保持平局索引偏序一致|单算子耗时从 `90.61 µs` 压至 `29.10 µs`（`-67.9%`），在 78 层整模静态核 [w80-tp8-ep8-sfa-sk-formal](evidence/ascend-decode/w80-tp8-ep8-sfa-sk-formal.json) 中保持 220 轮与输出 SHA 完全一致，将单轮 `host_tokens` 从 `39.104 ms` 压至 `37.977 ms`，TPOT 最小降至 `8.3532 ms`（中位数 `8.3589 ms`）|

---

## 2026-10-29：78 层 TP16 单轮分子（55.6–56.1 ms）与单样本尾部重复分母对账

在 `7.150.10.180`（`Ascend910_9392`）与 `7.150.11.49`（`Ascend910_9382`）单机 16 die、TP16、8K 前缀 `case-00`、1024 输出 token（`ignore_eos=True`）、3 次预热 + 3 次测量下完成九组 78 层真实权重端到端解码对照，按单轮总耗时分子 $T_{\text{round}}$ 与接受率分母 $E[\text{accept}]$ 拆解如下：

|实验标识|机型|核心机制|TPOT中位数(ms)|轮数|接受/轮|单轮总耗时(ms)|verify(ms)|host_tokens(ms)|输出SHA前12位|
|---|---|---|---|---|---|---|---|---|---|
|[w32-lookup-sfa1-ctrl](evidence/ascend-decode/w32-lookup-sfa1-ctrl.json)|9392|SFA block=1 + shortest=2|11.024|179|5.715|55.593|2.136|53.362|`7ce2cc012167`|
|[w32-lookup-envelope-opt](evidence/ascend-decode/w32-lookup-envelope-opt.json)|9392|SFA block=16 + 主机发射四联优化 + shortest=2|13.625|252|4.060|54.236|1.242|52.912|`ed4b659c3762`|
|[w32-unigram-envelope-opt](evidence/ascend-decode/w32-unigram-envelope-opt.json)|9392|SFA block=16 + 主机发射四联优化 + `--lookup-unigram`|11.823|213|4.803|55.662|1.200|54.377|`44e87bfb56af`|
|[w32-unigram-cyc-envelope-sfa1](evidence/ascend-decode/w32-unigram-cyc-envelope-sfa1.json)|9392|SFA block=1 + 主机发射四联优化 + `--lookup-unigram` + 周期短提案回绕|7.556|136|7.522|56.077|1.370|54.627|`0a0992795e4f`|
|[w32-unigram-cyc-envelope-sfa1-9382](evidence/ascend-decode/w32-unigram-cyc-envelope-sfa1-9382.json)|9382|同上（`9382` 跨机复现）|7.547|136|7.522|55.980|1.258|54.643|`0a0992795e4f`|
|[w32-unigram-cyc-recskip-sfa1](evidence/ascend-decode/w32-unigram-cyc-recskip-sfa1.json)|9392|同上 + 近邻优先退避与跳1回退|6.941|125|8.192|56.089|1.297|54.701|`0a0992795e4f`|
|[w32-tree-envelope-sfa1](evidence/ascend-decode/w32-tree-envelope-sfa1.json)|9392|SFA block=1 + 主机发射四联优化 + `--tree`（树状起草+行级掩码+compact）|19.287|250|4.092|76.971|1.919|67.152|`0a0992795e4f`|
|[w32-unigram-cyc-mlatail-sfa1](evidence/ascend-decode/w32-unigram-cyc-mlatail-sfa1.json)|9392|同上 + `--mla-tail`（78层MLA尾部融合，默认8核）|13.265|232|4.409|57.738|1.280|56.380|`2a5355844a30`|
|[w32-unigram-cyc-onesided-sfa1](evidence/ascend-decode/w32-unigram-cyc-onesided-sfa1.json)|9392|同上 + `--onesided-reduce`（离散SHMEM单边规约）|fail（数值退化）|33|31.000|63.308|1.425|61.256|退化（2种token）|

成立结论：
1. 分子与分母严格分离：`7.547 ms`（136 轮）与 `6.941 ms`（125 轮）的单轮设备耗时 `host_tokens` 为 `54.627–54.701 ms`（单轮总耗时 `55.98–56.09 ms`），较 `w32-lookup-sfa1-ctrl` 的 `53.362 ms` 未降反升；其表观 TPOT 下降完全来自单条校准样本 `case-00` 在 `ignore_eos=True` 强制生成 1024 token 后半程进入代码周期重复段（独立 token 数 364/1024），被短提案周期回绕与近邻退避连续命中 31 槽，不能外推为非循环语料（稳态接受率 `5.4–6.3` tok/轮）的端到端速度。停止在 CPU 字符串起草规则上继续刷单样本分母。
2. 主机发射包络（`--hoist-metadata --hoist-forward-context --rope-graph --packed-host-inputs`）的真实收益边界已锁定：在 78 层 TP16 上将 `verify` 阶段主机准备与入队耗时从 `2.136 ms` 压降至 `1.200–1.370 ms`（每轮省 `0.84–0.94 ms`），证明 CPU 侧发射余量已近物理地板，整轮唯一主矛盾是 `53.4–54.7 ms` 的 78 层 NPU 设备执行时间（`host_tokens`）。
3. 树状起草器对照（[w32-tree-envelope-sfa1](evidence/ascend-decode/w32-tree-envelope-sfa1.json)）证伪当前未融合形态下的树验证：在 `width=32` 预算下，分支占用行预算将最大深度限制在 16，且行级 `tree_mask` 与 KV cache `compact` 使单轮耗时从 `56.08 ms` 恶化至 `76.97 ms`（`host_tokens=67.152 ms`），轮数退至 `250`。
4. 78 层 `--mla-tail` 首跑（[w32-unigram-cyc-mlatail-sfa1](evidence/ascend-decode/w32-unigram-cyc-mlatail-sfa1.json)，MAP-Elites 格 `(hccl, fused_layer_bricks)`）证实：静态核由 `244` 降至 `241`、`verify` 发射降至 `1.280 ms`，但因 `mla_tail_binding.cpp` 在 `blocks=0` 时硬编码上限为 `std::min(8, maximum_blocks)`（仅用 A3 24 个 AIC 中的 8 个，使 `o_proj` 24 个输出 tile 串行跑 3 波），导致单轮 `host_tokens` 增加 `1.753 ms`（`+22.5 µs`/层）。
5. 78 层 `--onesided-reduce` 首跑（[w32-unigram-cyc-onesided-sfa1](evidence/ascend-decode/w32-unigram-cyc-onesided-sfa1.json)，MAP-Elites 格 `(discrete_shmem, multi_op_graph)`）证实：156 次离散单边规约在 `SuperKernel` 下单轮 `host_tokens` 达 `61.256 ms`（较 HCCL AIV 慢 `6.629 ms`），且因缺少跨轮 epoch 屏障导致对称窗口写覆盖，输出坍缩为常量循环（每轮 31 槽全中刷出虚假的 `2.090 ms`）并被反退化门禁拦截。
6. 定位并修复 `--routed-kernel` / `--routed-shared-kernel` 在非阻塞模式（`ASCEND_LAUNCH_BLOCKING=0`）下触发 `507015` AICore 异常（`The GM address accessed by scalar exceeds 48 bits`）的根因：`routed_int4_binding.cpp` 的 `UploadTiling` 在 `at::empty` 从 `NPUCachingAllocator` 复用刚释放的显存块后，未排空 `torch_npu` 后台 C++ 主机任务队列即直接执行同步 `aclrtMemcpy`，随后 `Execute` 触发 `c10_npu::getCurrentNPUStream().stream(true)` 将先前排队的算子刷入 NPU 流并覆盖 `scratch[kGateTiling]`（使 `gu_tiling` 变为激活值 `0x30339c7be5`）；同时 `torch_npu` 的 `static_kernel` 在 `_aclop_start_dump` 期间禁止分配新显存。通过在 `UploadTiling` / `Prepare` 中先执行 `aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream(true))` 并在 `CannModel.__init__` 预热工作区池，单卡 `davinci2` 复现用例验证通过。

---

## 2026-10-28：完整输入身份与资源描述回流

[checkpoint消费者核验](evidence/ascend-decode/checkpoint-a3-20261028/README.md)完成109文件、405008283962B的完整SHA与固定版本清单匹配，96个索引文件另过header/区间/张量覆盖；只读视图补齐两项optional对象。SDK9.2与匹配框架实际导入通过，整模型精度另验。[Carver资源修正](evidence/ascend-decode/carver-a3-20261028/README.md)已提交[上游PR #1955](https://github.com/tile-ai/tilelang-ascend/pull/1955)，真实9392资源读取、合法分块原生JIT和15项CPU检查通过，未声称自动选块或时延改善。

PR后续SDK重构的新增测试曾在collection提前初始化NPU，造成827项bad-fork失败；`233047cb`将导入延至测试执行，根线程CPU收集反例和断言AST对照通过。[当前CI记录](evidence/ascend-decode/carver-ci-20261029/README.md)：AscendC-PTO与原生NPU CI均通过，新增六项全过，主集1352项通过、3项跳过，示例153项通过；旧JIT证据保留原版本范围。

真实HF参考首次构造被Transformers5.19的RoPE构造接口拒绝；[显式CPU上下文修复](evidence/ascend-decode/hf-rope-constructor-20261028/readback.json)已过实际构造。完整32行+reset正在CPU执行；[CANN一例入口](../experiments/cann-precision/README.md)已有12项CPU检查及显式仅采集权限分离。[真实9382首跑](evidence/ascend-decode/cann-precision-a3-20261028/README.md)的16rank均在加载阶段因未知`rot`模块失败，前向帧为0。实际双进程注册表与加载方法对照确认遗漏`ascend_model`插件；补齐后，新9382完成109文件SHA、模型类解析、输入准入及96/96分片加载，随后首次前向前被采集器库哈希的`str`/`Path`类型错误中断。该错误已用真实映射ELF复现并修复；模型前向与严格HF比较另验。

[线性接受数修复](evidence/ascend-decode/linear-acceptance-20261028/README.md)从实际候选/预测的首次mismatch独立重算接受和预算裁剪，拒绝字段彼此一致却虚报全接受的收据；28项CPU检查通过，未构成设备数值或TPOT结果。

随后v4取得16rank×32个真实M1帧：[首帧与单位对照](evidence/ascend-decode/sfa-index-units-a3-20261028/README.md)显示embedding及第0层新KV逐位匹配HF，hidden1偏差8.18%、hidden2起NaN；M32被INT32/64观测槽不匹配中断。冻结BF16 SFA把token索引配成block16，同一实际算子的单die1/16/1见证逐位输出1/1.9375/1。观测dtype已修正并过15项CPU检查，block1单因素整模型复验另行验收。

block1重试已完成全部M1/M32/reset工件；首行NaN消失、预测恢复382，但独立HF首行仍438/1096项失败，M1/M32及精确reset也失败。HF自身32行+reset的25347项检查已全过。当前无本轮模型精度或新增TPOT通过；[原始结果](evidence/ascend-decode/sfa-index-units-a3-20261028/README.md#block1-整模型复验)保留。

[16rank三次M32边界采集](evidence/ascend-decode/reset-boundaries-a3-20261028/README.md)将首个已采差异推进至layer3 MoE：rank1/2的TP输入已变化，清尾部后rank3也变化；此前归一化与注意力采样重复一致。仅看rank0会误归因规约。下一步细分本地共享/路由专家及合并，无新增TPOT结论。

随后首差收窄至融合GMM1；[实际ELF重放](evidence/ascend-decode/gmm1-selection-a3-20261029/README.md)确认插件把原vendor加到overlay之前。保留原vendor路径使overlay优先后，同一9392单die的8192行INT8码全匹配CPU、32次逐位重复；FP32 scale尚有32行最多3 ULP，严格退出1。启动器修正路径顺序，整模型精度与TPOT继续验收。

[修正包的TP16 HF尝试](evidence/ascend-decode/fixed-hf-launch-20261029/README.md)在49节点加载68/96时被外部SIGKILL并删除，0个rank报告、0个Tensor工件；退出137。节点随后已被其他VLLM占用，未重建。本次不形成数值判定；后续先完成CPU完整SHA与顺序读，再经即时空闲检查启动设备作业。

180节点随后完成相同修正包的TP16严格复验，取得576个PT与16份报告，rank状态一致、每rank的35072项reset_exact检查全过；M1/M32从第13行开始不同，两次对照各582项失败。每rank共211428项检查、40807项失败；首行HF为425/1096失败，最终logits NRMSE为0.0771525。[原始读回与归档](evidence/ascend-decode/fixed-hf-launch-20261029/README.md#9392完整复验)保留，模型精度仍未通过。

[固定真实概率的Value重排](evidence/ascend-decode/mla-value-order-a3-20261029/README.md)在9392单die比较BF16先展开与后投影，各32次逐字节稳定；先展开匹配CPU，后投影对自身CPU最大差9.31e-10。两图有510/1024项不同、NRMSE为0.00333698。新用例的CPU设置已对齐冻结HF的mkldnn=False；该见证尚待实际层0 attention/Oproj输入确认对hidden1的贡献，未测时延。

[真实TP16第一层采样](evidence/ascend-decode/layer0-boundaries-a3-20261029/README.md)每rank22项检查全过，逐位复现原完整CANN首帧。input norm匹配HF；固定实际attention输出后，原HF post norm与MLP逐位重建CANN h1；固定实际16份o_proj输入，参考投影的BF16汇总也匹配实际attention。首token的第一层偏差收窄到o_proj之前。局部partial与FP32规约差异均未改变该处BF16汇总；无新增模型精度或TPOT通过。

同一层的[Value闭包](evidence/ascend-decode/layer0-boundaries-a3-20261029/value-result.json)固定HF概率并接原o_proj/norm/MLP：先展开分支逐位重建HF h1；后投影分支对HF h1的NRMSE为0.00348409，对实际CANN仍有0.00350671残差。Value见证已接入第一层，但尚不能单因解释CANN偏差；继续采集实际SFA边界。

[实际SFA边界](evidence/ascend-decode/layer0-boundaries-a3-20261029/sfa-result.json)再次逐位复现原完整CANN首帧；实际SFA输出接参考UV/o_proj/HF尾部，在16rank逐位闭合CANN h1。固定实际query/KV的FP32候选context对设备NRMSE为0.00028106–0.00101148，概率对HF为0.00452071–0.02901260；仅把概率转BF16仍未闭合。下一步区分QK分数舍入与query吸收投影，不把全局CPU softmax当CANN内部模拟。

[score/Value舍入与扩层](evidence/ascend-decode/attention-rounding-a3-20261029/README.md)：保持BF16概率的单因素CPU对照中，BF16 score令概率最大NRMSE从0.028113降至0.001300。组合先展开Value后，真实TP16 h1误差从0.003595降至0.001961；单die重放的score/P逐位闭合CPU，Value改FP32未改变最终PV，停止该旋钮。前三层误差为0.001961/0.004716/0.006684，16rank一致、原hidden容差内；h3只改善约14%，未达预注册30%预测。局部逐位对账没有转化为整模精度或时延通过；下一步沿原采集扩层，保留首个越界与离散选择证据，避免继续追平容差内的CPU比特。

[完整常驻模型](evidence/ascend-decode/resident-attention-a3-20261029/README.md)随后取得两臂各首帧/reset的64个PT和32份rank报告。同一冻结HF、相同模型与world、固定HF32=false，原臂复现既有0.0771525 logits误差；组合降至0.0608584，首个hidden越界h7→h8、首个专家选择分歧layer6→14，满足预先登记的两项传播预测。16rank的全部观测字段一致，reset及新增路由weights/logits逐位重复；首帧精度仍失败，无新增TPOT。

[prefix Value缓存](evidence/ascend-decode/prefix-values-a3-20261029/README.md)一次准备固定前缀投影、decode仅写当前行，在真实9382单die完成两臂各32次完整attention重放。缓存臂逐位匹配固定CPU PV，输入与缓存未变；目前只支持M1/8193，尚未验收整模精度及性能。

[随后三臂计时](evidence/ascend-decode/prefix-value-timing-a3-20261029/README.md)在同一9382单die取得60组原生/展开/缓存对照，各臂逐位复现自身冻结输出、输入与缓存未变。12组预热、六种次序、独立进程/缓存、计时外256MiB压力、profiler关闭；完整attention＋UV墙钟p50为263.544/783.941/906.016µs，event为197.310/738.320/808.510µs。缓存未达到相对展开至少20%改善，反而慢15.6%；相对原生60组均更慢，停止此实现的性能接入，保留数值干预价值。无新增TPOT。

[原生M1/M32 row13读回](evidence/ascend-decode/width32-firstdiff-20261029/README.md)绑定原始rank0两PT与teacher，CPU mmap真实读取、零NPU绑定、退出0。h0～h68逐位相同，layer68输出h69首次不同，NRMSE 0.001774；layer70 latent以0.012712首先越过原容差，indices集合重叠2045/2048。此前indices仅顺序不同，row13返回ID仍相同。层内attention/MLP及其他行历史KV尚未划分，下一次采样前移到layer68两积木边界；未据此宣告根因、精度或TPOT通过。

[layer68历史核对](evidence/ascend-decode/layer68-history-20261029/README.md)进一步读取rank0 serial00～13：14行h68合并残差、新latent/rope和routes均逐位相同，选择集合相同、部分顺序不同，386项冻结前缀检查通过。row13选择了全部14个新增位置，未发现不同历史KV；norm实际两输入、QL和层内结果仍需设备采样。已提取22.6MB原始输出控制，避免新采样逐次读取1.62GB宽帧。

[layer68三臂TP16采样](evidence/ascend-decode/layer68-boundaries-a3-20261029/README.md)确认原生/复制/位置排序的row13 h69与最终logits逐位相同rank数为0/0/16，logits NRMSE为0.0327916/0.0327916/0。16rank实际norm输入、QL/QPE与有效KV相同；原生首差只在rank12 SFA的1个BF16元素，随后传播到TP输出。[同die固定M1干预](evidence/ascend-decode/sfa-order-a3-20261029/README.md)只替换实际M32索引顺序，32次均逐位复现M32 row13输出；三臂96个设备输出另经CPU重读核对。索引顺序足以造成该局部分歧；排序改变原生M1数值，完整HF、多输入及性能仍待验收。

Value缓存性能预测失败后的这段执行由整模精度排障占据主线，尚未形成新增TPOT。[排序三臂计时](evidence/ascend-decode/sfa-order-timing-a3-20261029/README.md)已在9382单die完成M1/完整M32、两width各60组对照。width32原生/预排序/每次排序的墙钟p50为329.762/341.002/357.398µs，event为258.020/271.830/293.750µs；两width均未达预排序双钟p50/p95至少改善5%的预测，每次排序亦无净收益，停止此实现的性能接入。432个设备样本、全部双钟统计及完整输出另经独立重读；数值结果单独保留。整模精度排障不再作为其他组件计时的前置条件。

[合包输入准备](evidence/ascend-decode/host-inputs-a3-20261029/README.md)把host tokens、positions、slots及length放进一个pinned packet，保留固定typed views。9382单die的width32准备窗口墙钟p50为767.251→628.517µs，event为604.400→424.150µs，p95均下降；墙钟18.082%未达预注册20%，不更改门槛。计时包含fixture生成与CommonMetadata、不含vendor builders或模型，137.520µs配对节省不并入整轮账本。默认关闭，`--packed-host-inputs`供后续对照。

[真实TP16消费者](evidence/ascend-decode/host-inputs-model-a3-20261029/README.md)已在同一常驻模型核对两臂的M1首行、完整M32及reset。独立CPU读取96PT，旧设备输出、输入、新KV、跨臂/reset及16rank最终输出摘要均通过，16,496项比较无失败；每阶段前缀的18,528项现场断言另保留，未由PT独立重算。一个8K输入的eager数值结果，HF、实际模型graph、八输入与在线TPOT仍待验收。下一性能实验先取得同包真实width32整轮的诊断trace和profiler-off窗口，再选择通信buffer两端cast融合的完整消费者对照；保留原HCCL结合顺序及BF16舍入，剖析请求不入TPOT。

本轮执行反省：局部舍入对账挤占了完整请求对照；逐层准备约48秒，完成16/78层后才停止并换常驻模型。传播误差下降尚未形成通过精度的基线，完整性能A/B也未完成。后续常驻整模对照沿用固定容差，组件候选单独取得数值与配对墙钟证据；容差内的CPU比特差异不作为继续迭代的理由。

[速度门禁反例修复](evidence/ascend-decode/diversity-gate-20261029/README.md)：八输入、32请求、1024个零、双钟1ms的同一CPU夹具，从旧`passes_target=true`变为false。当前从请求工件重算token/bigram/周期及文件SHA，应用显式冻结policy，并保留双钟与覆盖条件。36项CPU检查通过；这是准入修复，未测设备时延。

## 2026-10-28：RSAG 所有权重排的 A3 数值门禁

[原通信链 bootstrap](evidence/ascend-decode/shmem-bootstrap-a3-20261028/README.md) 在 9392、CANN 9.1、源码构建 SHMEM 上通过 TP2/TP4 的升 rank FP32 算术对照；TP4 随机输入与实际 HCCL 有差异，原串行顺序不视为 HCCL 等价。随后 [RSAG](evidence/ascend-decode/rsag-a3-20261028/README.md) 将控制与候选统一为原 FP32 TreePlan，仅重排 owner 分片、scatter/gather 与 ready 协议。两臂 TP2/TP4、rows1/12/32 共 324 条输入/权重/两次规约/哨兵检查通过，162 条跨臂摘要一致；含 eager、变化输入 graph、轮换延迟和无中间 barrier 的连续 parity 回放。随后两臂真实 TP16、blocks24 的 32 份 rank 默认门禁通过。四 rank 的顺序见证显示 HCCL 随 rows 改变结合顺序；TP16 的该见证在三种 rows 均匹配固定树。特殊有限值的 CUBE 后结果及 edges 首规约有共同 CPU 差异，默认 `pass` 不覆盖这些观测。设备时延另测，固定树未据此成为 HCCL 的通用等价替换。

## 2026-10-28：RSAG 的 TP16 首轮计时未过准入

[三轮 ABBA 原始记录](evidence/ascend-decode/rsag-a3-20261028/README.md)保留6对独立world、每形状30样本、连续32个完整probe均摊的双时钟。rows1退化；rows12/32原始p50虽下降，最坏提交包络比例为30.88%/13.03%，均超过预注册1%（rows1为138.47%），不足以认定性能收益；不删样本或放宽阈值。固定树的原型与数值证据保留，转接真实权重的正式CANN精度采集。

## 2026-10-28：SFA 元数据预搬运未形成收益

[真实 9392 记录](evidence/ascend-decode/sfa-metadata-a3-20261028/README.md)：top-k/页表 MTE2→UB 候选与控制在 width1/8/32、8K、ACLGraph 各通过 60 个逐位边界检查。width32 的 30 对同机隔离进程 AB/BA 中，墙钟 p50 为 402.370→489.194 µs、设备 event 为 317.890→408.790 µs；配对节省 p50 分别为 −87.504/−87.510 µs。每对恢复 KV、变化输入和页表、计时外 256 MiB 压力，五个实际 CANN 边界与两臂摘要均匹配。原实现保留，首个 Luby 预算 1 结束；下一族改查通信数据流，资源描述校正独立侦察。单 die 合成 Prolog+SFA，未外推 TPOT。

## 2026-10-28：token staging 的真实 A3 验收

[9392 单 die 记录](evidence/ascend-decode/token-staging-a3-20261028/README.md)：36 个 payload/stream 检查通过。两个独立进程配对实验在 width8/16/32 各保留 60 对，全部 360 对新臂更快；width32 的配对节省 p50 为 2241.135/2127.836 µs。计时包含实际 round 执行、NPU 整数加一及主机提交；候选、预测、接受与缓存长度均精确一致。第二次占用采样未发现其他 NPU 进程，各臂独立缓存、每轮 256 MiB 压力位于计时外。预建全部缓存目录后的第三次 width32 复测另有 60/60 对更快，节省 p50 为 2228.885 µs。组件假设通过，完整 checkpoint、TP16、模型精度与在线 TPOT 继续验收。

## 2026-10-28：主机候选只上传一次

[CPU 配对差分](evidence/ascend-decode/token-staging-20261028/README.md)：width32 每轮 `64 as_tensor + 2 stack + 1 cat` 收敛为 `1 tensor`，轮 CPU p50 为 170.255→26.839 µs，30 对样本的节省中位数 143.364 µs；线性与树轮只下载预测。候选、预测、接受及状态检查通过；A3 H2D、精度和整请求 TPOT 尚未测定。

tree 容量只剩一行时，实际轮宽缩至 1，旧路径调用 `propose_tree(0)` 报错。现只验证 pending root，保持预测、compact、EOS 与缓存长度一致；[CPU 检查](evidence/ascend-decode/tree-capacity-20261028.json)覆盖八种声明宽度和 48 个单行尾部用例。

[在线 lookup 接入](evidence/ascend-decode/online-lookup-20261028/README.md)已通过 CPU/HTTP 与 16 并发 worker 检查：请求自有 prompt、width32、目标图预热回滚、零 MTP 模型加载和拒绝后复用。实际 CANN 图、HIXL、常驻显存、width32 数值及 TPOT 尚待设备验收；正式 CANN 精度门禁接线也尚未完成。

[离线 host 起草](evidence/ascend-decode/host-draft-model-20261028/README.md)也停止构造未执行的 MTP 模型，角色、状态与报告统一，18 项 CPU 检查通过。实际框架的全局 RoPE/共享 head 副作用仍须 A/B；tree 的 lineage 回执与树接受验收尚未闭合。

后续 [tree lineage 验收](evidence/ascend-decode/tree-lineage-20261028/README.md)补齐轮结果与交付记录的父数组，独立重建分支接受；25 项 CPU 检查覆盖 26,260 个小树和 24 个 session 的 1,801 个 span，另通过宽度、round、EOS/取消回归。CANN 设备树数值与 TPOT 仍未验收。

[P/D 计时账本](evidence/ascend-decode/pd-command-clock-20261028/README.md)修复只统计单轮调用、遗漏实际批命令的问题；24 项 CPU 回归通过，命令数、轮数与交付/cache 一致。批命令时长单独命名，未据此生成逐轮时延或 CANN 精度结论。

[lookup CPU 对照](evidence/ascend-decode/lookup-policy-20261028/README.md)统一反向扫描与当前索引的最长尾部为 32，三个回归检查通过。无索引 `rfind` 在本地纯复制上的每请求总成本 p50 为 48.173→0.397 ms，歧义短尾却为 64.830→439.356 ms；四种输入、240 个样本的提议与接受签名均与独立扫描一致。保留实验与原始样本，生产实现仍用原索引；设备成本与 TPOT 尚待测量。

后续 [精确压缩尾部索引](evidence/ascend-decode/lookup-radix-20261028/README.md)替代多项式哈希，并修正一个碰撞遮蔽更长命中的合法词表反例。79 项 CPU 回归、24,159 次独立提议检查与 126 个容量/EOS 场景通过。本地四类负载的每请求起草总成本 p50 从 47.988–64.813 降至 6.491–10.459 ms，120 对全部下降；无匹配稳态仍有 16.081 ms 高尾。完整分位数、差值与源码绑定保留在证据中，尚无 A3 或 TPOT 结论。

[全局 radix arena](evidence/ascend-decode/lookup-arena-20261028/README.md)的 4608 个原始样本全部签名一致。坐标版使复制/轻度编辑总 CPU p50 退化约 8–9%，无保留控制下仍成立，未替换生产；四面板未重现旧约 16 ms 高尾。

后续[局部坐标边](evidence/ascend-decode/lookup-coordinate-20261028/README.md)保留节点字典，只删除持久 tuple 标签与逆序 word 构造。生产 A/B 的四类总 CPU p50 约降 6–40%；完整签名、79 项回归、126 个实际 session 与逐前缀正负例通过。carry 编辑稳态 p99 仍从 2.536 增至 3.771 ms；另一个带 callback 的 1.734 ms commit 与 GC2 区间重叠 1.706 ms，记录回收数为 0。原始尾部与所有退化保留，生产 GC 配置不变。

[同源码原生 lookup](evidence/ascend-decode/lookup-native-20261028/README.md)保持算法文件不变，通过增补声明消除热路径的 Python 属性与遍历分派。本地四类 8K+1024 轨迹总 CPU p50 从约 5.0–8.9 降至 1.9–4.0 ms，正常 GC 的 240 对总成本全部下降；carry 编辑稳态 p95 仍从 2.177 增至 2.708 ms。79 项回归、24159 次独立提议、126 个 session、安装及入口检查通过；1536 个原始样本的 864384 轮与完整统计另经独立读回。默认仍用 Python，服务器须重建并显式安装受检产物；NPU 与 TPOT 尚未测定。

[私有无环节点 GC](evidence/ascend-decode/lookup-gc-20261028/README.md)保留起草器的循环回收，本地每节点构造分配少 16 字节；两臂各 79 项回归、20 例生命周期与候选安装通过。正常 GC 的总成本 207/240 对下降，复制收益很弱；carry 歧义稳态 p95 从 0.665 增至 1.337 ms。1536 个原始样本、864384 轮和全部退化经独立读回，生产全局 GC 配置不变；服务端重建、NPU 与 TPOT 继续验收。

[FP32 head 参考](evidence/ascend-decode/fp32-head-reference-20261028/README.md)补齐显式输出算术、实际导入源与模式准入。匹配的 C8 header 已在仓库逐文件核对；[78 层合成 CPU 执行](evidence/ascend-decode/hf-runtime-20261028/README.md)修正当前 HF gate/MoE 路由协议，实际运行最终与 77 个诊断 head。两种 head 模式各 2379 项采集/reset 对照、48 项独立检查、44 项回归及 62 项元数据检查通过。BF16/FP32 的合成贪心结果分别为 0/17；真实 checkpoint、Prolog-v3、CANN FP32 caller 及实际 graph 数值继续验收。

[32 行参考帧](evidence/ascend-decode/graph-frame32-20261028/README.md)保存合成 78 层模型的 96 次实际 HF teacher forward，组成 A/B/A 三帧，各含 628 字段。完整严格账本的 297632 项检查中，256 项为预期数值负例失败，意外失败为零。旧接收器会让空参考零检查通过；当前接收器从保存的 Tensor 核对 60288 个字段行、重算 629 项标量 oracle，并拒绝 15 类空缺及身份类型反例。真实 8K/top-2048、NPU producer、SuperKernel 图与正式 CANN 精度门禁仍待闭合。

[持久观测槽](evidence/ascend-decode/graph-producer-20261028/README.md)在实际 `CannModel.verify` 入口执行 CPU AOT 夹具，A/B/A 的 96 行通过 104832 项原标量检查及 96 项输出 token ID 对照；八个负例均拒绝，包括实际删除 AOT stamp 写回后执行的坏图。观测槽保留完整字段、当前代次与独立复制，未执行真实 checkpoint、attention、NPU/SK 或 HF 精度验收。

## 整请求速度与输出

### 2026-10-17 至 23：历史整请求记录与更正

#### 10-17：[vLLM P+D 基线][vllm-baseline]

同一 8K 输入，3 次预热、32 次串行 1024 输出；TP16、MTP3、graph4，客户端 TPOT p50/p95 为 16.390/18.295 ms。

vLLM 同时执行 P 与 D。32 条输出在 `temperature=0, seed=0` 下仍各不相同，新请求不宜用作逐 token 参照；客户端与内部回调计时窗口不同，加速比须在同一窗口下比较。

#### 10-17：[M1/M4 ACL Graph][cann-graph]

M1 32 步的 Graph/eager/reset 张量逐位一致；内部三请求中位数 44.68 ms/token。M4 的 30 个请求各输出 32 token，p50/p95 为 38.70/39.02 ms；随后三个 1024 输出请求中位数 20.44 ms，303 轮。

M4 草拟仍为 eager；两批请求的输出长度和接受数不同，数值与计时按各自配置使用。

#### 10-19：[pinned staging][metadata-h2d]

两个配对的轮时差为 −3.384/−3.791 ms；pinned 六请求为 11.6198–11.6929 ms，pageable 九请求为 12.3175–12.6394 ms。

共享设备上的同源配对支持该元数据搬运改动；历史 chain 与冻结 teacher 在 offset4 分歧，精度验收未通过。

#### 10-20：[plain combine][free-machine]

首轮四臂复用了同一静态编译缓存，A/B 失效。隔离缓存后两对请求均值下降 0.104/0.269 ms。

隔离缓存后的配对差分有效；最低单请求值仍须连同输出、编译产物和质量记录使用。

#### 10-23：[whole-machine 对照][whole-machine]

同一 8K 前缀快照、lookup width12：256 输出 TPOT 42.46 ms、接受 1.364；1024 输出为 22.13 ms、接受 2.59；轮约 57 ms。

输出长度改变了轨迹和接受数；该输入的短输出与长输出须分别计时。

#### 10-23：[9.3794 ms 来源复核][record-9379-provenance]

找到服务器历史 `auto12` 记录，列出的三请求为 9.4007/9.3879/9.4434 ms；命令选择 automaton，旧 `drafter` 字段硬编码为 lookup。输出 1024 token 仅 55 种，113 个不同 bigram。

“没有记录”和“从未运行 automaton”两条判断撤回。历史值有运行来源，但重复输出未满足质量相当的比较条件；归档仍缺 9.3794 的精确重算。

#### 10-23：[shared-FIA/tree 对照][shared-fia-arms]

chain width8 为 14.743 ms、仅 32 种 token；chain width32 为 29.148 ms、188 种；tree width32 为 17.044 ms。缺少位置/compaction 或使用有损掩码的 SFA tree 臂无效。

循环输出会抬高输入 bigram 命中率，语义正确性仍未成立。14.743 ms 仅作历史运行读数，性能改进未通过验收。

### 2026-10-23：width 16/24/32 的 lookup 对照

实验比较加宽验证后的每 token 耗时。三组均为完整 78 层、同一冻结 8K 前缀、lookup 起草器、SFA、`--target-graph --compile-target --super-kernel --fp32-head`；各用新容器，3 次预热、3 次测量，每请求输出 1024 token。记录于 10 月 23 日产生，24 日取回。

| 验证宽度 | 输出数/轮数 | TPOT 中位数 |
| -- | -- | -- |
| 16 | 3.483 | 17.132 ms |
| 24 | 3.180 | 20.108 ms |
| 32 | 5.721 | 12.413 ms |

width16/24/32 每请求分别运行 294/322/179 轮。用 TPOT × 输出数/轮数估算，轮成本分别约为 59.67/63.95/71.01 ms。

[运行说明][width-ladder]与三个摘要：[width16][w16-lookup]、[width24][w24-lookup]、[width32][w32-lookup]。精确轮成本须由逐轮记录计算。内部 TPOT 的计时分母如下。

12.413 ms 是这组对照中最低的内部 TPOT，按 `(末次 callback − 首次 callback)/(输出 token 数 − 1)` 计算。首个输出来自 prefill；窗口包含草拟、验证、接受提交和 Gloo 命令，排除前缀就绪、HTTP、HIXL 与 reset。每次请求重复注入同一个前缀。计时实现见 [bench_cann_mtp.py](../tests/glm52/bench_cann_mtp.py)。

摘要中的 `all_rank_state_agreement=true` 表示所记录的一致性检查通过；正式精度与多输入验收状态见 `precision_verified=false`、`formal_1024_token_run=false`。中位数和 p95 统计的是三个请求各自的平均 TPOT，与逐 token 间隔 P90、AgentX 的指标不同。

width24 的平均接受数较低，原因待查。该组使用线性 `LookupDrafter`，[实现][lookup-draft]没有 `propose_tree`，因此“树预算分配导致下降”的归因不成立。宽度会改变轮起点和设备数值路径；同一位置的候选前缀与整请求平均接受数须分别比较。下一检验是在固定目标串上重放各宽度的轮起点、候选和接受位置，并检查设备的宽度一致性。

### 2026-10-23：固定宽度换起草器，固定起草器换宽度

同一输入、SFA、1024 输出，3 次测量：

| 配置 | TPOT 范围 | 输出数/轮数 |
| -- | -- | -- |
| width12 + automaton | 23.799–23.864 ms | 2.570 |
| width12 + lookup | 22.148–22.162 ms | 2.590 |
| width4 + automaton | 23.960–24.022 ms | 2.325 |

同三组请求的输出记录：

| 配置 | 输出摘要前缀 | 不同 token 数 |
| -- | -- | -- |
| width12 + automaton | `2cb99464d9a4` | 139 |
| width12 + lookup | `2cb99464d9a4` | 139 |
| width4 + automaton | `f06b1373167f` | 136 |

原始记录：[w12 automaton][w12-auto-rank-0]、[w12 lookup][w12-lookup-rank-0]、[w4 automaton][w4-auto-rank-0]。固定 width12 时输出相同，lookup 的整请求更快，说明比较起草器还须计入草拟代价。换宽度后输出不同，待定位设备算术、状态和掩码的影响。token 多样性相近只描述输出分布；输出摘要相同只覆盖这次起草器对照。任务质量、正确性和其他输入的未来行泄漏仍需各自检查。

[CPU 宽度不变性检查][check-round-invariance]覆盖三个确定性目标、三个真实起草器与宽度 2/4/8/12/16/32，共 54 个组合；错误未来掩码和未归并接受路径的负对照会失败。这验证了标量目标上的宿主编排；NPU 算术与实际 attention 掩码在该检查之外。

## 数值一致性与在线 P/D

### 2026-10-16 至 20：局部算术与完整 teacher（参照序列）

#### 10-16：最小 decode 是否正确

[CPU/NPU smoke](evidence/ascend-decode/README.md)：4 层、hidden256、合成 BF16 权重，初始请求、reset、不同短请求各检查；CPU/NPU 均 1729/1729 通过。

该最小模型的 token、KV、索引/专家选择与状态检查通过。

#### 10-17：真实算术差异来自哪里

[W4 MSD](evidence/ascend-decode/w4-msd.md)、[A8 rounding][a8-rounding]、[RoPE][rope-rounding]和 [DynamicQuant Div][dynamic-quant-div]分别绑定打包、缩放与舍入规则。

独立 CPU 参照必须复现被比较路径的中间数据类型与舍入。

#### 10-17：fused SwiGLU 是否可重复

[屏障对照][swiglu-barrier]：11 个错误 scale 等于旧 gate 的 amax/127；增加 vector barrier 后，三 rank 的冻结专家重放 832/832 行逐位一致。

该受控改动支持 UB 生产消费顺序的修正。

#### 10-17：M1 是否复现同 CANN 路径

[32-step all-rank 对照][cann-vllm-32]：8K 前缀，16 ranks；344064 个浮点张量逐位匹配，reset 对比 172032 个也逐位匹配。

一条冻结轨迹上的 M1，在同 CANN 算术下控制与执行一致。

#### 10-17：独立 HF 参照是否一致

[修正 oneDNN 后的 78 层比较][hf-corrected]：两个 teacher 位置 final top1 均一致，但 hidden、logits 和 index 集仍有记录中的差异。

1587 项参照执行/reset 检查通过；16 ranks 的设备数值仍有差异，未判定整体通过。

#### 10-18：完整 teacher 与 LM head

[layer-major 78×32][layer-major]：BF16 head 命中 31/32，差异点为 16.75 平局；真实 BF16 权重、FP32 输出 head 在 M1/M4 命中 32/32。TP16 head Graph 对比 475791360 个 FP32 值逐位一致。

固定 teacher 的完整执行与独立 head 路径成立。layer-major 改变驻留顺序，此处只报告数值；Graph 检查范围为 head。

#### 10-20：宽验证保持 teacher 吗

[完整 M32][m32-full]：同源 TP16、78 层，SFA 为 32/32、shared-FIA 为 30/32，后者错在 query3/21；2528 个产物经审计。

shared-FIA 在该固定输入上存在完整 teacher 差异，局部提速与数值差异须一并评估。

#### 10-20：历史 chain 第一次分歧

[首次分歧][old-chain-first-difference]定位到第二轮 query0、position8195：6500 对 teacher2524；227 项提交/交付与记录预测一致。[两轮冻结重放][two-round-replay]得到 4/4 标签。

输出交付与记录一致；分歧位于目标计算或其执行条件。

#### 10-20：norm/quant 融合是否改变结果

[受控替换][norm-quant-fusion]：只替换前三层的 post-attention norm/quant，完整 78 层从 4/4 变为 3/4，在全部 16 ranks 复现 2524→6500。

这三个替换足以复现该冻结输入的第一次 token 翻转；须保留中间 BF16 舍入。其他宽度的分歧原因另查。

### 2026-10-17：真实 CannModel P/D 冷启动

[在线记录][pd-live]与[原始审计][pd-live-audit]覆盖真实 router → vLLM-Ascend prefill → 16 个 HIXL WRITE → 设备驻留前缀 → CannModel M1 → HTTP。输入 8192 token，返回 8 token，执行 7 个真实 decode 步骤，主缓存位置 8192→8199，最后输出保留为 pending，reset 后 slot 回到 idle。

KV 由在线链路接收，随后停止 P、加载 D，过程未用文件回灌。P/D 共驻预热因显存不足失败；成功请求总耗时 117.59 秒，其中 inject 的等待和加载约 109.18 秒。首个 decode 间隔 3790.4 ms，后六个 189.5–208.6 ms。这次冷启动验证了在线功能，稳态计时列入末节。

[serve_cann_pd.py](../tests/glm52/serve_cann_pd.py)使用 `CannModel`；上述在线记录只验证了 M1。lookup、width32、compiled target、SuperKernel、FP32 head 与 target graph 的组合在 10-28 完成 [CPU 接入检查](evidence/ascend-decode/online-lookup-20261028/README.md)，实际在线设备运行尚待验收。[前缀字节/receipt 校验][prefix-8k]与[多输入数值和请求计时协议][measurement]分别保留传输与验收约定。

## 原生融合与跨 rank 规约

本组归档在 [megakernel-device-20261021][megakernel-device]。实验使用 TP16 分片尺寸、关闭 EP，覆盖 M1/2/12/32 的原生阶段与整层入口。以下按实际设备 rank 数分组，参数均为各记录的合成参数；完整 checkpoint 与 16 个设备 rank 的验收见末节。

### 单 rank：层内衔接与局部计时

#### [原 DecoderLayer attention 路径][decoder-wk-m12-int8-wide-v1]

原始 DecoderLayer 6→7，M1/2/12/32、每宽度 8 个变化输入；修正 M1/M12 的 K 投影后，816 个边界匹配，两臂都保留官方 FFN。

TP 边界用本地恒等操作代替，验证层内 attention 衔接。

#### [完整原生 attention/FFN Graph][decoder-native-full-graph-wide-v1]

原始 DecoderLayer 6→7，四宽度、两种残差入口；16 次捕获、128 次回放，3264 项字节边界检查通过，回放无 Python 调用。

保留原 final reduction helper 和各自 workspace 所有权；TP 设备集合通信未执行。

#### [单 rank 两层链][layer-local-chain]

连续 indexed/shared 两层；eager 与官方 ACL Graph 各检查 3488 个字节边界，Graph 含 4 次捕获、32 次变化输入回放。

attention 的实际输出进入 FFN，保留 workspace 与缓存所有权；TP 规约仍用恒等操作代替。

#### [N512 局部整层计时][native-layer-native-timing-v3]

M12，两块 scratch arena 相对每层三次原生 launch，配对中位数分别节省 20.216/21.212 µs/两层；60/60 对更快。

计时范围是单 bank、重复合成权重下的单 rank 层内路径。

#### [直接写 Top-K 输出][native-layer-pair-timing-v1]

尽管字节检查通过，同一 N512 对照的两个 arena 分别慢 14.26/9.09 µs/两层，60 对均未胜。

保留原 Top-K 复制；少一次复制没有在此配置形成收益。

### 两 rank：整层衔接与计时

#### [两 rank 整层 eager][native-tp-chain-wide-v2]与[Graph][native-tp-graph-wide-v1]

M1/2/12/32；每组 7232 个中间字节检查、256 次真实 HCCL 对比通过；Graph 为 8 次捕获、64 次变化输入回放。SHMEM window 跨层跨宽度复用，宿主不重置 epoch。

两 rank 的数值、可见性与生命周期检查通过；两项相加尚不能区分多 rank 规约括号。

#### [两 rank M12 完整路径计时][native-tp-timing-m12-v1]

两 arena 节省 23.049/23.264 µs/两层，60/60 配对更快。每图重复 16 个两层单元、每事件 30 次回放，先取两 rank 本地设备事件时间最大值，再配对相减。

收益同时包含层编排与 SHMEM/HCCL 实现变化。one-pair 敏感性对照节省 32.449/31.283 µs。

### 三 rank：规约顺序与整层集成

#### [三 rank 规约顺序负对照][comm-order-three-rank-v1]

M1/M12，`[1, 2^-25, -1]`：串行 FP32 加法得 0，安装的 deterministic HCCL 得 `2^-25`。36 个 eager/graph 用例复现。

串行顺序在此三 rank 输入上与 HCCL 不等价，替换规约须核查加法括号。

#### [私有 tree reducer][comm-tree-three-rank-v1]

三 rank、M1/M12，72 个用例与实际 HCCL 逐位一致；覆盖消去、有符号零、BF16 次正规数、溢出和最终 BF16 舍入平局，144 项紧邻消费者检查通过。

比较范围为第一个规约及其消费者；未测 NaN 输入与性能。

#### [三 rank 整层 tree 集成][native-tp-tree-v1]

两个 FP32 tree 规约均在层 launch 内。M1/M12，eager 与官方 Graph 各 5424 个字节检查、192 个 tree 检查通过；两臂 96 个输出摘要相同。Graph 捕获不改 678 个已观察边界，回放无 Python 调用。

三个设备 rank 执行，另 13 个 CPU peer 仅提供构造尺寸；本项未计时。

复现 attention 结果须使用记录中的冻结库、实际 dispatch 和数值顺序。WK 的 K 累加契约分别见 [M1 实现][indexer-wk-m1-native]、[M12 修正][indexer-wk-m12] 与 [M32 修正][indexer-wk-kshift]。[O 投影审计][o-epilogue-source-audit]中的实际调用要求先算 `channel * token`，再乘 accumulator；旧 tail 的依次相乘会改变结果。修正依据是所记录的 M、dispatch 和中间数据类型；scale 数据类型或旧接口文档的默认库不足以确定这条数值路径。

[已安装 HCCL 二进制/源码审计][tp16-reduction-order-source-audit-v1]给出候选 TP16 树的距离 8/4/2/1 及独立 CPU 见证；实际 TP16 选择器、拓扑和设备算术列入末节验收。该 BF16 全互发方案在 TP16 下的远端张量数据字节数（不含标志与协议开销）是所比较 FP32 分区集合通信的 4 倍；该结果用于比较流量，时延须由对应运行测定。

## 整轮耗时与性能分析

### 2026-10-20 至 22：支持局部机制的测量

#### 10-20：[关键路径采样][critical-path]

旧 width8 的两个 profile 轮中，记录到 TP cast/规约、backbone 和空闲区间；profile 轮约 62.4 ms，无 profile 约 53.05 ms。

trace 可定位依赖；11.48 ms 空闲是否可消除须做对照，重叠通信时长须按依赖关系计入关键路径。

#### 10-21：[eager/Graph 调用成本][graph-floor]

记录区分 eager 与 graph 的调用/回放代价。

eager 宿主墙钟与完整 Graph 内的每 kernel 成本须分别测量。

#### 10-22：[向量阶段额外调用][cost-is-per-pass]

12×8192 FP32，五次重复、每次 30 迭代；sum 额外调用 eager/graph 约 11.13/11.12 µs，topk 约 17.55/17.11 µs。

这组形状下，图捕获后仍有额外设备工作；增量随算子与形状而定。

#### 10-22：[共享数据载入][shared-load]

合成 vector 阶段，1→4 项操作：复用同次 UB 载入 59.37→61.10 µs，各自读写 59.74→133.84 µs。

支持合并这组数据遍历；其他层须按实际数据量重测，并保持 ordered combine 的求和顺序。

#### 10-22：[Matmul 调用次数][invocation-has-no-fixed-cost]

M12、2624 输出列、六种 K；第二次 invocation 增量随 K 变化，拟合截距−1.58 µs、斜率 4.41 µs/1000K。

增量随 K 变化，所测范围不支持每次 invocation 固定 11.6 µs 的旧模型；其他 kernel 的固定成本另测。

#### 10-22：[TP16 行成本][row-cost-collective]

BF16、156 次相依 graph 规约，width12→32 斜率约 0.019 ms/行；width12 链约 5.125 ms。

增量仅对应这条 BF16 规约链；生产 FP32 cast/reduce 全路径和其他轨迹的整轮斜率须分别测量。

#### 10-22：[专家行成本][row-cost-experts]

合成 W4A8 TP16 分片、轮换 weight bank、每形状 20×20 调用；8→256 活跃专家的 gate_up/down 增量分别约 0.709/0.586 µs/专家。

斜率对应所测合成形状；teacher 与运行中被拒绝候选的专家并集不同，普通 GMM 与生产 fused GMM1 也须区分。

### 2026-10-23：宿主发射和设备等待分开计时

[round-is-device]使用 TP16、78 层、shared-FIA、width8、metadata hoist；277 个测量轮，每轮四阶段各标记一次。propose/verify/host_tokens/resolve 分别为 0.037/2.138/50.454/0.090 ms。`verify` 是宿主提交区间，`host_tokens` 包含设备完成等待。该记录不支持“每 kernel 有几十微秒宿主提交费，减少 launch 即可省掉整轮固定项”的解释。这四阶段是宿主计时，设备 kernel 的重叠须另查。

该组 TPOT 约 14.605 ms，输出质量见前述 shared-FIA 历史记录。metadata/rope 改动前后实测为 14.743→14.605 ms；单 die 微基准外推的 7.28 ms/轮收益未获这次运行支持。

### 2026-10-23：width12 稳态性能采样

[采样请求记录][w12-profile-rank-0]与[kernel 账单][kernel-account]覆盖 51 轮、131529 次 launch、260 种 kernel。每轮 kernel 时长（Duration）之和 53.052 ms，其中 fused GMM1 为 8.840 ms、down GMM 为 6.554 ms、SFA 为 7.089 ms。53.052 ms 是 kernel 时长之和；整轮墙钟须单独读取。采样请求的输出长度和轨迹与 1024 输出、无 profiler 的对照不同。

`aicore_time`、`aiv_time` 和各流水线计数按核统计，时间可重叠。因此，15.480 ms AIC scalar 加 13.471 ms AIV scalar 得出的“28.95 ms 可消除串行墙钟”不成立；MAC 与 MTE2 相加也不给出整轮物理下界。这些热点可用于选择下一次阶段替换对照，全局瓶颈还需该对照确认。

历史 `round = 45.88 + 0.814 × width`、固定 kernel 费和单 die 各项相加的[round ledger][round-ledger]保留为模型记录。该模型混合单 die、不同算子和历史整轮锚点，2.x ms 可达性及硬件下界的结论均未得到验证。后续预算使用同一版本、同一轨迹的整轮原始计时。

## Attention、缓存与宽验证

### 10-18：[SFA 替代路径][sfa-reuse]

合成 M4 完整 graph：SFA 88.44 µs、gather/FIA 136.26 µs。真实 layer0 M4 的连续缓存对照：SFA 89.05 µs、direct INT32 gather/FIA 69.15 µs；运行时 A/B/reset 逐位匹配各自路径。

连续缓存特化改善了该输入，通用 gather 较慢；跨 kernel 的数值差异仍存在。

### 10-19：[C1 dispatch][sfa-routing]

M8/P8192、普通 indices；V1 约 90–92 µs，C1 约 998–1003 µs，三臂输出一致。

仅切 dispatch 的 C1 方案更慢；full-K 输入未覆盖 bounds terminal 分支。

### 10-20：[causal mask 与共享 FIA][fia-mask]

masked/filtered CPU FP64 差异 1.943e−16；safe-gather+mask 对未来 KV 变化不敏感。四层 M8 graph 从 425.780 降至 364.470 µs，30/30 配对更快。

mask 与局部 Graph 检查通过；完整 M32 teacher 仍为前述 30/32，宽度间的等价性未成立。

### 10-20：[M16 共享 FIA][m16-screen]

四层 graph，配对中位数 373.365→328.620 µs，29/30 胜；12 项共享/重建与 24 项 Graph/reset 逐位检查通过。

收益覆盖这四层的固定输入。

### 10-20：[FIA 精度开关][fia-row-correction]

8 个冻结 Q/KV/mask 测试输入，`inner_precise=0→2` 的 524288 个 BF16 值逐位不变。

切换此开关后，这些输入的差异仍在。

### 10-20：[late attention][late-attention]

M32 层 66–73，19 对新 cache 与 M1 逐位一致；shared/rebuilt FIA 至 pre-O 逐位一致，SFA/FIA pre-O 仍有差异。

缓存一致而 pre-O 仍有差异，比较范围定位到该局部 attention 输出。

### 10-20：[paired C8 cache][index-cache-pair]

复用 Norm scatter 并以 FP16 scale 字节 view 成对写入；18 项完整 backing 的 eager/Graph/reset 检查通过，M1/M8/M32 各 10/10 配对更快。

改动仅作用于 INT8 缓存，FP8/C4/BF16 路径未改变。

### 10-22：[attention 行成本][row-cost-attention]

单 die，4 heads/top-k2048/8K/BF16、20×20 调用；逐行 gather/FIA 随 width 增长，shared-FIA 在所测 width1–64 约 85–113 µs。

微基准形状决定了增量；稀疏集合、掩码和读布局必须与运行路径一致。

### 10-23：[真实选择集合][shared-gather-union]

一次 12 行、每行 2048 选择，共 24576 项，并集 3732，重叠比例 6.59；与 row0 的交集比例不随行号单调。

该索引缓冲区有复用机会；79.8→12.1 MB 和 5.5 ms 收益均为旧外推，实际省字节量与轮时仍待测。

### 10-23：[width16 SFA][s16-sfa-rank-0]与[shared-FIA][s16-shared-rank-0]

SFA 约 17.07 ms、接受 3.480；shared-FIA 约 26.62 ms、接受 2.195，两臂输出不同，推算轮时约 59.4/58.4 ms。

输出与接受流不同，TPOT 尚不能区分缓存读取的贡献，近似轮时也不足以排除 gather 的关键路径影响；下一对照须冻结同一 Q/KV、选择与接受流。

## MoE、量化与通信组件

### 10-17/18：[HCCL 模式][tp-mode]与[完整单边 push][execution-gap]

FP32 96KiB、156 次相依 graph 规约，AI_CPU/AIV 的 rank-MAX p50 为 10.931/5.044 ms；另一次共享设备上，完整双 slot push 为 35.72 µs，对照完整 HCCL 算术 59.18 µs。

两组是独立计时；peer load、裸 HCCL、cast/reduce 全链及含 READY/epoch 的协议须分别计时。

### 10-20：[MIX shared expert][mixbrick]

单 die 手写 cube→vector→cube 链；M8 动态量化版本 25.724 µs，对照四算子 33.952 µs，所测 M8/M32 值匹配。

动态 scale、跨核 flag 与 top-level barrier 是该路径的条件；静态 scale 的 13.863 µs 归于另一配置。

### 10-20：[routed kernel][routed-kernel]

16 专家×每专家 2 行，专家轴分工，固定 tile 后约 41.99 µs；L2 hint 及 hoist Matmul 对象未见收益。

与 vendor 157.2 µs 的形状不同，撤回“3.93 倍”及其 TPOT 推算；该合成形状自身的数值与时延有效。

### 10-20：[INT4 基础路径][int4-weights]

高级 Matmul 接口的 INT4 组合失败；基本 API 以 `0x11/0x21` 填充验证两个 nibble 分别参加 Mmad，工作路径使用 MIX 形状。

所测布局与基础指令链可执行；完整 checkpoint、生产 NZ 布局和整模型性能另行验收。

### 10-20：[compact GMM1][gmm-compact]

63 项 CPU 不变量、24 项变化 group 的 Graph 逐位检查、4 次真实 checkpoint 调用；同 ELF 配对在所测 M1/M8/M32 形状改善。

保留为可选路径，生产路由分布尚待验证。

### 10-20：[GMM1 full-A][full-a]、[N split][n-split]、[K512]

full-A 四组均回退；N128 改善 M1/M8/shared8 却恶化 M32 random，N64 四组均慢；K512 相对原 N256 控制不构成统一改进。

各形状取舍不同，保留原默认 tiling。

### 10-20：[compact GMM2][gmm2-compact]及[adaptive]

compact 改善 M1/M8/shared8，但 M32 random 近似不变；adaptive 改善 M32 random/M8，却增加 M1/shared8 成本。变化路由和真实 checkpoint 局部数值检查通过。

可按真实 active-count 选择候选，回退 tile 的成本也须计入。

### 10-20：[dispersed M32 guard][guard-release]

18 项实际发布端 CPU 检查、58 项 Graph/CPU-MSD 比较通过；三个 teacher 重构分布节省 11.18–22.12 µs。

guard 的收益覆盖所测 teacher 分布；运行中的 lookup 拒绝后缀仍须采样。

### 10-20：[reduced greedy head][reduced-head]

真实 TP2 eager/Graph/reset 匹配，但 M1/M8/M32 配对墙钟增加 14.18/20.74/29.85 µs。

所测 TP2 配置变慢，直接复用该 head 路径无收益。

### 10-20/21：[空转 kernel 复核][kernel-timing-void]

旧计时路径因 cube 侧错误读取 UB 等条件未执行预期专家工作，修复后的耗时与原值不一致。

旧最快值因未执行预期工作而无效，融合收益须随有效输出重测。

### 10-22：[empty-group 对照][bank-skip]

96 行、16 专家 bank、BF16 down 形状，7×50 次：保留 16 专家但仅 5 个非空 87.23 µs，紧排 5 专家 87.10 µs；全 16 非空 250.53 µs。

同时改变 bank 大小与活跃专家数的旧实验有混淆。该形状未测出删除空专家的收益，旧“narrow bank 省 6–17 ms/轮”撤回；保留实际活跃组与源码跳过空组的证据。

## 起草器的接受数与成本

### 10-19：[indexed lookup][lookup-contract]

一条 8K+1024 历史轨迹，30 对/width；候选与边界不变，含索引构造的 lookup 总耗时约 1030→20.6–20.9 ms。

CPU 数据结构收益成立。pending 须已在上下文中，填充行不提交，末尾 bonus 按请求长度裁剪；MTP 缓存独立管理。

### 10-19/20：[unigram 与 adaptive][policy-summary]

同轨迹最短 suffix2→1，M8/16/32 轮数 253/198/174→228/171/147；[adaptive][adaptive-summary]为 175 轮/2240 行，固定 32 为 147 轮/4704 行。

选策略须同时计入轮数、验证行数和各宽度分桶的实测成本。

### 10-20：[trusted teacher33][teacher33]

真实索引起草器、独立扫描和回放一致；M8/16/32 为 32 个新输出均需 28 轮。

该短 teacher 轨迹的接受数低于较长的历史重复串。

### 10-20：[copy tree][copy-tree]与[wide frontier][wide-frontier]

固定 32 预算的 flat/tree2/tree4/tree8 分别 147/135/132/140 轮；同一轨迹 M32/64/128/256 为 147/144/144/144 轮。

树形与加宽在该串上的边际收益有限；设备 tree 速度与其他工作负载的宽度上限尚待测量。

### 10-20：[merged tree][merge-tree]

用四种次序分配已有 chain 长度，接受数 8.317–13.286。

数字来自旧轨迹的链长度重分配；未访问位置的 MTP 候选未知，新策略的真实接受率仍待测。

### 10-22：[aligned cursor][aligned-drafter]

合成顺序编辑串上改善；两条实际历史输出重放时，width32 aligned 接受约 1.94/1.88，lookup 约 5.22/5.02，automaton 约 5.95/6.02。

撤回将合成编辑串增益外推到模型输出的预测。旧输出也高度重复，两类输入均未覆盖独立 agent 编辑工作负载。

## 早期实验索引

修改以下组件时，先从对应记录核对配置、实际产物和源码版本。

- 真实 checkpoint 读取与专家切片：W4 打包、TP shard、scale/bias 与真实投影的局部校验。证据：[权重读取][ascend-decode-weights]、[专家链][expert-chain]。

- 稀疏 indexer：BF16 与 INT8 分开验收，选择集合相同不等于编码相同。证据：[8K 选择](evidence/ascend-decode/indexer.json)、[输入量化][indexer-input-quant]。

- KV 接收与捕获：main P、MTP P−1、parent hidden 状态及字节接收凭据。证据：[布局检查][ascend-layout-v2-source-binding]、[HIXL 捕获][prefix-8k]。

- HBM 微基准：四组轮换 513MiB 输入、100 预热/1000 样本、15 项校验和；设备事件中位数 1.155 TB/s，墙钟 1.019 TB/s，作为该 die/负载的流式读取参照。证据：[a3-hbm-20261017.json](evidence/a3-hbm-20261017.json)。

- 完整编译与算子绑定：区分配置接受、实际 ELF 加载、Graph 数值和请求速度。证据：[execution-gap]、[静态编译包来源][shared-fia-model-cache]。

- MTP 自身 head 与 forward：seed、shared-index chain、conditioned 输入及 target/draft cache 位置；冻结 parent 结果与实际 chain 分开。证据：[layer-major 报告][layer-major]、[offline-development]。

- 原生组件与真实消费者：从局部字节检查追到原生层消费者，核对合成参数、NZ 布局与 TP 规约的各自覆盖范围。证据：[routed INT4][routed-int4]、[MLA tail](../tests/glm52/mla_tail/README.md)、[indexed attention][attention-indexed]、[完整 FFN][ffn-original-shared-gu-static]。

- checkpoint 恢复：109 文件已核对路径/大小，缺失 quarot 文件已恢复；95 个大权重分片未重新计算哈希，全量字节复验尚未完成。证据：[checkpoint-restore][checkpoint-restore-v1]。

## 待完成的共同验收

下一步以同一配置连接数值、常驻运行与整请求收益。各实验特有的条件保留在对应小节。

1. 同配置的数值与多输入：width32/lookup 的 12.413 ms 来自三次重复冻结前缀请求。八个独立输入、完整数值、任务质量和长会话尚待验收。最小合成模型尚未覆盖完整 W4A8C8、TP16 和 8K；M1 的同 CANN 单轨迹结果尚未覆盖 width32、其他输入和任务质量。局部算术、head Graph、attention、缓存及 GMM 的检查须继续接到完整 teacher 与请求 TPOT。
2. 同配置的常驻在线运行：把最好内部配置接入已贯通的 CannModel P/D runner，完成常驻部署、预热后生产端与客户端计时、reset 和背压检查。
3. 真实 TP16：whole-layer 的单、双、三 rank 结果使用合成参数。仍须加载完整 checkpoint，执行真实 TP16 设备集合通信，记录实际 rank 数、HCCL 选择器、拓扑与规约算术，并用同源 baseline 检查请求输出和客户端 TPOT。若要把局部层计时或 reduced greedy head 的 TP2 结果推广到 TP16，须在真实 TP16 路径复测。
4. 同配置的整请求收益：每个新 kernel 或编译替换都须绑定实际加载产物，在冻结输入上检查数值、缓存、输出和紧邻消费者，再测整轮差分，最后在同一配置的完整请求中确认收益。计数器、流量模型与局部计时用于选择对照实验，整轮差分保留为机制分析的证据。

[a8-rounding]: evidence/ascend-decode/a8-rounding.md
[adaptive]: evidence/ascend-decode/gmm2-compact-20261020/adaptive/README.md
[adaptive-summary]: evidence/ascend-decode/lookup-contract-20261019/adaptive-summary.json
[aligned-drafter]: evidence/ascend-decode/aligned-drafter-20261022/README.md
[ascend-decode-weights]: evidence/ascend-decode/weights/README.md
[ascend-layout-v2-source-binding]: evidence/ascend-decode/ascend-layout-v2-source-binding.json
[attention-indexed]: ../tests/glm52/whole_layer/attention_indexed.md
[bank-skip]: evidence/ascend-decode/bank-skip-20261022/README.md
[cann-graph]: evidence/ascend-decode/cann-graph-20261017/README.md
[cann-vllm-32]: evidence/ascend-decode/cann-vllm-32-20261017.json
[check-round-invariance]: ../tests/glm52/check_round_invariance.py
[checkpoint-restore-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/checkpoint-restore-v1-a3-194/README.md
[comm-order-three-rank-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/communication-order-three-rank-v1-a3-194/AUDIT.md
[comm-tree-three-rank-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/communication-tree-three-rank-v1-a3-194/AUDIT.md
[copy-tree]: evidence/ascend-decode/lookup-contract-20261019/copy-tree/README.md
[cost-is-per-pass]: evidence/ascend-decode/cost-is-per-pass-20261022/README.md
[critical-path]: evidence/ascend-decode/critical-path-20261020/README.md
[decoder-native-full-graph-wide-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-native-full-graph-wide-v1-a3-194/summary.json
[decoder-wk-m12-int8-wide-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-wk-m12-int8-wide-v1-a3-194/summary.json
[dynamic-quant-div]: evidence/ascend-decode/dynamic-quant-div.md
[execution-gap]: evidence/ascend-decode/execution-gap-20261018/README.md
[expert-chain]: evidence/ascend-decode/weights/expert-chain.json
[ffn-original-shared-gu-static]: ../tests/glm52/whole_layer/ffn_original_shared_gu_static/README.md
[fia-mask]: evidence/ascend-decode/fia-mask-20261020/README.md
[fia-row-correction]: evidence/ascend-decode/shared-fia-model-20261020/fia-row-correction/README.md
[free-machine]: evidence/ascend-decode/free-machine-20261020/README.md
[full-a]: evidence/ascend-decode/gmm-compact-20261020/full-a/README.md
[gmm-compact]: evidence/ascend-decode/gmm-compact-20261020/README.md
[gmm2-compact]: evidence/ascend-decode/gmm2-compact-20261020/README.md
[graph-floor]: evidence/ascend-decode/graph-floor-20261021/README.md
[guard-release]: evidence/ascend-decode/gmm2-compact-20261020/guard-release/README.md
[hf-corrected]: evidence/ascend-decode/hf-corrected-20261017/README.md
[index-cache-pair]: evidence/ascend-decode/index-cache-pair-20261020/README.md
[indexer-input-quant]: evidence/ascend-decode/indexer-input-quant.md
[indexer-wk-kshift]: ../tests/glm52/whole_layer/indexer_wk_kshift/README.md
[indexer-wk-m1-native]: ../tests/glm52/whole_layer/indexer_wk_m1/native.h
[indexer-wk-m12]: ../tests/glm52/whole_layer/indexer_wk_m12/README.md
[int4-weights]: evidence/ascend-decode/int4-weights-20261020/README.md
[invocation-has-no-fixed-cost]: evidence/ascend-decode/invocation-has-no-fixed-cost-20261022/README.md
[k512]: evidence/ascend-decode/gmm-compact-20261020/k512/README.md
[kernel-account]: evidence/scalar-bound-20261023/kernel-account.txt
[kernel-timing-void]: evidence/ascend-decode/kernel-timing-void-20261021/README.md
[late-attention]: evidence/ascend-decode/shared-fia-model-20261020/late-attention/README.md
[layer-local-chain]: ../tests/glm52/whole_layer/layer_local_chain/README.md
[layer-major]: evidence/ascend-decode/layer-major-20261018/README.md
[lookup-contract]: evidence/ascend-decode/lookup-contract-20261019/README.md
[lookup-draft]: ../tilert/models/glm_5_2_ascend/lookup_draft.py
[m16-screen]: evidence/ascend-decode/fia-mask-20261020/joint/m16-screen/README.md
[m32-full]: evidence/ascend-decode/shared-fia-model-20261020/layer-major/m32/full/README.md
[measurement]: evidence/ascend-decode/measurement.md
[megakernel-device]: evidence/ascend-decode/megakernel-device-20261021/README.md
[merge-tree]: evidence/ascend-decode/merge-tree-20261020/README.md
[metadata-h2d]: evidence/ascend-decode/metadata-h2d-20261019/README.md
[mixbrick]: evidence/ascend-decode/mixbrick-20261020/README.md
[n-split]: evidence/ascend-decode/gmm-compact-20261020/n-split/README.md
[native-layer-native-timing-v3]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-native-timing-v3-a3-194/summary.json
[native-layer-pair-timing-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-pair-timing-v1-a3-194/summary.json
[native-tp-chain-wide-v2]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-chain-wide-v2-a3-194/summary.json
[native-tp-graph-wide-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-graph-wide-v1-a3-194/summary.json
[native-tp-timing-m12-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-timing-m12-v1-a3-194/summary.json
[native-tp-tree-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-tree-v1-a3-194/AUDIT.md
[norm-quant-fusion]: evidence/ascend-decode/shared-fia-model-20261020/norm-quant-fusion/README.md
[o-epilogue-source-audit]: evidence/ascend-decode/megakernel-device-20261021/integration/o-epilogue-source-audit/README.md
[offline-development]: evidence/ascend-decode/offline-development.md
[old-chain-first-difference]: evidence/ascend-decode/shared-fia-model-20261020/old-chain-first-difference/README.md
[pd-live]: evidence/ascend-decode/pd-live-20261017/README.md
[pd-live-audit]: evidence/ascend-decode/pd-live-20261017/audit.json
[policy-summary]: evidence/ascend-decode/lookup-contract-20261019/policy-summary.json
[prefix-8k]: evidence/ascend-decode/pd-hixl-20261017/prefix-8k-20261017.json
[record-9379-provenance]: evidence/ascend-decode/record-9379-provenance-20261023/README.md
[reduced-head]: evidence/ascend-decode/reduced-head-20261020/README.md
[rope-rounding]: evidence/ascend-decode/rope-rounding.md
[round-is-device]: evidence/ascend-decode/round-is-device-20261023/README.md
[round-ledger]: evidence/ascend-decode/round-ledger-20261023/README.md
[routed-int4]: ../tests/glm52/routed_int4/README.md
[routed-kernel]: evidence/ascend-decode/routed-kernel-20261020/README.md
[row-cost-attention]: evidence/ascend-decode/row-cost-attention-20261022/README.md
[row-cost-collective]: evidence/ascend-decode/row-cost-collective-20261022/README.md
[row-cost-experts]: evidence/ascend-decode/row-cost-experts-20261022/README.md
[s16-sfa-rank-0]: evidence/width16-20261023/s16-sfa-rank-0.json
[s16-shared-rank-0]: evidence/width16-20261023/s16-shared-rank-0.json
[sfa-reuse]: evidence/ascend-decode/sfa-reuse-20261018/README.md
[sfa-routing]: evidence/ascend-decode/sfa-routing-20261019/README.md
[shared-fia-arms]: evidence/ascend-decode/shared-fia-arms-20261023/README.md
[shared-fia-model-cache]: evidence/ascend-decode/shared-fia-model-20261020/cache/summary.json
[shared-gather-union]: evidence/shared-gather-20261023/union.txt
[shared-load]: evidence/ascend-decode/shared-load-20261022/README.md
[swiglu-barrier]: evidence/ascend-decode/swiglu-barrier.md
[teacher33]: evidence/ascend-decode/lookup-contract-20261019/teacher33/README.md
[tp-mode]: evidence/ascend-decode/tp-mode-20261017/README.md
[tp16-reduction-order-source-audit-v1]: evidence/ascend-decode/megakernel-device-20261021/integration/tp16-reduction-order-source-audit-v1/README.md
[two-round-replay]: evidence/ascend-decode/shared-fia-model-20261020/two-round-replay/README.md
[vllm-baseline]: evidence/ascend-decode/vllm-baseline-20261017/README.md
[w12-auto-rank-0]: evidence/scalar-bound-20261023/w12-auto-rank-0.json
[w12-lookup-rank-0]: evidence/scalar-bound-20261023/w12-lookup-rank-0.json
[w12-profile-rank-0]: evidence/scalar-bound-20261023/w12-profile-rank-0.json
[w16-lookup]: evidence/ascend-decode/width-ladder-20261023/w16-lookup.json
[w24-lookup]: evidence/ascend-decode/width-ladder-20261023/w24-lookup.json
[w32-lookup]: evidence/ascend-decode/width-ladder-20261023/w32-lookup.json
[w4-auto-rank-0]: evidence/scalar-bound-20261023/w4-auto-rank-0.json
[whole-machine]: evidence/ascend-decode/whole-machine-20261023/README.md
[wide-frontier]: evidence/ascend-decode/lookup-contract-20261019/wide-frontier/README.md
[width-ladder]: evidence/ascend-decode/width-ladder-20261023/README.md
