# GLM-5.2 W4A8C8 TP8+EP8 Megakernel 与基准套件

本目录承载 Ascend A3 单机 8 die（TP8+EP8）GLM-5.2 W4A8C8 8K/1K 基准数据集与辅助脚本（核心 Megakernel 源码位于 `tilert/models/glm_5_2_ascend/ops/`）。目标唯一锁定 [case-01-coding](case-01-coding)（batch=1、ISL=8192、OSL=1024），要求稳态中位 TPOT < 3.0 ms，1024 词元输出命中黄金 SHA256（2f751372fc82cf290c8e35d3e7cc3cad0296fde4d8d188fba102ac34f82c2962），且 8 卡主网与独立串行条件草稿 KV 位级对齐。架构总纲见 [standalone-decode.md](../../../docs/standalone-decode.md)，投机台账见 [speculative-decoding.md](../../../docs/speculative-decoding.md)。

## 📊 基准集与样本隔离纪律

|用例|样本角色|ISL/OSL|使用约束|
|---|---|---|---|
|[case-01-coding](case-01-coding)|唯一正式验收目标|8192/1024|严禁在其生成序列或墙钟上拟合超参；仅用于终态门禁与里程碑配对复核|
|[case-02-review](case-02-review)|样本外调优集|8192/1024|树拓扑宽度 W、层宽调度 b_sched 与稀疏预算的样本外筛选|
|[case-03-refactor](case-03-refactor)|样本外调优集|8192/1024|同上|
|[case-04-math](case-04-math)|样本外调优集|8192/1024|同上|

## ⏱️ 当前已验证基线（Ascend910_9392，TP8+EP8）

|里程碑|验证宽度与树调度|轮数/每轮接受|75层链耗时|稳态中位TPOT|8卡状态与黄金SHA256|
|---|---|---|---|---|---|
|MTP W8 单链基线|W=8 线性单链|197 / 5.193|—|9.1010 ms|通过|
|DSpine 锥形树 + K11030|W=16，[4,4,4,4,2,2,2,2,1,1]|149 / 6.866|41.888 ms|7.5988 ms（最快 7.4716 ms）|100% 逐位对齐|
|K110383（NZ 注意力尾 + GU_B12 + DN_KB4）|W=16 同构验证|149 / 6.866|41.399 ms|单轮验证墙钟 43.172 ms|8 卡 0-ULP|
