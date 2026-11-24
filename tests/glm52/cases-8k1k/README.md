# GLM-5.2 W4A8C8 8K/1K 基准套件（共享输入 + 按拓扑分轨真值）

本目录承载 GLM-5.2 W4A8C8 标准 8K/1K（batch=1、ISL=8192、OSL=1024）基准输入与多拓扑黄金真值（核心 Megakernel 源码位于 [ops](../../../tilert/models/glm_5_2_ascend/ops)）。所有拓扑共享同一套冻结输入与预填充首词，按芯片与规约拓扑分轨保存确定性真值：不同并行度改变 BF16/FP32 跨卡规约树括号（如 `TP8` `[4,2,1]` 对 `TP16` `[8,4,2,1]`），在近似同分位点产生各自唯一可复现的 0-ULP 输出序列。

## 🗂️ 目录分层

|路径|作用域|说明|
|---|---|---|
|[cases-index.json](cases-index.json)|拓扑无关|4 组冻结用例索引（`case_id`、`seq_len`、`first_token_id`、`prompt_sha256`）|
|[prompts-8192.json](prompts-8192.json)|拓扑无关|4×8192 原始提示词元数组|
|[sources/](sources)|拓扑无关|4 份提示源模板，供 [cases.py](scripts/cases.py) 校准生成 `prompt.json`|
|`case-XX-*/prompt.json`|拓扑无关|单用例冻结 8192 提示（与 `/prestage/cases-8k1k/case-XX-*/` 对齐）|
|`case-XX-*/golden/<track>/`|按拓扑分轨|存放对应拓扑的 `result.json`、`manifest.json`、`decoded.json`|
|`golden/<track>/bench-summary.json`|按拓扑分轨|该拓扑全部已捕获用例的摘要总表，由 [finalize.py](scripts/finalize.py) 生成|

## 📊 用例与隔离纪律

|用例|样本角色|首词 ID|提示 SHA256|`a3-tp8-ep8` 黄金 SHA256|`a3-tp16-ep16` 黄金 SHA256|
|---|---|---|---|---|---|
|[case-01-coding](case-01-coding)|唯一正式验收目标|13534|`a93d673b...`|`2f751372...`|`4b0d2fc1...`|
|[case-02-review](case-02-review)|样本外调优集|334|`dc019623...`|`b4f29fde...`|待捕获|
|[case-03-refactor](case-03-refactor)|样本外调优集|1499|`16f65c81...`|`65031e32...`|待捕获|
|[case-04-math](case-04-math)|样本外调优集|8419|`ba9be528...`|`68d439b6...`|待捕获|

## ⏱️ 已验证拓扑基线（`case-01-coding`，冷启动 2+5 正式门禁）

|轨道|硬件与规约树|主干节点 / 里程碑|稳态中位 TPOT|轮数 / 轮均接受|校验门禁|
|---|---|---|---|---|---|
|`a3-tp8-ep8`|Ascend910_9392 8 die，`[4,2,1]`|`57b05532`（零 vLLM）|5.5721 ms（最快 5.5679 ms）|141 / 7.255|8 卡主网 + 串行草稿 0-ULP|
|`a3-tp16-ep16`|Ascend910_9392 16 die，`[8,4,2,1]`|`0c2c5085`（Milestone A）|5.4512 ms（最快 5.3857 ms）|142 / 7.204|16 卡主网 + 串行草稿 0-ULP|
|`a3-tp16-ep16`|Ascend910_9392 16 die，`[8,4,2,1]`|Milestone B2（审计通过待合入）|5.0673 ms（最快 5.0467 ms）|143 / 7.154|16 卡主网 + 串行草稿 0-ULP|
