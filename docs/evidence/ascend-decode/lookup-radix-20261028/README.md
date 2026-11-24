# 精确压缩尾部索引

`LookupDrafter` 用压缩 radix trie 保存每个已知续写位置之前的逆序尾部，取代逐 token、逐尾长的多项式哈希更新。最长/最短尾部、最近匹配、上下文、计数器与调用接口保持原约定；比较原始 token，消除哈希碰撞遮蔽更长命中的情况。[实现](../../../../tilert/models/glm_5_2_ascend/lookup_draft.py)同时供离线与在线 lookup 使用。

## 语义与不变量

上下文长度为 n 时，只索引 `shortest <= end < n` 的位置；`context[end]` 已存在，才有已知续写。每个 end 插入一次 `reverse(context[max(0, end-longest):end])`，其前缀覆盖该位置的所有允许尾长。

每个非空 trie 前缀的 `latest` 是匹配它的已知续写位置的最大值。插入按 end 递增：完整经过的节点更新 latest；分叉或较短窗口将边拆开，旧子边保留旧集合，新公共前缀与新分支取新 end。压缩边内部没有分叉或终点，因此其中各前缀共享子节点的匹配集合。查询沿当前逆序尾部走到最长匹配前缀，取其 latest；在边中途停止也适用，故同时满足最长与最近。每次插入最多新增两个节点；尾长限制为 W 时，单次插入与查询的 token 比较成本为 O(W)。

合法词表反例：`A=[30000,30712,30000,32048,30503]`、`B=[30306,30000,30004,30000,30000]` 在旧 61-bit 哈希下相等。`A+[111]+B+[222]+A[-4:]+[333]+A` 的最长 5-token 匹配应续写 111；旧索引退到四个 token 并提议 333，新索引与独立扫描均提议 111。这是起草策略差异；生成仍由目标验证器决定。[原失败](before-cpu.json)。

## CPU 结果

[79 项回归](checks-cpu.json)覆盖独立扫描、宽度不变性、缓存、TP16 命令、session、HTTP 与 16 并发标量 worker。[独立审查](audit-cpu.json)另执行 24,159 次提议检查、7,651 组小上下文/尾长配置的节点不变量检查，以及 126 个真实 session 编排的容量/EOS 场景；包括 generator、空提交、重叠、压缩边中途匹配与深度至 96。[审查脚本](../../../../experiments/lookup-index/audit.py)使用独立反向扫描与串行目标，检查有效缓存逐行一致。

以下为 Git `a24dfd4` 的 tuple radix 对 Git `1ef8a8d` 哈希实现的本地配对测量：macOS arm64、Python 3.11.15；四个合成 8192-token 输入、各 1024 输出，width32、shortest2/longest32；两次预热，30 对交替样本，无 affinity/锁频。建立成本含共同首输出，稳态只计 propose/commit 调用；总成本按每个样本相加后取统计量。p95/p99 使用最近秩。目标推理、编排与 I/O 均在计时之外。

| 负载 | 建立 p50, ms | 稳态 p50, ms | 稳态 p95, ms | 稳态 p99, ms | 总成本 p50, ms |
| --- | --- | --- | --- | --- | --- |
| 纯复制 | 42.300→5.726 | 5.710→0.765 | 5.912→0.821 | 6.002→0.837 | 47.988→6.491 |
| 复制中 51 次编辑 | 42.550→5.741 | 7.373→2.238 | 8.046→4.708 | 10.227→4.888 | 49.921→7.969 |
| 歧义 bigram | 48.938→8.019 | 15.707→2.344 | 32.953→7.620 | 36.324→10.577 | 64.813→10.459 |
| 无匹配 | 48.627→6.600 | 15.370→1.838 | 17.882→13.993 | 18.341→16.081 | 64.122→8.590 |

四类输入的全部 240 个样本均与独立扫描的完整提议/接受签名一致；每类总成本 30/30 对下降，稳态分别为 30/30、30/30、30/30、29/30 对下降。无匹配输入仍有 16.081 ms 的稳态高尾；每个请求样本的 GC collection 数保存在原始记录，尚未定位这些回收与高尾的时间关系。所有计时加总、分位数、配对差值及测量源码哈希已读回核查。[摘要](paired-summary.json)、[完整样本](paired-cpu.json.gz)、[同 AST 的另一轮原始测量](repeat-paired-cpu.json.gz)与[对应冻结源码](repeat-source.txt)、[测速脚本](../../../../experiments/lookup-index/radix_probe.py)。[无索引 rfind 的退化结果](../lookup-policy-20261028/README.md)保留。

[全局 arena 与 GC 分阶段对照](../lookup-arena-20261028/README.md)未重现该高尾；坐标 arena 改善无匹配输入，却使复制和轻度编辑退化，未替换生产。保留局部字典、仅将边改为上下文坐标的实现另有[生产源码 A/B](../lookup-coordinate-20261028/README.md)。各记录绑定对应源码；callback、起点与报告保留方式分别记录，历史高尾的原因继续独立核查。

从仓库根目录复跑：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests/glm52 python3 -m unittest check_lookup check_lookup_benchmark
PYTHONDONTWRITEBYTECODE=1 python3.11 experiments/lookup-index/audit.py \
  --root "$PWD" --output /tmp/lookup-audit.json
PYTHONDONTWRITEBYTECODE=1 python3.11 experiments/lookup-index/radix_probe.py \
  --root "$PWD" --baseline-ref 1ef8a8d --output /tmp/lookup-paired.json
```

有界后缀结构的外部背景见 [Snowflake 工程文章](https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/)；本页的精确策略与 CPU 收益由上述实现和原始记录定义。

## 设备对照

沿用 [单 die→少 rank→空闲 A3 顺序](../online-lookup-20261028/README.md)。在匹配环境的独立 checkout 中，对 `1ef8a8d` 与本次实现使用相同 checkpoint、capture、overlay、编译选项与暖缓存条件，各自保存源码/实际 ELF 身份及完整输出；每个冻结输入分别运行：

```sh
torchrun --nproc-per-node=16 tests/glm52/bench_cann_mtp.py \
  --checkpoint "$CHECKPOINT" --capture-case "$CAPTURE_CASE" \
  --output "$RUN_DIR" --lookup --width 32 --target-graph \
  --compile-target --super-kernel --fp32-head \
  --warmup 3 --requests 32 --max-tokens 1024
```

本轮没有设备执行。A3 宿主收益、width32 数值、正式 CANN 精度门禁与在线 TPOT 仍须按 [共同测量协议](../measurement.md)验收。
