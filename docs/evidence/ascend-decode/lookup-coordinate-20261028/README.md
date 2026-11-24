# 用上下文坐标保存 radix 边

[`LookupDrafter`](../../../../tilert/models/glm_5_2_ascend/lookup_draft.py)保留节点局部字典与 longest/latest 策略，边从复制的逆序 tuple 改为 append-only accepted context 中的 `(hi,length)`。插入与查询比较两个有界正向切片，相等时逆序也相等；不相等时从高下标向低下标寻找首个分歧。每次更新仍插入一个有已知续写的窗口，单次插入/查询的比较上界仍为 O(longest)。

例如 `context=[1,2,3,9,1,2,3]`、longest=3 时，旧位置的区间 `[0:3]` 与当前尾部 `[4:7]` 相等，latest=3，故提议 `context[3]=9`。当前尾部之后尚无 token，不入索引。context 列表扩容保留下标语义；边分裂 k 个 token 时，公共边保留旧 hi，旧子边的 hi/length 同减 k。压缩边任意非空前缀的 latest 仍是所有匹配且有续写的位置的最大 end。

## 语义验收

正式实现通过 [79 项 CPU 回归](checks-cpu.json)，覆盖宽度不变性、缓存、TP 命令、16 线程和实际 HTTP 请求；[独立审查](audit-cpu.json)另执行 24159 次提议、7651 组小上下文配置、217685 个逐边前缀检查与 126 个实际 scalar session 的容量/EOS 场景。上下文、完整提议、命中计数和输出接口沿原约定，范围外整数与最长 96 的尾部也按原整数语义比较。

审查包含错误压缩负例：`[1,2,7,9,2,8,2]`、longest3/shortest1 的正确提议为 8；非法合并边后，完整边终点的 latest 仍正确，边内部前缀却退到旧位置并提议 7。逐前缀审查实际拒绝该状态。

此前分别[审查局部坐标、lazy children 两个原型](prototype-audit.json)，每类 4737 个 context、23685 次独立扫描提议、1405342 个任意前缀 latest 检查，并逐轮核对四条 8K+1024 轨迹。lazy children 单独省去空字典却拖慢复制/编辑/歧义输入；[全局 arena](../lookup-arena-20261028/README.md)也有复制负载退化，均未混入本次实现。

## CPU 成本

最终生产源码对 Git `019c97b` 的旧 tuple radix 做配对 A/B：macOS arm64、Python 3.11.15，固定四条 8192+1024 轨迹，width32、shortest2/longest32。每点两次预热、30 组交替配对；正常 GC 无 callback 的臂报告性能，阶段 callback 与仅测量窗口关闭自动 GC 的臂用于诊断。fresh 在每条样本前计时外 full collect，carry 保留自然跨样本状态。无 affinity/锁频，没有推理或 NPU 执行。

总成本为 setup 与全部 propose/commit 调用时间之和，排除 driver、teardown、I/O。下表为正常 GC、无 callback 的总成本 p50，单位 ms；每格为基线→坐标版。

| 负载 | fresh | carry |
| --- | --- | --- |
| 复制 | 8.348→7.812 | 8.335→7.785 |
| 5% 编辑 | 9.601→8.791 | 9.598→8.813 |
| 歧义 bigram | 9.963→6.819 | 10.064→6.760 |
| 无匹配 | 8.054→5.018 | 8.367→4.979 |

初始化、稳态与总成本 p50 均下降。上述指标的配对胜数均为 30/30，只有 carry 编辑负载的稳态/总成本为 29/30。完整 p50/p90/p95/p99/std/max 与全部退化保存在 [fresh](paired-fresh.json.gz)、[carry](paired-carry.json.gz)；每点 30 请求的最近秩 p99 就是最大值。

carry 编辑负载的稳态 p99 从 2.536 增至 3.771 ms、总成本 p99 从 9.893 增至 10.625 ms，单次 commit 最大值从 252.666 增至 1806.333 µs。正常无 callback 的这次长调用尚未归因；query 的部分高分位和少数最大值也有退化，均保留在 `candidate_regressions`。

[直接区间核对](independent-audit.json)记录了另一观测臂的 candidate sample18、position686：1.733583 ms 的 commit 中，GC2 callback 区间覆盖 1.705708 ms，collected=0。它说明该调用墙钟与全代扫描的重叠，不回填无 callback 样本的原因。关闭自动 GC 的臂仍有长调用，分配、调度与工作量继续独立核查；生产 GC 配置不变。

最终二臂共有 1536 个原始样本；原型三臂另有 768 个。两个生产面板均绑定实际加载类路径、候选源码、基线 commit/blob、精确参数和当前 HEAD，区分当时未提交的候选。`*.samples.jsonl.xz` 保存全部逐轮原始数据，[独立读回](independent-audit.json)重算全部签名、时间加总、总成本 p50、配对胜数、GC 计数、压缩前后 SHA；[原型冻结源](sources/)随凭据保存。

## 复跑

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests/glm52 python3 -m unittest \
  check_lookup check_lookup_benchmark check_round_invariance \
  check_mtp_round check_mtp_session check_mtp_tp check_cann_lookup_threads
PYTHONDONTWRITEBYTECODE=1 python3.11 experiments/lookup-index/audit.py \
  --root "$PWD" --output /tmp/lookup-coordinate-audit.json
PYTHONDONTWRITEBYTECODE=1 python3.11 experiments/lookup-index/coordinate_probe.py \
  --root "$PWD" \
  --reference docs/evidence/ascend-decode/lookup-policy-20261028/rfind-cpu.json.gz \
  --gc-start carry --output /tmp/lookup-coordinate-paired.json
```

设备按 [单 die→少 rank→空闲 A3](../online-lookup-20261028/README.md)继续验收：相同 checkpoint、输入、编译/overlay 与加载身份下比较完整输出及内部/客户端时间。CPU proposer 成本与整模型精度、正式 CANN 门禁、在线 TPOT 分别记录。
