# A3 RSAG 数值门禁

在 `Ascend910_9392`、CANN 9.1.0、SHMEM `24c15d21` 上，比较同一两次规约+CUBE probe：控制全互发后执行 FP32 TreePlan，候选将 512B 数据行交给 owner 求同一树再广播。算法与窗口协议见[实验入口](../../../../experiments/communication-rsag/README.md)。

| rank 数 | rows | 两臂 rank 报告 | 数值记录 | 跨臂摘要比较 |
| --- | --- | --- | --- | --- |
| 2 | 1/12/32 | 4 | 108 | 54 |
| 4 | 1/12/32 | 8 | 216 | 108 |

324 条记录的输入、权重、首规约、中间 CUBE 消费后的第二规约均与独立 CPU 树逐位一致，输出前后各 512B 哨兵完好；162 条跨臂的四张量 SHA256 一致。含 eager、变化输入 graph、轮换延迟 rank，另有每形状连续八次 graph replay，中间无全 rank barrier 或设备同步，覆盖双 parity 复用。

[readback.py](readback.py) 从原始报告重验数值、跨臂摘要、冻结源码、manifest、build report 与 ELF 身份；[读回结果](readback.json)保留验收范围。`raw/source/` 是编译/门禁时的原始八文件快照，其 README 的待测描述属于当时状态。两臂独立进程和缓存，运行前后设备进程快照在 `raw/receipts/rsag/`。

随后在全部 16 个实际 9392 die、blocks24 下，两臂 32 份 rank 报告的默认门禁通过。连同两次 TP4 与 TP2，累计 1,404 条默认数值记录、702 条跨臂摘要一致；额外 960 条特殊输入跨臂记录的两次原生输出摘要也一致。[读回](readback.json)逐项重验；初次 TP2/TP4 的独立结果保留为 [v1](readback-v1.json)。

HCCL 对照是单独观测：`HCCL_OP_EXPANSION_MODE=AIV`、`HCCL_DETERMINISTIC=true`，FP32 all-reduce 后只在末端转 BF16。四 rank 的 `[1,2^-25,-1,0]` 见证在 rows1/12 匹配固定树，在 rows32 匹配串行；16 rank 的该见证在三个 rows 均匹配固定树。这反证了固定树可覆盖所有 shape/rank 的 HCCL 顺序。finite 夹具在 CUBE 后、edges 在首规约还有共同的 CPU 差异；两臂相同不消除这些差异。顶层 `pass` 仅覆盖默认 tagged/random 门禁，特殊夹具仍是观测。

本组数值入口没有性能测量。配对时延另记；有限夹具相等不构成全 BF16 语义证明，也不构成模型精度或 TPOT 结论。

## 首轮 TP16 计时未通过验收

三轮 ABBA、6 对独立 world；每 world/shape 预热10次、测30次，每样本捕获连续32次完整双规约+CUBE，再按32均摊。每样本换输入、区间外256MiB读写压力；每 rank 原始 wall/event、全 rank 提交/完成时刻、输出/哨兵、缓存路径和进程快照均保留。以下是未通过准入的原始 p50，单位 µs/完整 probe：

| rows | host 控制→候选 | event 控制→候选 | 最大提交包络/32 ÷ 控制 event p50 |
| --- | --- | --- | --- |
| 1 | 144.174→185.377 | 114.947→159.313 | 138.47% |
| 12 | 493.572→439.855 | 468.498→410.782 | 30.88% |
| 32 | 1018.000→826.369 | 990.009→799.537 | 13.03% |

预注册要求所有形状、双时钟 p50 至少改善10%、p95不回退、至少30对 world，且最坏提交包络比例≤1%。本次 rows1 退化，三个形状的包络均超标；保留全部样本，不改阈值，不认定性能收益。连续32次中只有首调用受压力条件化，这些均摊值不称为32次独立冷调用。

[replay_pilot.py](replay_pilot.py) 解开逐 world 原始归档，用冻结的实际统计源码重算所有 rank 时钟、身份、数值与占用准入；[读回](readback-pilot.json)复现 `accepted=false`。实际 world6 的16进程映射另保留122个库的路径与计时后文件SHA，CANN运行库来自9.1；9.2的独立导入预检不参与本次计时。
