# SFA 元数据预搬运：数值通过，时延退化

在 A3 `Ascend910_9392` 的物理 die 0，以 CANN 9.1.0、Torch/Torch NPU
2.12.0 编译并执行[源码候选](../../../../experiments/sfa-metadata/README.md)。
候选把每个 AIV 的 top-k 分片和页表搬入现有 UB 空区，保持 KV pair
顺序、因果界、MMAD 和原同步。新增两次 DMA 及配对 `MTE2_S(1)`，
UB 分配不增加；未安装全局 overlay，也未改变默认实现。

预设假设为 width32 同步墙钟 p50 至少改善 5%。实际配对结果否定该假设，
保留原实现，结束本轮预算 1。减少 GM 标量读取本身不能证明关键路径收益；
新增搬运、等待及生成代码各自的贡献尚未拆分。

## 数值与实际产物

[控制组](gate-control-v3.json)和[候选组](gate-candidate-v1.json)分别在
width1/8/32、8K、ACLGraph 下执行四代变化输入。每组 60 个边界对照均
逐位一致，覆盖 query、query RoPE、完整 main/RoPE KV 和 attention。
两组均对照同输入 CANN 链。输入包含排列页表、正反索引及变化 hidden；
本项使用合成 TP16 分片尺寸，实际只有一个设备 rank。

| 产物 | 实际加载库 SHA256 |
| --- | --- |
| [控制](control-provenance.json) | `126bfc0353c0c92e3f5895d446605303f6bd93fe1ac78348d2e3dc05a6631aaf` |
| [候选](candidate-provenance.json) | `1500756b71a61b4538122492b9c9d6c4b9bb2cf70a606ca7dc29efb126d39dc2` |

编译清单保留当时的源码身份；配对程序另外记录实际导入的新 fixture
源码身份。两个早期控制组启动分别因覆盖 CANN Python 路径和未创建
`ASCEND_CACHE_PATH` 失败；修复启动环境后通过，没有改动算术。

## 同机隔离进程配对

[原始报告](pair-v1/report.json)保留 10 对预热后的全部 30 对 AB/BA。
每臂独立进程和缓存目录；每对恢复完整原始 KV，变化输入、top-k 和页表，
核对输入摘要相同。每个计时结果都继续对照实际 CANN 的五个边界，
并要求两臂输出摘要一致。没有剔除样本。

| 时钟，µs | 原实现 p50 | 候选 p50 | 配对节省 p50 | 更快对数 |
| --- | --- | --- | --- | --- |
| 同步主机墙钟 | 402.370 | 489.194 | −87.504 | 1/30 |
| NPU event | 317.890 | 408.790 | −87.510 | 0/30 |

墙钟包含两次 event 记录与最终同步，event 覆盖一次原生 Prolog+SFA 图回放。
每样本先读写 256 MiB 缓冲，置于计时区间外；这不等于实测 L2 已清空。
profiler 关闭。[15 次占用采样](pair-v1/occupancy.log)只发现本任务的
NPU PID。[独立读回](readback.json)重算配对中位数，并将性能假设标为
`rejected`；原始报告的 `status=pass` 表示程序和数值检查通过。

本项未运行真实权重、TP16 设备通信、完整 decoder 或在线 TPOT。
数值成功与性能失败均保留，不据 CPU GM 读取计数推算设备收益。
