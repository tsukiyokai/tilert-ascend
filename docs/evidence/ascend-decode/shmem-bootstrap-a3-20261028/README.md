# A3 SHMEM 通信环境验收

在同一 `Ascend910_9392` 主机，以 CANN 9.1.0 从源码构建 SHMEM
`24c15d21c26972404b0bb0e9cfd08d03930d15c6` 及当前原始通信探针。
库只放在本任务目录。运行前检查 NPU 进程；新容器绑定空闲的 die 0–3，
TP2 使用 0/1，TP4 使用 0/1/2/3。

| 实际设备 rank 数 | 行数 | 每形状变化回放 | 主对照 |
| --- | --- | --- | --- |
| [2](bootstrap-tp2/ranks/) | 1 | 4 | 升 rank FP32 求和→BF16，CUBE 消费，再同序求和 |
| [4](bootstrap-tp4/ranks/) | 1/12/32 | 各 4 | 同上 |

各 rank 的两次规约与 CUBE 结果均通过独立 CPU 算术对照。回放轮换延迟
不同 rank，设备 epoch 连续推进，图销毁及 stream 同步后才释放窗口。
TP2 的首规约另与该次实际 HCCL 逐位相等；TP4 回执保留 HCCL 差异字段，
其验收主对照仍为源码声明的升 rank FP32 顺序。

[构建日志](build.log.gz)、[控制构建](control-build-v1.log.gz)、逐 rank
回执及[独立读回](readback-tp4.json)绑定实际库、已映射 SHMEM、源码及
设备身份。它们证明本次运行环境与原控制链可执行；FP32 TreePlan、
RSAG、真实 TP16 及性能另行验收。本次使用 `--no-timing`，没有时延收益
或模型 TPOT 结论。
