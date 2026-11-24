# Token staging 的 A3 对照

在 `Ascend910_9392` 的物理 die 0 执行现有 token staging 改动。
[payload 检查](device-check-v3.json)的九种宽度、四代变化输入共 36 例通过，
覆盖整数批量上传、混合设备 scalar、INT32 转 INT64 和独立 stream。

配对实验执行实际 `MtpRoundExecutor`、`device_tokens` 和 `host_tokens`；
目标计算是 NPU 整数加一，草拟器是确定性递增序列。该实验测量 token
准备、设备加法、下载、接受及状态提交，不执行模型权重、attention 或 TP。
旧臂取 `ff2ab6c` 的两个实际模块，新臂取 `4d42381`；两臂共用当前
`MtpState`，文件哈希随报告保存。每轮完整候选、预测、提交和缓存长度
均对照独立整数序列，再比较两臂摘要。

## 独立进程配对对照

每次运行的两臂各有独立 Python 进程、工作目录和五类缓存目录，
轮流执行 AB/BA，10 对预热、60 对测量，不剔除样本。每个计时轮之前
读写完整 256 MiB 缓冲并同步；压力操作在计时区间外。计时使用同步前后
的主机 `perf_counter_ns`，关闭 profiler 与 graph capture。预设假设为
width32 的配对 p50 节省至少 100 µs。

| 运行 | 宽度 | 原轮 p50，µs | 新轮 p50，µs | 配对节省 p50，µs | 更快对数 |
| --- | --- | --- | --- | --- | --- |
| [v1](staging-pair-v1/report.json) | 8 | 837.317 | 207.023 | 630.240 | 60/60 |
| v1 | 16 | 1384.483 | 216.433 | 1166.150 | 60/60 |
| v1 | 32 | 2476.944 | 235.484 | 2241.135 | 60/60 |
| [v2](staging-pair-v2/report.json) | 8 | 814.161 | 233.564 | 588.247 | 60/60 |
| v2 | 16 | 1377.412 | 213.443 | 1164.890 | 60/60 |
| v2 | 32 | 2365.070 | 236.479 | 2127.836 | 60/60 |
| [v3](staging-pair-v3/report.json) | 32 | 2474.199 | 239.239 | 2228.885 | 60/60 |

首两次共 360 对；第三次预先创建全部缓存目录后，仅复测 width32 的
60 对，收益保留。首两次绑定[冻结脚本](source-v1/device_bench.py)，
第三次绑定当前脚本。全部 420 对保留原始样本、顺序、签名及 p90/p95/p99/std。p50 差与配对
差的中位数分别计算，不能互换。第二次运行的 [占用采样](staging-pair-v2/occupancy.log)
含九个主机快照，NPU PID 均属于本任务容器，未发现其他 NPU 进程。
每次运行的两臂均记录实际映射的 116 个 CANN/Torch NPU 动态库；路径与
字节哈希集合一致，没有生成独立编译缓存文件。

环境为 [CANN 9.1.0 镜像](container.json)、Torch/Torch NPU 2.12.0、
驱动 25.5.5、[固件 7.8.0.9.220 与 9392 板卡](board.txt)。容器设置
`ASCEND_VISIBLE_DEVICES=0` 和 `ASCEND_RT_VISIBLE_DEVICES=0`。
初始两种非特权容器均在 `aclInit` 前因驱动命名空间占用失败，原始
[首次回执](device-check.json)和[第二次回执](device-check-v2.json)保留；
改用本任务的特权容器后通过，没有停止其他容器或修改驱动设置。

这是 A3 token staging 组件收益。高接受率来自明确的整数夹具，不作为
模型输出质量或接受率证据。正式模型精度、16 rank 状态、8×8K 输入、
实际请求 TPOT 与在线双时钟仍须独立验收；完整模型标志保持 false。

## 复跑

在新的实验目录导出旧臂的 `cann_model.py`、`mtp_round.py` 和完整
`commit.txt`；采用相同源文件与匹配 A3 环境运行：

```bash
python experiments/token-staging/device_check.py \
  --device npu:0 --output "$RUN/device-check.json"
python experiments/token-staging/device_bench.py \
  --baseline-root "$BASELINE" --source-commit "$SOURCE_COMMIT" \
  --output "$RUN/paired/report.json"
```

每个 `RUN` 使用新目录。容器保留在连续实验期间，实验结束后只清理
本任务的容器，并检查设备进程及 HBM 释放。
