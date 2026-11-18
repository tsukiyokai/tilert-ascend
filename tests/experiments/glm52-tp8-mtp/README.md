# TP8 MTP 解码实验

范围：A3 Ascend910_9392，GLM-5.2 W4A8C8 TP8+EP8，`case-01-coding`，8192 输入、1024 输出。目标与当前证据见 [路线](../../../docs/speculative-decoding.md)。

|入口|作用|
|---|---|
|[prepare_host.sh](prepare_host.sh)|创建独立容器，复用只读权重和运行时；`TILERT_TOOLKIT_ROOT` 选择 CANN 9.2/ATB 根目录|
|[start_daemon.sh](start_daemon.sh)|启动 8 rank，控制文件限定在实验目录；`TILERT_SHMEM_LIB_DIR` 选择规约库|
|[build_reduce.sh](build_reduce.sh)|构建具有独立内核符号的容量修复库，保留原库|
|[check_topk.py](check_topk.py)|以实际 vendor 源码和 CPU Torch 检查草稿索引接口，含移除修复的负对照|
|[read_pairs.py](read_pairs.py)|读取五组交替运行，直接核对黄金词元、八卡主网和独立串行草稿状态，再计算未四舍五入的 TPOT|
|[run.json](../../.artifacts/runs/glm52-tp8-mtp-20261111/run.json)|112 机器的首次基线、原始 MTP 结果与规约修复证据|
|[run-capacity.json](../../.artifacts/runs/glm52-tp8-mtp-20261111/run-capacity.json)|138 机器的新库整模型验证记录|

CPU 索引检查：`TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/experiments/glm52-tp8-mtp/check_topk.py --runtime tests/.artifacts/runs/glm52-tp8-mtp-20261111/topk/runtime/vllm_ascend --output /tmp/glm52-topk.json`。输入使用该次部署的 runtime212 快照，检查实际 get/update 与上下文代理方法，不执行 NPU 前向。

设备任务顺序：`job_init_78l.py` → `job_prefill_case.py` → `job_bench_case.py` → `job_mtp_prepare.py` → `job_mtp_width_check.py` → `job_mtp_probe.py` → `job_mtp_graph_check.py`。任务文件位于 [scripts](../../glm52/megakernel_tp8_ep8/scripts)。每步先检查本轮完成记录及 8 rank 错误，才提交下一步；运行中断后以进程状态判定是否结束。

`sys._mtp_probe` 可指定 `widths`、`hybrid` 与 `output_dir`。精度诊断与图捕获在计时外执行，性能比较需要同机交替重复；原始单次结果不作为稳定加速结论。固定 TopK 分组的 W8 图执行在同机五组交替测量中位数为 9.69 ms/token，八卡词元、主网和独立串行草稿状态均对齐；尚未达到 TPOT < 3.0 ms。

配对汇总：`python read_pairs.py <paired目录> <mtp-state-narrow目录> <bench_result.json> --require-warm-head --output <汇总路径>`。每组目录为 `pair-0-variant-0` 至 `pair-4-variant-1`；0 是对照，1 是候选。实际计时起点须同步，图捕获与状态摘要在计时外完成。
