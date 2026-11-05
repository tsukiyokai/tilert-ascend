# 在线 lookup 的 CPU 接入验收

[CANN P/D 入口](../../../../tests/glm52/serve_cann_pd.py)支持每请求 lookup、width32、compiled target、SuperKernel、FP32 head 和 target graph。M1/MTP2/MTP4 保持原默认路径。lookup 不加载 MTP 模型，仍接收 v3 的 main P、MTP P−1 与 parent hidden，并用目标 head 校验 prefill 首 token。

每个 rank 的 WRITE 准入必须携带本次完整 `prompt_token_ids`；缺失或格式错误在占槽及 TP 广播之前拒绝。接收端复制为请求自己的 tuple，reset 清除请求引用。首次具有足够容量的注入执行三次 uncaptured warmup、全部 rank 会合、一次 capture/replay、恢复位置 P、再次会合；后续请求复用图。容量不足时执行实际尾部宽度。预热位于 token 交付之前。rank 回执及请求 timing 的 `execution.target_graph` 保存实际 capture 数、宽度和注入时 replay 数；`target_graph_requested` 单独记录配置。

## 已执行的 CPU 检查

[lookup/HTTP 七项](lookup-cpu.json)覆盖 CLI 到构造函数、零 MTP 加载、prompt 身份与 reset、首 token 拒绝、槽位复用、预热回滚及短容量。[原接收端 33 项](receiver-cpu.json)和 [TP 命令 17 项](tp-cpu.json)通过。

[16 线程回执](threads-cpu.json)执行真实 `RankExecutor.serve`、命令队列及 16 方 barrier，逐 rank 保存原始事件、输出和有效缓存。三个请求依次覆盖短容量、首次捕获、换 prompt 复用；17 方 barrier 缺少参与者的负对照必须失效并回收全部线程。目标算术为独立标量模型，graph 是观察顺序用的标记；这些检查未执行 NPU、ACLGraph 或 HIXL WRITE。

```bash
PYTHONPATH=. python3 tests/glm52/check_cann_lookup_pd.py --output /tmp/lookup.json
PYTHONPATH=. python3 tests/glm52/check_cann_lookup_threads.py --output /tmp/threads.json
```

第一条需要 CPU torch、FastAPI 与 HTTP 测试依赖；第二条只需标准库。

## 设备推进顺序

1. 在选定空闲 die 的匹配 CANN 环境运行 [payload/stream 检查](../../../../experiments/token-staging/device_check.py)：`PYTHONPATH=. python3 experiments/token-staging/device_check.py --device npu:0 --output /path/to/new-run/token-device.json`。脚本覆盖九种宽度、四代变化数据及混合 scalar，失败保存回执并以非零状态退出。该脚本目前只完成本机语法与 CLI 检查。
2. 选定少量空闲 die 逐个执行同一 payload 检查；跨 rank 算术与窗口生命周期沿用 [少 rank 设备契约](../../../../tests/glm52/whole_layer/vendor_hook_tp_tree/README.md)的独立验收。
3. 在空闲 A3 的 `trt-asc` 中，以新的工作/缓存/输出目录、匹配的 SDK/overlay、独立 P/D 资源运行 TP16 服务。P 端设 `TRT_MTP_PARENT_HIDDEN=1`、`TRT_DECODE_HOST=$DECODE_HIXL_IP`、`TRT_CTRL_PORT=5556` 并发送完整 prompt IDs，D 端的核心参数如下；环境与缓存规则见 [lesson](../../../lesson.md)。

```bash
torchrun --nproc-per-node=16 tests/glm52/serve_cann_pd.py \
  --checkpoint "$CHECKPOINT" --hixl-host "$DECODE_HIXL_IP" \
  --hixl-library "$HIXL_LIBRARY" --gpu-memory-utilization 0.85 \
  --lookup --mtp-width 32 --compile-target --super-kernel \
  --fp32-head --target-graph --timing-dir "$RUN_DIR/timing" \
  --output "$RUN_DIR/ranks"
```

设备验收读取实际 Python/ELF 身份、图回执和 16 rank 状态，先确认恢复、变化输入、EOS/取消、首 token 与完整输出，再进行同配置配对计时。常驻显存、真实 HIXL、CANN width32 数值与整请求 TPOT 均未验证。本机当前未取得 `trt-asc` 登录；没有设备任务在后台等待。

现有 `precision_gate.py` 接收 NativeDecodeEngine 的 MTP1/2/4 证据，`bench_pd_decode.py` 也绑定该 schema，不能给 CANN lookup/width32 签发正式验收。CANN 数值证据与配置绑定仍是下一项 CPU 可闭合的接线工作，当前 `precision_verified` 和 `performance_measured` 均保持 `false`。
