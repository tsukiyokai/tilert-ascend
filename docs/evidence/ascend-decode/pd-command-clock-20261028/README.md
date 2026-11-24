# P/D 命令与解码轮分别计账

`AscendMtpPDEngine`实际调用 `advance_mtp_n`，旧 `bench_pd_decode.validate_producer`只统计 `advance_mtp`，会把正常请求判为没有轮。修改前 [22 项 CPU 回归](before-cpu.json)出现六个失败和一个错误。

现在同时识别单轮和批命令，将每条命令的实际轮数、接受 token、匹配数与自然接受数绑定到 session 的逐轮 span，并核对输出数量和连续 cache 位置。单轮仍按一轮计算；批命令不能按一次调用算成一轮。汇总保留实际 `count`、`command_count`和平均接受数，时长改标 `p50_command_ms`/`p95_command_ms`；没有逐轮时间戳时不生成逐轮时延分布。

修改后 [24 项检查](after-cpu.json)全过：真实标准库 TP16 命令流程、8K 前缀与 1024 输出，分别执行单轮、默认批调用及每命令三轮；同时拒绝伪造轮数、接受 token、span 宽度/位置、统计计数和倒退时钟。NDJSON、双时钟、取消、reset 与准入身份检查保持通过。没有运行 NPU 或 CANN 模型。

```bash
PYTHONPATH=. python3 tests/glm52/check_decode_timing.py --output /tmp/pd-clock.json
```

本次保留 Native 精度 schema 与 CLI 的 1/2/4 宽度限制。CANN 需要独立的证据类型，不能靠忽略 `expert_backend`、包名或源码键的不匹配接入。当前 CANN verify 只检查默认 M2/M4；compiled MTP runner 可以缺少 teacher 与串行 phase；saved-HF comparator 的 `diagnostic`也不是精度通过。正式接线仍须同配置 HF/串行、width32、状态与完整输出的真实产物，并绑定八个独立 8K 输入、权重字节及实际加载的设备产物。
