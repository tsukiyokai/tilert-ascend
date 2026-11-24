# 主机候选只上传一次

`ff2ab6c` 的 host-drafter 轮把候选逐个 `as_tensor(device)` 后 `stack`，验证后又对同一候选重复上传，与预测拼接并下载。现在纯主机整数批量构造一个 Tensor，候选保留在主机，线性与树轮只读取预测；模型起草仍读取候选和预测两组设备结果。单组读取省去 `cat`，不新增异步主机缓冲。

## CPU 差分

macOS arm64、Python 3.11.15、torch 2.9.0、单线程；同一个递增标量目标运行实际 `MtpRoundExecutor` 与 `CannModel.host_tokens`，3 批预热，30 对交替样本，每样本 100 轮，不剔除样本。计时包含 token 准备、目标的 CPU Tensor 加法、接受与状态更新，不含请求初始化。

| 验证宽度 | 原轮 CPU p50，µs | 新轮 CPU p50，µs | 配对节省 p50，µs |
| -- | -- | -- | -- |
| 8 | 55.502 | 15.714 | 39.477 |
| 16 | 93.611 | 19.174 | 74.628 |
| 32 | 170.255 | 26.839 | 143.364 |

width32 的真实 torch 调用从 `64 as_tensor + 2 stack + 1 cat` 变为 `1 tensor`；候选、预测及接受 token 完全相同。[原始样本](cpu-benchmark.json)保存命令、源码身份、p90/p99/std 与调用计数。这是 CPU 实测；A3 H2D 行为、设备同步和整模型 TPOT 仍须真机差分验证。

```bash
PYTHONPATH=. python3.11 experiments/token-staging/bench.py \
  --baseline-ref ff2ab6c \
  --output /tmp/token-staging.json
PYTHONPATH=. python3.11 tests/glm52/check_token_staging.py
PYTHONPATH=. python3 tests/glm52/check_lookup.py
```

7 项 token 检查覆盖实际 CPU Tensor 构造、混合设备 scalar、非法主机值、单/双组读取及三种起草器的串行一致性；10 项 lookup 检查通过。另通过状态机、round、TP16 scalar dispatch、session 和宽度不变性检查，覆盖 EOS、取消、尾部预算、复用输出缓冲与树路径 compact。对应 JSON 保存在本目录。`check_lookup` fixture 补齐已有 validator 所需的 `layers=None`。

## 后续设备验收

在同源 CANN 环境先做单 die `device_tokens`／`host_tokens` payload 与流顺序检查，再在 TP16 width32/lookup 的完整请求做配对差分，绑定 Python 与实际 ELF 身份；保持原输入、接受数、数值门禁和计时边界。当前本机没有实验别名与专用密钥，直连已知服务器的 SSH 被关闭，尚未传送或执行设备候选。

## 外部机制线索

[SGLang #35207](https://github.com/sgl-project/sglang/pull/35207)把与接受结果无关的主机准备提前，并跳过无消费者的 D2H 同步；它支持检查数据是否已在主机、是否确有消费者这一方向。其报告的速度不能迁移到本项目。[SAM-Decoding](https://arxiv.org/html/2411.10666v2)提供增量后缀自动机检索机制，现有自动机仍按独立工作负载比较接受数。知乎对「昇腾/PIPE_S」和「低时延/推理」的检索未给出本次 token 传输的直接证据。
