<img src="../assets/logo.png" alt="TileRT" width="96">

# TileRT-Ascend

TileRT-Ascend 是 Ascend 上的专用低时延 decode 后端。它接收成熟服务栈交来的状态，继续生成 token，并在声明的模型与配置范围内验证数值、状态和完整解码时间。GLM-5.2 / A3 是首条验证路径，正在完成整模型精度、常驻服务和执行片段的集成。

单请求下，计算可能很短，发起、搬运、同步和交接仍需付出完整成本。项目从 token 的关键路径出发，让静态权重提前到位，让结果在可用时交给后继，让通信与计算按数据依赖推进。每接管一段执行，都检查完整请求收益与维护成本，再决定是否扩大到相邻算子、层或设备。

[开始验证](#%E5%BC%80%E5%A7%8B%E9%AA%8C%E8%AF%81) · [实施计划](plan.md) · [工程教训](lesson.md) · [实验记录](probe.md)

## 谁可以采用

首条路径面向已有 Ascend 服务栈、能提供常驻资源的团队，优先验证低并发且 decode 占主要等待时间的请求。使用真实 GLM-5.2-W4A8C8 权重，一台 A3 的 16 个 die（计算芯片）做 16 路张量并行（TP16），一次处理一条请求。

| 参与者 | 分工 |
| -- | -- |
| 服务栈 | 处理请求路由、准入和输入计算（prefill） |
| Decode 后端 | 生成 token，组织投机验证、提交与回滚 |
| 执行实现 | 按约定的布局、数值与生命周期完成计算和通信 |

当前整模型路径使用 [CannModel][cann-model]，复用 vLLM-Ascend 的模型、量化和 CANN 算子。[CANN P/D 入口][cann-pd] 将它接到 TP 命令循环与 HTTP；[Ascend connector][connector] 和 HIXL 交接历史状态缓存（KV cache）。框架内适配与独立 decode 池计划沿用同一执行核心，按接入需求验证各自的生命周期。

其他团队应能复现完整请求，再替换一个执行片段。首个采用案例检验接入成本、输出质量、请求完成时间与每卡吞吐；后续采用案例决定哪些接口值得通用化。通用机制成果沿途回流全部 GitCode `cann/*` 仓库、vLLM-Ascend、TileLang-Ascend 与 TileRT。任务与验收顺序见 [plan](plan.md)，正交探索档案与时间账本见 [probe](probe.md)。

## 已验证到哪里

TPOT 表示平均输出 token 间隔；M1、M12 表示目标模型一次处理 1、12 个位置。原始配置与读数汇总在 [probe](probe.md)。

| 路径 | 结果 | 范围 |
| -- | -- | -- |
| [功能参考][smoke] | 1,729 项检查 | 四层合成 BF16；CPU / 单 die |
| [真实 P/D][pd-live] | 8K 输入，HTTP 返回 8 token | 16 路 HIXL WRITE，CANN M1 |
| [冻结前缀 decode][width32] | TPOT 中位数 12.413 ms | 同一 8K 前缀，1024 输出，TP16 |
| [原生全层组合][native-tree] | 变化输入与树规约检查通过 | 三个设备 rank，M1 / M12 |

功能参考的 NPU 路径使用 TileLang RMSNorm，真实 W4A8C8 与 TP16 另行验收。P/D 记录先停止 prefill，再加载 decode，验证了活设备 KV 交接；常驻服务尚待验收。

12.413 ms 来自 width32 lookup、SFA、编译目标图、SuperKernel 和 FP32 head，三次预热、三次测量。内部回调计时排除 HTTP/HIXL，原记录的精度与正式验收标志均为 `false`。原生组合使用合成 TP16 形状权重，另 13 个 CPU 进程只提供构造信息；真实 checkpoint、TP16 设备规约及整模型计时尚待验证。

`load_backend("glm5_2_ascend")` 加载 BF16 功能参考 `DecodeEngine`；`NativeDecodeEngine` 用于另一条 CANN 数值参考路径。整模型性能实验使用 `CannModel`，整模型常驻 tile runtime 仍是待验证的实现方向。

## 开始验证

### 本机检查状态与布局

在仓库根目录使用 Python 3.11 或更新版本运行，只需标准库：

```bash
report_dir="$(mktemp -d)"
PYTHONPATH=. python3 tests/glm52/check_mtp_state.py --output "$report_dir/mtp-state.json"
PYTHONPATH=. python3 tests/glm52/check_mtp_round.py --output "$report_dir/mtp-round.json"
PYTHONPATH=. python3 tests/glm52/check_ascend_layout.py --output "$report_dir/ascend-layout.json"
```

三条命令均应以状态码 0 退出，并写出通过报告。它们分别检查多 token 预测（MTP）的状态不变量、真实轮调度配合标量参考的行为，以及用张量元数据模拟的 Ascend KV 布局。NPU 算术与 HIXL 传输使用下面的 A3 入口验证。

### A3 整模型验证

准备完整 GLM-5.2-W4A8C8 checkpoint、tokenizer、MTP 权重和 8K prefill 快照。[统一启动脚本][launcher] 使用 CANN 9.2、NNAL/ATB、配套的 torch/torch_npu、成对固定的 vLLM/vLLM-Ascend 源码树，以及 `custom_transformer` 算子 overlay。P/D 另需可用的 HIXL bridge。依赖身份和参数随结果保存。

| 要验证的内容 | 入口 |
| -- | -- |
| 前缀恢复后的 M1 数值与状态 | [bench_cann_decode.py][bench-decode] |
| 投机、图模式和内部 decode 时延 | [bench_cann_mtp.py][bench-mtp] |
| Prefill、KV 交接与 HTTP 输出 | [prefill 启动][prefill]、[CANN P/D][cann-pd] |
| 原生片段的算术与图回放 | [TP layer 契约][tp-layer]、[三 rank 图检查][native-tree] |

每次实验使用新的 `TAG` 和独立输出目录，启动脚本会清空同名结果与编译缓存。按 [测量协议][measurement] 先检查恢复后的状态与输出，再测时延。冻结的是历史前缀，生成期间 KV 仍追加。P/D 按 [状态协议][pd-state] 验收；在线入口只开放已实现并验证的选项。构建与排错见 [调试入口](lesson.md#%E8%B0%83%E8%AF%95%E5%85%A5%E5%8F%A3)。

## 从哪里替换实现

执行片段可以跨算子，也可以包含通信；它须明确生产、发布、消费与存储复用的次序。工具链按实际 shape、生成代码和完整路径测量选择，可使用 TileLang-Ascend、AscendC 或 PTO。

| 接入部分 | 先读取的约定 |
| -- | -- |
| 服务框架或传输 | [KV 布局][layout]、[P/D 状态][pd-state]、[MTP parent][mtp-parent] |
| Attention | [indexed attention][attention] |
| MoE | [W4A8 routed][routed]、[routed/shared 适配][routed-shared] |
| 跨卡执行 | [通信协议][communication] |

服务接入核对版本、rank、完成回执、pending token、取消和槽位复用。Attention 核对稀疏选择、main/indexer 页映射、当前 stream、scratch 与图捕获生命周期。MoE 核对真实权重布局、原 top-8 slot、量化、combine 和 TP finalize 的归属。跨卡实现核对算术树、数据可见性、设备 epoch、窗口独占和驻留条件。

每次替换依次检查局部数值、变化输入图回放、实际 TP16、整模型输出和服务请求。比较完整被替换路径的耗时，同时记录资源占用与失败行为；收益成立后再扩大执行范围。

## 证据与参考

[evidence/](evidence/) 保存原始回执、测量、审计与协议，按日期和配置读取；[reference/](reference/README.md) 保存执行机制与外部资料的研究笔记。

[attention]: ../tests/glm52/whole_layer/attention_indexed.md
[bench-decode]: ../tests/glm52/bench_cann_decode.py
[bench-mtp]: ../tests/glm52/bench_cann_mtp.py
[cann-model]: ../tilert/models/glm_5_2_ascend/cann_model.py
[cann-pd]: ../tests/glm52/serve_cann_pd.py
[communication]: ../tests/glm52/whole_layer/communication.md
[connector]: ../tilert/pd_vllm/ascend_connector.py
[launcher]: ../tests/glm52/launch_arm_unified.sh
[layout]: ../tilert/pd_vllm/ascend_layout.py
[measurement]: evidence/ascend-decode/measurement.md
[mtp-parent]: evidence/ascend-decode/mtp-parent-handoff.md
[native-tree]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-tree-v1-a3-194/AUDIT.md
[pd-live]: evidence/ascend-decode/pd-live-20261017/README.md
[pd-state]: evidence/ascend-decode/pd-ascend.md
[prefill]: ../tests/glm52/serve_vllm_prefill.sh
[routed]: ../tests/glm52/routed_int4/README.md
[routed-shared]: ../tilert/models/glm_5_2_ascend/routed_shared.py
[smoke]: evidence/ascend-decode/README.md
[tp-layer]: ../tests/glm52/whole_layer/vendor_hook_tp_chain/README.md
[width32]: evidence/ascend-decode/width-ladder-20261023/w32-lookup.json
