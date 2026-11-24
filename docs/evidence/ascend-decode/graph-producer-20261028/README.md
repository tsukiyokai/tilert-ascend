# 编译图内的持久观测槽

[`PersistentTrace`](../../../../experiments/graph-evidence/producer_probe.py)在 embedding、input norm、SFA 索引参数与 router 返回值边界写入持久 Tensor 槽，并为每个字段写入本次 epoch。begin 和 readback 在编译外运行；槽位地址不变，结果在下一轮覆盖前复制，buffer 保留到引用它的图销毁。输入 header 取自实际观测槽，预期 header 在调用前独立构造。

实验直接调用仓库的 `CannModel.verify`，以 CPU 模型和设备上下文替换 NPU 依赖，仅编译 backbone.forward，head/argmax 留在外侧。夹具为 78 层、width32、hidden4、vocab17、2048 列索引；索引与路由值实际影响后继状态。它检验观测的保留、新鲜度、行身份、缓存与复制所有权，没有执行生产 checkpoint、真实 attention、ACL capture/replay、SK 或独立 HF 精度验收。

## 执行与拒绝

[实际报告](report.json.gz)的 A/B/A 三次调用共 96 行、628 字段/帧，通过 60288 个字段行与原 `compare_step(exact=True)` 的 104832 项标量参照检查。该比较器未比较输出 token ID，此实验另作 96 项精确对照，包含 shape 与 dtype。借用的 hidden 输出被后续调用覆写，已复制的 A 帧在 reset 后保持精确一致。

Dynamo 图有 469 个 copy；AOT 把 234 次 stamp 更新合并为 `select_scatter`，最终保留 429 个输入写回：234 个观测槽、1 条 stamp 向量、193 个 KV 缓冲和1个复用 hidden 输出。图的完整节点、DCE 前后结构与代码随报告保存。

八个拒绝检查覆盖原 M1 trace 的 width32 限制、错误输出 token ID、浮点输出 ID、缺失字段 stamp、过期 replay、请求行置换、持久地址替换，以及实际删除 AOT stamp 写回。最后一项在删节点、DCE 和 recompile 后，再经同一个 compiled 调用执行 `engine.verify`；234 个 stamp 实际全部留为 `-1`，readback 拒绝。它验证了执行后的拒绝链，没有用节点计数代替结果检查。

## 生效配置与设备接入

本地 vLLM-Ascend `d52c1b8` 的配置构造器接收 `enable_super_kernel` 但不使用它；本地 torchair `ecf37e3` 在 SK 优化开启时明确跳过 optimized tensor dump。这两个反例只覆盖归档的本地源码，不能确定服务器版本或实际 SK 状态。后续须记录加载来源、生效配置及实际编译/执行结果。

实际 NPU 模型仍须检查 opaque attention/MoE 边界能否产出全部字段、流顺序、图池与 buffer 生命周期、SK 下保留行为，以及观测图与正式性能图的算术关系。按[单 die、少 rank、空闲 A3](../online-lookup-20261028/README.md)递进。正式图直接按原阈值对 HF，或建立覆盖相同边界的位等连接；两个近似对照不能替代这一步。`precision_verified`、`npu_executed` 和性能标志保持 false。

## 复跑

```sh
python experiments/graph-evidence/producer_probe.py --output /tmp/producer-new.json
```

需要 CPU Torch 2.9 与报告所列本地源码目录。源码文件、版本和完整报告均按[字节清单](manifest.json)保存，[读回记录](readback.json)绑定全部来源与拒绝结果；生产模型和 NPU 接入单独验收。
