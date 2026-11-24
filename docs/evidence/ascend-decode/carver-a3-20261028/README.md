# A3 Carver 资源描述核验

在实际 `Ascend910_9392` 上，旧描述将 SKU 归为 `Ascend910A`；候选先匹配完整 SKU，读取当前选定 CANN SDK 的对应 INI，再以当前后端 arena 裁剪 Carver 预算。9382 必须读取自己的配置，不借用 9392。原始容量与可用容量分别记录。

| 项目 | 旧描述 | SDK 9.1 原容量 | 候选可用预算 |
| --- | --- | --- | --- |
| L0C | 262144 B | 131072 B | 131072 B |
| L1 | 1048576 B | 524288 B | 524032 B |
| UB | 262144 B | 196608 B | 196352 B |
| L2 | 16777216 B | 201326592 B | 201326592 B |

实际 `Ascend()` 与真实 `AscendDefaultPolicy._check_l0_usage` 对 FP16 输入、FP32 累加的 `256×256×64` 分块从允许改为拒绝；`128×256×64` 仍允许。后者经完整 TVM/TileLang 原生构建与真实 A3 JIT 执行，三个变化输入的 `256×256` 输出均与独立 CPU 精确计算逐位一致，共 196608 个 FP16 输出元素。

[原生报告](raw/receipts/tilelang-profile/jit-128-256/report.json)、生成源码、基线/候选描述、所选 SDK 文件 SHA、九个仓库/子模块版本、构建日志与复现脚本均保留。未提交的安装包和编译库只保留[身份](binary-identities.json)。独立 CPU 测试 15 项通过；源码修正已提交 [TileLang-Ascend PR #1955](https://github.com/tile-ai/tilelang-ascend/pull/1955)，上游基线为 `43b0b697`。

两版实际 `recommend_hints` 均返回空列表。调试日志显示继承的 CUBE axis `[1,2]` 与二维 tile 不匹配，在容量检查前被拒绝；本补丁没有修复该问题，也没有证明自动选块改善。旧超容量分块仅做隔离编译，编译器进程退出 139，未执行设备 kernel。本组没有配对时延、模型精度或 TPOT 结论。

本组原生JIT先于PR的SDK重构。后续`233047cb`修复了新增测试在collection阶段提前初始化NPU的问题，CPU collection对照通过；当前head的原生CI状态见[独立记录](../carver-ci-20261029/README.md)。
