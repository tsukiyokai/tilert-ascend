# FP32 head 的独立参考

[HF 采集器](../../../../tests/glm52/run_hf_reference.py)增加显式 `--fp32-head`：输入与有效 head 权重保留 BF16，FP32 积直接作为投影输出。`HfReference` 的最终 forward 与逐层诊断使用同一个 head；默认及非 head 投影保留原舍入。FP32 模式拒绝非 BF16 hidden 和量化 head。

BF16 `x=[1,1]`、`w=[1,2^-8]` 的精确积为 `1.00390625`，BF16 输出舍入为 `1.0`。将后者再转 FP32 仍为 `1.0`，因此旧 BF16 trace 无法代替新的 FP32 真值。[原始 CPU 数值与位模式](head-cpu.json)使用 32 个不同词表行、TP16 输出切分和独立标量公式，实际执行 raw reader 与 head 适配器。另一合成词表的 BF16/FP32 贪心结果分别为 0/17，说明模式也属于输出语义；真实模型的模式间输出一致性仍须验收。

manifest 记录 `fp32_head` 和 `weight_arithmetic.lm_head` 的输入/权重/累加/投影输出 dtype、TP16 与词表范围。FP32 准入由调用方声明模式，同时绑定采集器、HF 适配器、权重读取器三个当前源码哈希，及两个实际导入类的路径/哈希；缺失、过期或模式不符均拒绝。采集器在生成 trace 前核对实际来源，在初次执行、reset 与文件读回时检查全部 head 输出为 FP32。dtype 比较界与原精度阈值保持原约定。

## 已执行的 CPU 检查

[43 项回归](head-cpu.json)包括十项 head 检查、11 项标量 verify 检查与 22 项 Native 门禁夹具；另有 [62 项纯元数据检查](metadata-cpu.json)。head 检查覆盖真实 BF16/F32 数值、原始 BF16 权重、跨词表分片的贪心差异、非 head 隔离、非法输入与量化 head、CLI、78 个输出位置的 dtype 变异及实际类文件绑定。环境为 CPU Torch 2.9.0、Transformers 5.12.0、Python 3.11.15。

头部单测的类来源与 78 位置变异分别采用真实类对象和合成 trace。[完整合成 CPU 运行](../hf-runtime-20261028/README.md)另已实际执行 78 层构造与 forward、最终/77 层诊断 head、reset 和文件读回，并核对仓库内匹配的 C8 源码。完整生产 checkpoint、Prolog-v3、CANN FP32 caller 与实际 graph 数值继续验收。`precision_verified` 与 `formal_precision_gate_passed` 保持 false。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests/glm52 python3 -m unittest \
  check_reference_head check_cann_verify check_precision_gate
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \
  tests/glm52/check_hf_reference.py --output /tmp/hf-metadata.json
```

第一条需要 CPU Torch、safetensors 和匹配的 Transformers；第二条只需标准库。

## 后续数值接线

在具备完整 checkpoint、冻结 prefix/teacher 与匹配 C8 源码的 CPU 环境，分别生成 BF16 与 FP32 参考。`TEACHER_TOKENS` 是该前缀已固定的教师 token 数组，输出目录独立：

```bash
PYTHONPATH=. python3 tests/glm52/run_hf_reference.py \
  --checkpoint "$CHECKPOINT" --prefix "$PREFIX_DIR" \
  --tokens "${TEACHER_TOKENS[@]}" --generate 1 --capacity 9344 \
  --mla-preprocess prolog_v3 --source-commit "$SOURCE_COMMIT" \
  --fp32-head --output "$RUN_DIR/hf-fp32"
```

现有 CANN 收集器尚未显式接入 FP32 参考模式。接线时从实际 `target.fp32_head` 向 validator 传值，并把同配置的实际 graph 快照直接对照 HF；两个近似对照通过不能推出相同误差界下的传递通过。完整逐层证据、八个冻结输入与常驻 P/D 仍按 [共同协议](../measurement.md)推进。
