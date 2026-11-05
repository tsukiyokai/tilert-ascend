# 一例 8K/32-row CANN → HF 精度入口

本入口只检查真实 TP16、78 层、8K 冻结前缀后的 32 个 teacher 输入。目标固定为 eager CannModel、SFA、FP32 head；M1×32、M32、M32 reset 三条执行均保存完整真实 logits/token IDs、79 个 hidden、78 层诊断 logits、实际索引/路由及全部新 KV。原 Native 门禁、数值阈值及模型源码均不修改。compiled graph、ACLGraph、SuperKernel、lookup 接受、在线 P/D、八个输入与 TPOT 不属于本入口的通过范围。

默认严格模式先校验原 HF manifest、实际 trace 字节哈希、checkpoint 指纹、captured prompt 与各 rank 的冻结前缀身份。M1 复用 CannTrace/CannChoices；M32 复用 experiments/graph-evidence 的 PersistentTrace，不绕过原 CannTrace 的 width 限制。每行直接调用原 compare_step 默认比较，绝不传 diagnostics；独立核对返回 token IDs、真实 logits argmax 与 HF argmax。任何 HF、M1/M32、prefix 或精确 reset 差异均使 precision=false，入口非零退出；诊断完成不等于通过。

`--capture-only` 可在 CPU HF 尚未完成时先采集相同的真实 M1×32/M32/reset 工件；该模式必须省略 `--reference`，不读取 HF 文件、不执行 HF 比较。checkpoint/teacher/prefix 准入、返回 ID 对自身 logits argmax、M1/M32 一致性、精确 reset、prefix/stamps 与各 rank 状态检查仍执行。完成采集只会得到 `captured_unverified` 或 `captured_with_failures`，并记录 `capture_complete`、`capture_checks_passed`；即使所有检查成功，`case_precision_passed`、`precision_verified`、`formal_precision_gate_passed` 也始终 false。退出0只表示未经独立HF验证的采集完成，执行检查失败则退出非零。原始逐行/宽帧、输入/源库身份及其 SHA 全部保留，后续可据相同 HF、teacher 与 prefix 身份离线严格比较，不需重跑 CANN；未实现离线比较时不得自行升级状态。

## 输入与运行命令

需要同一只读完整 checkpoint、由真实 P 生成的 v3 capture case，以及现有 teacher JSON。capture case 须包含 16 份 main-prefix/rank-N.pt 和完整 manifest；teacher 使用 frozen_teacher.load_teacher 的原格式，包含 32 个 teacher_tokens、32 个 expected_next_tokens、prompt_sha256、prompt_tokens、first_token_id。其 labels 只用于已有 teacher 身份/连续性核验；最终精度比较的独立标签取自 HF logits，不把旧 CANN 预测充作 HF oracle。

先在适用的 CPU HF 环境生成独立参考，`TOKENS_32` 为 teacher JSON 的完整 32 个输入 ID。C8 目录必须满足原参考采集器的源码哈希契约：

```bash
python tests/glm52/run_hf_reference.py \
  --checkpoint "$CHECKPOINT" --prefix "$CAPTURE_CASE/main-prefix" \
  --tokens $TOKENS_32 --generate 1 --capacity 9344 \
  --fp32-head --mla-preprocess prolog_v3 \
  --c8-source-directory "$C8_SOURCE_DIRECTORY" \
  --source-commit "$SOURCE_COMMIT" --output "$HF_OUTPUT"
```

HF 参考须有 78 层、32 个同序 teacher 输入、8192 起点、全部 32 份 trace、oneDNN disabled 和精确 reset 成功，且原 validate_reference_manifest 接受其实际加载源码及算术契约。与 CANN 不符时保留失败，不调整参考或阈值追随结果。

匹配的 A3/CANN/torch_npu/vLLM-Ascend 环境中，由根线程核查 16 个空闲 die、预先 source 所选 SDK，再执行：

```bash
torchrun --nproc-per-node=16 \
  experiments/cann-precision/run.py \
  --checkpoint "$CHECKPOINT" --capture-case "$CAPTURE_CASE" \
  --reference "$HF_OUTPUT" --teacher "$TEACHER_JSON" \
  --source-commit "$SOURCE_COMMIT" --capacity 9344 \
  --gpu-memory-utilization 0.85 --output "$PRECISION_OUTPUT"
```

输出目录必须是新目录。每 rank 保存实际输入/源库身份、完整 tensor 工件和逐项数值检查；rank0 summary 绑定 16 份报告 SHA。此命令不使用 profiler，不输出时延，不加载 MTP 模型。三条完整 trace 的 78 层 FP32 诊断 logits 本身约占 69GiB/TP16，另需 hidden、KV、选择与报告空间；不能把缺失层或截断词表静默视为完整证据。

HF 仍在生成时，先用独立新输出目录执行采集：

```bash
torchrun --nproc-per-node=16 \
  experiments/cann-precision/run.py --capture-only \
  --checkpoint "$CHECKPOINT" --capture-case "$CAPTURE_CASE" \
  --teacher "$TEACHER_JSON" --source-commit "$SOURCE_COMMIT" \
  --capacity 9344 --gpu-memory-utilization 0.85 \
  --output "$CAPTURE_ONLY_OUTPUT"
```

输入可先单进程检查，不构造 CANN 模型或初始化 NPU；输出另用新目录：

```bash
python experiments/cann-precision/run.py \
  --checkpoint "$CHECKPOINT" --capture-case "$CAPTURE_CASE" \
  --reference "$HF_OUTPUT" --teacher "$TEACHER_JSON" \
  --source-commit "$SOURCE_COMMIT" --check-inputs-only \
  --output "$INPUT_ADMISSION_OUTPUT"
python experiments/cann-precision/check.py -v
```

CPU 检查使用真实 Torch Tensor 和既有实际 `CannModel.verify` 的 CPU 算术夹具，覆盖完整字段正例、数值/选择/缺层失败、错误返回 ID、typed identity、精确 reset、32 行观测包装与独立标量对照、缺 stamp 和 teacher 身份拒绝；另检查无 HF 永远不能签发精度通过，以及默认 CLI 仍要求 reference。比较视图仅将每行恰好一个 index scale 规范为 HF 的 `[1]` 形状；实际 CANN 的平面 scale 和旧 CPU 夹具的 `[M,1]` raw 都原样保存，值与 dtype 不变。它不允许多 scalar 广播，不改变阈值。

`summary.json.case_precision_passed` 只表示这一例 eager 对照；`precision_verified` 与 `formal_precision_gate_passed` 始终为 false。后续八例、实际编译图/SK 与线上接受/缓存状态仍需各自证据。当前设备路径尚未由本入口执行，不把 CPU 夹具当生产数值通过。
