# TP16 Attention边界定位

复用冻结源码中的 `tests/glm52/run_cann_layer_major.py`，运行一个 teacher token和前1–3个dense层。`run.py` 保存实际 attention、post norm、dense MLP、两次 FP32 TP 规约，以及 SFA 原 query/KV/索引/页表/长度/输出；hook 只克隆观测值。执行后，原入口逐位检查完整 CANN 首帧的 hidden、索引和新 KV，检查层对象释放。它不运行完整模型，也不测时延。

`prepare_baseline.py` 从完整采集的 `serial-00.pt` 生成原入口需要的 M1 文件视图。保留原失败状态、原始与派生 SHA，以及实际 79 个 hidden、78 层 KV/索引；不生成替代数值。

```bash
python run.py --source /work/source-v12 \
  --checkpoint /models/glm52 \
  --capture-case /assets/checkpoint-owned/prefix8k-v3/case-00 \
  --teacher-json /work/receipts/teacher-32.json \
  --prompt-json /work/receipts/prompt-8192.json \
  --baseline /work/inputs/layer0-baseline-v3 \
  --output /evidence/layer0-boundaries \
  --layers 1 --steps 1 --query-width 1 --fp32-head
```

上述命令由真实 A3 的 `torchrun --nproc-per-node=16` 启动；运行前核实 16 die 空闲、SKU、冻结输入和 OPP 搜索顺序。`--check-inputs-only` 只核验主机输入，不执行 NPU。

全部 rank 工件完成后，`compare.py` 在匹配版本、`mkldnn=False` 的 CPU 上执行三项干预：实际 attention 输出接原 HF norm/MLP；实际 post norm 输出接原 HF MLP；实际 down 输入接原参考 down 投影。另用实际 o_proj 输入检查本地投影。第一层 input norm 按 HF manifest 选择实际 prolog 算术，post norm 使用原 HF 类。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python compare.py \
  --source-root /work/source-v12 --checkpoint /models/glm52 \
  --captures /work/evidence/layer0-boundaries \
  --hf-step /work/reference/step-08192.pt \
  --output /work/evidence/layer0-tail.json --threads 8 --check-o-proj
```

先核查 `attribution_prerequisites_passed`：原 h0、边界连接、rank 一致性和完整模型首帧复现均须成立。`diagnostic_complete` 仅表示比较执行完成。实际 FP32 partial 的 CPU rank 顺序求和不模拟 HCCL 树；只有最终 BF16 差异才可能影响该边界输出。局部对照不替代完整模型精度、八输入或 TPOT 验收。

`--value-fixture FILE` 固定全部64head的HF概率，将先展开与后投影Value两图连接原o_proj及HF尾部；必须先逐位重建HF h1。`--sfa-boundaries` 按实际页表还原8193个有效KV，先以实际SFA输出闭合UV及后续，再比较FP32概率、BF16概率、BF16 score及概率候选。score单因素对照始终保留BF16概率；组合候选按原前缀/当前行边界先展开Value，并以HF概率分支作为正控制。保留原索引顺序，拒绝负索引、重复项和活跃页别名，不读取未用KV尾。这些CPU候选不模拟CANN的分块FlashAttention内部。

`run.py --expanded-value` 安装上述组合的设备诊断图，适用于`--layers 1..3 --steps 1 --query-width 1`；它固定`allow_hf32=False`并在逐层卸载时恢复。该图每次重新展开前缀Value，未作为性能候选。原CANN精确比较的预期数值变化继续记fail；结构检查和独立HF容差分别核验。`compare.py`仍用于第一层边界，前三层传播结果见[实际A3读回](../../docs/evidence/ascend-decode/attention-rounding-a3-20261029/README.md)。

`prepare_difference.py`从16rank工件独立重建CPU组合，核验全局输出SHA后保存首个不同rank的真实输入与权重映射；`replay_difference.py`在单die A3分离score、概率、Value和PV。每臂32次重复只覆盖PV，Value各展开一次。浮点Value投影的BF16与FP32两臂若产生相同PV，停止该精度旋钮，继续检查跨层传播。
