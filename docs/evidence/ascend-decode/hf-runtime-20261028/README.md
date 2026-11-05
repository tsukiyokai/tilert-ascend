# 78 层 HF 的合成 CPU 执行

[运行检查](../../../../tests/glm52/check_reference_runtime.py)实际构造 Transformers 5.12.0 的 78 层模型，执行 HF forward、最终 head 与 77 个逐层诊断 head，保存 trace 后 reset、重放并读回。BF16 与 FP32 两条真实采集命令各有 2379 项检查通过；[独立检查](runtime.json)48 项通过，[回归](regressions.json)44 项通过，另有 [62 项元数据检查](metadata-cpu.json)。

合成 checkpoint 保留 TP16、64 attention heads、192+64 的 query/key 几何、128 维/32 头 indexer、full/shared 层序与第 6–66 层的 C8 key 编码；hidden/latent rank 为 16、词表 32、专家 16，投影权重为 FLOAT。attention 与 FFN 的输出为零，embedding、latent、RoPE、index key 和路由有实际变化。输入 `1,2,1` 的短序列用于检查执行、接线与重放；它不验证真实 W4A8C8 checkpoint、长前缀的 top-2048 选择或 NPU 精度。

`x=[1,…,1]` 时残差保持原值，最终 norm 的 BF16 输出仍为 1；head 的两个词表行为 `[1,0,…]` 与 `[1,2^-8,0,…]`。独立标量公式核对全部 78 个 head：BF16 把两行都舍入为 1，FP32 保留 1 与 1.00390625，贪心结果分别为 0/17。三个位置的全部 hidden、latent、75 个 MoE 路由与均匀 attention 也分别核对独立公式。reset 与磁盘读回逐项比较实际 trace。

## 修正的执行契约

当前 HF gate 返回 router logits，`GlmMoeDsaMoE.route_tokens_to_experts` 才产生专家与概率。[适配器](../../../../tilert/models/glm_5_2_ascend/reference.py)调用该 HF 原函数，保持稳定 top-k 与概率舍入；路由 hook 从实际 MoE 结果采集选择。旧版 gate 的三元组协议继续沿原分支执行。Prolog 的 routed scaling 仍由原 gate 属性控制；本次完整运行使用 native MLA。

参考实现及采集器接受显式 `c8_source_directory` / `--c8-source-directory`，验证固定 header 哈希并记录解析后的目录。仓库 [C8 源码副本](../c8-arch22/)的两个文件与要求逐字节匹配；修改一个文件或指定缺失目录均被实际构造拒绝。默认 SDK 路径规则不变。[早期默认路径失败](../fp32-head-reference-20261028/constructor-prerequisite.json)仅说明该绝对路径不可用。

## 复跑与原始证据

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests/glm52 python3 \
  tests/glm52/check_reference_runtime.py --output /tmp/hf-runtime
```

需要 CPU Torch、safetensors、匹配的 Transformers。程序生成约 80 MB 的 checkpoint；生成规则随源码保存，权重文件哈希写入 `runtime.json`。本次为 Python 3.11.15、Torch 2.9.0、Transformers 5.12.0。每条采集命令另记录实际导入 HF 文件、参考类文件和源码哈希。

原始 manifest 与 trace 以 gzip 无损保存：BF16 [manifest](bf16/manifest.json.gz)、FP32 [manifest](fp32/manifest.json.gz)，同目录各有三个 `step-*.pt.gz`；解压后可按 manifest 的 SHA256 核对。`precision_verified`、`formal_precision_gate_passed` 与性能声明仍为 false；真实 checkpoint、Prolog-v3、CANN FP32 caller 与实际 graph 的逐层对照继续验收。
