# Host 起草不加载未执行的 MTP 模型

`bench_cann_mtp.py` 的 lookup/automaton/tree 角色统一由 `host_drafted` 判定；这些角色的 `create_draft` 返回 `None`，设备 MTP 继续构造 `CannMtp`。状态账本按实际角色要求无 draft 长度或完整 MTP 长度；CLI 在加载前拒绝 host 模式的 MTP graph、投影和内嵌 MTP 选项。报告记录实际 `draft_model_loaded`，automaton/tree 的标签不再依赖额外的 `--lookup`。tree 源码一并加入 hash 清单。

[18 项 CPU 检查](cpu.json)通过，包含三种 host 角色零构造调用、设备 MTP 一次构造、无效选项、无 draft 的 lookup 请求、串行输出、cache/reset、profile 生命周期与伪造 draft 状态拒绝。重现：

```bash
PYTHONPATH=.:tests/glm52 python3 -m unittest check_bench_cann_mtp check_lookup -v
```

未改变 layer78 的预留缓存布局；v3 capture 仍读取并校验 MTP 文件中的 parent hidden。这次检查未运行 CANN 模型，未测 HBM、加载时长或 TPOT。

## 构造副作用须另行对照

`CannMtp._load_model`共享 target 的 embedding/head，FP32 head 安装会再次替换共享 `head.quant_method`。`CannModel.refresh_rope`读取 vLLM-Ascend 模块全局 cache；本地 vLLM-Ascend `d52c1b8de956507e6ace7ba351a998ed5cee6ce5` 的 `ops/rotary_embedding.py:111–115,447–448`存在构造时重写全局 cache 的路径。该本地版本不是历史 runtime212。

因此，设备验收需在实际 runtime212 比较 MTP 构造前后全局 RoPE 的身份、内容以及固定输入的 target 输出，之后再比较省去构造器的请求数值与计时。CPU constructor spy 只证明没有调用构造器。旧 12.413 ms 记录维持原配置身份。

## 后续 CPU 缺口

tree 的 `validate_local`仍按线性 zip 重算接受。`nodes=[7,99,8,9]`、`parents=[-1,0,0,2]`、`predictions=[8,100,9,10]`的合法路径是 `[0,2,3]`，应提交 `[8,9,10]`，当前检查只预期 `[8]`。需在轮回执保存 lineage，再独立验证路径、提交与缓存；本次未将 tree 角色修复记作树验收完成。

后续 [tree lineage 检查](../tree-lineage-20261028/README.md)已闭合上述 CPU 回执缺口；本页的 18 项记录保留当时源码身份。
