# 树轮回执保留完整 lineage

`ExecutedRound.parents`保存 pending root 与全部验证行的父节点，root 为 −1；线性轮为 `None`。`MeasuredRank`及 session accepted span 原样保存该数组，交付检查核对两份回执。树记录缺少或伪造 lineage 时拒绝验收，旧记录无法据线性顺序补出分支。

验收器逐节点向根回溯，只保留每条边均满足 `prediction[parent] == token[child]` 的完整路径，选择最长者，并列时取终点索引最小者；再按剩余输出预算裁剪该路径的预测。执行器从根向前标记可达，两种检查都计算根路径上同一组边等式的合取，故对合法父数组选出同一路径。内部节点可以是接受终点，完整 lineage 中的 padding 也可以成为合法 root 子节点。

反例 `nodes=[7,99,8,9]`、`parents=[-1,0,0,2]`、`predictions=[8,100,9,10]`现重建路径 `[0,2,3]`及提交 `[8,9,10]`。预算只剩一项时保留自然接受数 3、实际提交 `[8]`，main cache 只消费原 pending token。缺失、自环、未来父节点、布尔 parent、错误接受数及把树标成线性均有拒绝检查。

[25 项 CPU 检查](receipts-cpu.json)包含 26,260 个小树/二值标签穷举，以及 3 种独立标量规则 × 8 个宽度的 24 个真实 session、1,801 个 accepted span。它们通过实际执行器、测量记录与交付 session，逐项核对 lineage、接受 token 与串行输出。另通过 [宽度及单行尾部](invariance-cpu.json)、[round](round-cpu.json)和 [EOS/取消/session](session-cpu.json)回归。

```bash
PYTHONPATH=.:tests/glm52 python3 -m unittest check_bench_cann_mtp check_lookup -v
PYTHONPATH=. python3 tests/glm52/check_round_invariance.py --output /tmp/tree-invariance.json
PYTHONPATH=. python3 tests/glm52/check_mtp_session.py --output /tmp/tree-session.json
```

这些结果验证 CPU 状态与回执，不验证 CANN 的树 attention 数值、设备 cache 字节或 TPOT。正式 CANN 精度门禁及实际设备对照仍待完成。
