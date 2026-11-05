# 全局 radix arena 的负结果

全局 arena 未替换生产索引。tuple arena 在四类负载的总成本 p50 都退化；坐标 arena 改善歧义与无匹配输入，却使复制和轻度编辑退化约 8–9%。删除基准内逐轮对象保留后，取舍仍相同。该对照未修改生产索引或 GC 配置；后续[局部坐标边](../lookup-coordinate-20261028/README.md)单独验收。

坐标版用整数数组保存边区间/latest，边引用 append-only context，以单个全局 transitions 表取代节点的局部字典；合法词表键为 `(parent<<18)|token`，范围外整数使用 tuple 键。tuple 版保留边标签，单独观察全局容器/访问方式的组合变化。两版均通过 [独立扫描与任意边前缀检查](tilert-lookup-arena-correctness-20261028.json)：各 4115 个 context、16460 次提议、52578 个前缀不变量，包括增量、8K、L64/96、重叠、空提交与旧 hash 碰撞反例。证明沿用[精确 radix 的 latest 不变量](../lookup-radix-20261028/README.md)。

## CPU 对照

macOS arm64、Python 3.11.15；固定四条合成 8192+1024 轨迹，width32、shortest2/longest32。每点两次预热、30 组配对，轮换三臂的六种顺序；无 affinity/锁频。四个面板分别交叉原 replay/无逐轮保留、fresh/carry GC 起点，各含三种 GC 模式，共 4320 条测量样本与 288 条预热样本。仅计 setup、propose、commit，排除推理、I/O 与 teardown。

下表取无逐轮保留、正常 GC、无 callback 的总成本 p50；每格为 fresh/carry，单位 ms。

| 负载 | 生产 radix | tuple arena | 坐标 arena |
| --- | --- | --- | --- |
| 复制 | 8.256/8.330 | 9.291/9.280 | 8.984/8.982 |
| 5% 编辑 | 9.323/9.351 | 10.532/10.342 | 10.138/10.115 |
| 歧义 bigram | 9.873/9.954 | 11.685/11.711 | 8.827/8.847 |
| 无匹配 | 8.094/8.056 | 8.930/8.927 | 6.129/6.124 |

坐标版的总成本配对胜数为 fresh `0/0/30/30`、carry `0/2/30/30`，顺序同表。全部原始提议/接受签名与独立扫描一致。[原 replay 摘要](tilert-lookup-arena-plain-summary-20261028.json)、[无保留摘要](tilert-lookup-arena-noretention-plain-summary-20261028.json)另保留 setup、稳态、commit p99、请求稳态 p99 和配对胜负；30 请求的最近秩 p99 就是最大值。

## 高尾与观测范围

fresh 在每次计时前 full collect；carry 不强制回收。GC 模式分别为开启无 callback、开启并观测、仅测量窗口禁用自动 GC 并观测。[callback](https://docs.python.org/3.11/library/gc.html#gc.callbacks)按 setup/propose/commit/driver/teardown 记录回收区间；开启无 callback 的臂用于核对观察器扰动。carry 中生产臂确有全代回收落在计时调用内；事件只能定位阶段，不能单独识别触发对象与被扫描对象。

原 replay 在请求内保留 round dict、interval tuple 与 proposal；无保留版用预分配数值缓冲，结束后重建原始报告，并提前切好 commit 输入、计算接受量。两版之间同时改变了这些分配与工作，差值不能全归给 GC。线上 session 本身也保留接受回执；该对照没有测量常驻服务。

四面板未重现历史无匹配约 16 ms 稳态高尾；无保留面板的生产稳态 p99 为 fresh/carry `1.820/1.794 ms`。历史记录保持原值，当前证据不足以把那次高尾认定为索引固有 GC。[局部表示的后续对照](../lookup-coordinate-20261028/README.md)分别检验了局部字典下的坐标标签、叶子字典延迟分配。

## 原始记录与复跑

[独立读回](independent-audit.json)逐行重算 4608 个签名、计时加总、全部总成本 p50，并核对观测事件代际计数与 `gc.get_stats`、关闭窗口零回收及源码 SHA。四组 `*.json.gz` 保存完整报告，`*.samples.jsonl.xz` 保存全部原始样本；解压 payload SHA 与压缩文件 SHA 均在该审查中记录。[冻结源码](sources/)包含原型、driver、汇总与读回程序。

```sh
gzip -dc docs/evidence/ascend-decode/lookup-policy-20261028/rfind-cpu.json.gz \
  > /tmp/tilert-lookup-rfind-probe-20261028.json
python3.11 docs/evidence/ascend-decode/lookup-arena-20261028/sources/tilert-lookup-arena-bench-20261028.py \
  --gc-start carry --output /tmp/arena-retained.json
python3.11 docs/evidence/ascend-decode/lookup-arena-20261028/sources/tilert-lookup-arena-noretention-bench-20261028.py \
  --gc-start carry --output /tmp/arena-noretention.json
```

driver 固定本次 checkout 的绝对根目录；生产 radix 的冻结副本与 Git `352ab99` 相同。异机或新源码下重新测量并保存对应身份。本页没有 A3 或 TPOT 性能结论。
