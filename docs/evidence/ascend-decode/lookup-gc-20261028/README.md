# 私有索引节点的 GC 表示

[`LookupDrafter`](../../../../tilert/models/glm_5_2_ascend/lookup_draft.py)的正常构造、commit 和 propose 保持有限无环子图。初始只有根；插叶只连接新节点；拆边将父到旧子的边换成父到新节点、再到旧子。证明时可给节点赋不变的有理数序号：父小于子，拆边的新序号取在两者之间，新叶取更大的值。四条子链接赋值均保持严格递增，因此不会形成环。这一性质不依赖坐标是否正确，也适用于正常 API 的回调重入；直接改写内部链接不在该范围内。

commit 的 `int(token)` 在生成 C 中使用 `PyNumber_Long`，键不会保留能反指起草器的整数子类对象。节点其余载荷是 `Py_ssize_t`，只拥有子字典和后代；起草器自身仍受 GC 跟踪。[构建器](../../../../experiments/lookup-native/build.py)据此只在编译输入中给节点添加 `@cython.no_gc` 和 `@cython.cclass`，原 `.py/.pxd` 字节不变。两份扩展同时装载后，module、节点类型和 GC 标志分别核对。

[Cython 的 no_gc](https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#disabling-cyclic-garbage-collection)要求对象不参与引用环。本次变化同时影响 GC header、traverse、分配、析构及子字典跟踪，不只影响扫描暂停。早期仅在增补声明中添加装饰器未改变实际 GC 标志；直接检查运行对象后才建立有效对照。

## 生命周期与正确性

基线 binary SHA 为 `f03d6bf62929b1f97f4bfe65576f49deee48b6f5f4935cea383bd3ec0294a6aa`，候选为 `37f5e5f88deb5a2789c8f0053a58d1b31d7a2512d128940f75256b596c579fe0`。两臂各通过 79 项回归、24159 次独立提议与 126 个实际 scalar session。[20 项生命周期检查](lifetime.json.gz)覆盖四类 8K 输入、深重复尾部、正常释放和起草器自环。候选所有节点与子字典均未被循环 GC 跟踪，起草器仍被跟踪；构造分配在 full collect 后均归零。

在本地 64 位 CPython 3.11.15，每节点构造分配恰少 16 字节。复制输入有 291 个节点，少 4656 字节；无匹配输入有 8804 个节点，少 140864 字节。该数量来自 tracemalloc 对实际构造调用的归属。

候选另经[隔离安装检查](installation/checks.json.gz)：冲突文件保持原字节，正常 import 不需要运行时 Cython，重复安装保留 inode 与 mtime。仓库默认导入仍为 Python；服务端按目标 ABI 重建并显式安装。

## 配对成本与退化

macOS arm64，同一 Python、Cython、编译器、编译参数及算法/类型源；width32、longest32/shortest2，四条固定 8192+1024 合成轨迹。每点两次预热、30 对交替测量；fresh 在计时外逐样本 full collect，carry 保留跨样本状态。正常 GC 无 callback 的结果如下，单位为每请求 setup 加全部 propose/commit 的 ms，排除 driver、接受计算、teardown 与 I/O。未绑核或锁频，没有模型推理或 NPU 计时。

| 负载 | fresh p50 | 胜数 | carry p50 | 胜数 |
| --- | --- | --- | --- | --- |
| 复制 | 3.881→3.809 | 23/30 | 3.773→3.765 | 18/30 |
| 5% 编辑 | 4.126→3.996 | 29/30 | 4.035→3.945 | 24/30 |
| 歧义 bigram | 2.180→1.781 | 30/30 | 1.961→1.711 | 27/30 |
| 无匹配 | 2.058→1.641 | 30/30 | 1.902→1.572 | 26/30 |

复制收益很弱；编辑稳态 p50 在 fresh/carry 分别从 0.691/0.663 降至 0.610/0.603 ms。长尾没有闭合：fresh 复制稳态 p99 为 0.486→0.563 ms；carry 歧义稳态 p95 为 0.665→1.337 ms；carry 无匹配 setup p90 为 1.511→2.600 ms。完整分位数、标准差、206 个统计上升项及全部非胜配对均保留，未移除离群样本。

No-GC 节点没有取消进程的自动回收。carry、带 callback 的歧义负载 sample5/position999，一次 1.331 ms commit 与 gen2 区间直接重叠 1.329041 ms。它不能回填无 callback 样本的原因；全局 GC 配置保持不变。

[独立读回](independent-audit.json)重建全部 1536 条样本、864384 轮提议、11140 个 GC 事件、24 组完整统计与配对结果，同时核对构建、生成 C、实际双扩展装载、回归和生命周期。原始 [fresh](fresh.samples.jsonl.xz)、[carry](carry.samples.jsonl.xz)与[压缩前后字节清单](manifest.json)随证据保存。复跑见[实验入口](../../../../experiments/lookup-gc/README.md)；CPU 成本、服务端重建、NPU 精度和在线 TPOT 分别验收。
