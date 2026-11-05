# 同源码的 lookup 原生执行

[`lookup_draft.py`](../../../../tilert/models/glm_5_2_ascend/lookup_draft.py)保持字节不变。增补 [`.pxd`](../../../../tilert/models/glm_5_2_ascend/lookup_draft.pxd)将节点字段、上下文位置与遍历循环编译为原生代码，保留原 longest/latest 策略；token ID、尾长上界、提议数量与计数器仍用 Python 整数。`propose` 的局部 context 保留 object 类型，使超大 count 沿 Python 切片规则裁剪；动态属性、弱引用与循环回收也分别检查。

构建采用 [Cython 的增补声明机制](https://cython.readthedocs.io/en/latest/src/tutorial/pure.html)，保留 bounds/wraparound/None/overflow 检查。构建目录按完整包名排布，隔离加载须实际出现扩展类型字段与 C vtable；生成的 C 与宿主 arm64 反汇编随产物保存。默认部署仍使用 Python；显式安装后，新进程才会加载同名扩展。

## 正确性与身份

[实际二进制](build/lookup_draft.cpython-311-darwin.so.gz)的 SHA256 为 `f03d6bf62929b1f97f4bfe65576f49deee48b6f5f4935cea383bd3ec0294a6aa`。[基础进程](checks/basic.json.gz)通过 72 项回归、24159 次独立提议、217685 个逐边前缀及 126 个实际 scalar session；[HTTP 进程](checks/http.json.gz)另通过 7 项检查。[基准入口](checks/benchmark-entry.json.gz)通过 15 项回归，并实际验证两个入口在导入 Torch 前拒绝过期扩展。

五类修改反例覆盖加载前的 `.py`/`.pxd` 变化，以及加载后的源码、声明和模块路径文件替换。[安装检查](installation/checks.json.gz)另核对普通 import、无 Cython 运行时、冲突文件保留与重复安装不改 inode/mtime。安装同一个已验收二进制，启动记录同时保存其文件哈希、原算法/声明哈希、ABI、构建选项及工具版本。构建与安装文件不可覆盖；装载时路径文件哈希的含义与设备 ELF 的实际执行验收分别记录。

## CPU 对照

macOS arm64、CPython 3.11.15，原 Python 与原生扩展使用同一个算法源。固定四条 8192+1024 的合成轨迹，width32、longest32/shortest2，每点两次预热、30 组交替配对；fresh 逐样本在计时外 full collect，carry 保留跨样本 GC 状态。正常 GC 无 callback 的臂报告性能，阶段 callback 与窗口内关闭自动 GC 的臂用于诊断。未绑核或锁频，没有模型推理、NPU 或网络计时。

下表为每请求 setup 与全部 propose/commit 调用时间之和的 p50，单位 ms；排除接受计算、driver、teardown 与 I/O。每格为 Python→原生。

| 负载 | fresh | carry |
| --- | --- | --- |
| 复制 | 7.823→3.799 | 7.920→3.856 |
| 5% 编辑 | 8.937→4.005 | 8.874→3.994 |
| 歧义 bigram | 6.894→2.032 | 6.943→1.997 |
| 无匹配 | 5.049→1.971 | 5.023→1.936 |

正常 GC 的 setup/总成本均为 240/240 对下降；稳态为 237/240 对下降。carry 编辑负载稳态 p95 为 2.177→2.708 ms、胜数 28/30；carry 无匹配稳态为 29/30。fresh 歧义/无匹配的 propose 最大值也有退化。所有 p50/p90/p95/p99/std/max、配对胜数与退化保留在 [fresh](fresh.json.gz)、[carry](carry.json.gz)；每点 30 请求的最近秩 p99 为最大值，未删除离群样本。

共 1536 条原始样本，包括预热和诊断臂。[独立读回](independent-audit.json)用另一个对齐整数反向搜索重建提议，核对 864384 轮、14980 个 GC 事件、完整统计与配对结果；从归档重新装载二进制，核对原生字段类型，全部通过。[清单](manifest.json)绑定压缩前后字节、源码、二进制、生成 C 与完整原始数据。CPU proposer 收益不等同于服务器或在线 TPOT 收益，GC 配置保持原值。

## 复跑与设备顺序

构建、检查、配对测量和安装命令见[实验入口](../../../../experiments/lookup-native/README.md)。Darwin 产物用于复核本地执行，不能装到 A3 Linux。服务器按实际 Python ABI 重建并核对完整输出，再按[单 die→少 rank→空闲 A3](../online-lookup-20261028/README.md)继续 CANN、图、精度与双时钟 TPOT 验收。
