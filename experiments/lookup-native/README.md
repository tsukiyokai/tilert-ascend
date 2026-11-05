# Lookup 原生编译与验收

从同一个 `lookup_draft.py` 与增补 `.pxd` 构建 CPython 扩展；构建环境需要 C 编译器、Cython 3.3.0 与 setuptools 82.0.1。扩展运行时不导入 Cython 或 Torch。必须在目标主机、匹配的 Python ABI 下重新构建。

当前构建器只对私有无环节点关闭循环 GC，起草器自身继续支持引用环；正常方法只连接后代，并把键归一化为整数。实际 GC 标志在隔离装载时检查。[节点 GC 对照](../lookup-gc/README.md)保留受 GC 跟踪的控制构建、生命周期与长尾结果。

```sh
python experiments/lookup-native/build.py --output /tmp/lookup-native-build
python experiments/lookup-native/check.py \
  --build /tmp/lookup-native-build/build.json --output /tmp/lookup-native-checks
python experiments/lookup-native/check_install.py \
  --build /tmp/lookup-native-build/build.json \
  --checks /tmp/lookup-native-checks/checks.json --output /tmp/lookup-install-checks
python experiments/lookup-native/probe.py \
  --build /tmp/lookup-native-build/build.json \
  --checks /tmp/lookup-native-checks/checks.json \
  --reference docs/evidence/ascend-decode/lookup-policy-20261028/rfind-cpu.json.gz \
  --gc-start fresh --output /tmp/lookup-native-fresh.json
```

另以 `--gc-start carry` 和新输出路径复测自然跨样本 GC。正确性检查分别启动无 Torch 的基础进程与 Torch CPU HTTP 进程；性能程序复用坐标实验的计时器、固定轨迹与 GC 观测，保存所有逐轮提议和时长。

安装会新增一个覆盖同名 `.py` 导入顺序的扩展文件。先核对结果，再安装同一个已检查产物，并在新服务进程中读取 `host_drafter`：

```sh
python experiments/lookup-native/install.py \
  --build /tmp/lookup-native-build/build.json \
  --checks /tmp/lookup-native-checks/checks.json --output /tmp/lookup-installed.json
```

安装器原子发布文件，拒绝替换已有不同二进制；相同产物保持 inode 与 mtime。更新扩展须先结束使用旧文件的服务，再移走旧文件并重新安装。源码或声明变化后，旧扩展会拒绝加载。启动入口另复核模块文件、源码与装载时记录的哈希；原始 `LookupDrafter.report()` 字段保持不变。

本次[完整证据](../../docs/evidence/ascend-decode/lookup-native-20261028/README.md)仅覆盖本地 CPU；服务器重建、实际 CANN 请求及 TPOT 继续分别验收。

```sh
python experiments/lookup-native/audit.py \
  --archive docs/evidence/ascend-decode/lookup-native-20261028 \
  --output /tmp/lookup-native-readback.json
```

归档审计从 Git 和归档哈希恢复依赖，不依赖原 `/tmp`。不兼容 Darwin CPython 3.11 ABI 的主机追加 `--skip-live-load`，保留字节、类型声明、提议和统计检查，并明确记录未重新装载二进制。
