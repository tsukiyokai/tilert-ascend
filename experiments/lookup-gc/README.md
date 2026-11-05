# Lookup 节点 GC 对照

`lookup-native/build.py` 在编译输入中给私有节点添加 `no_gc/cclass`，保留算法 `.py` 与类型 `.pxd` 的原字节。节点仅拥有整数键、子字典及后代；起草器自身继续支持动态属性、弱引用和循环回收。内部新增回边或 Python 对象载荷后，须重新验证此引用关系。

`build.py` 生成保留节点 GC 的控制产物，便于在服务器同一 Python ABI 下重做对照。构建前安装 Cython 3.3.0 和 setuptools 82.0.1；运行时不需要 Cython。

```sh
python experiments/lookup-gc/build.py --output /tmp/lookup-tracked
python experiments/lookup-native/build.py --output /tmp/lookup-acyclic
python experiments/lookup-native/check.py \
  --build /tmp/lookup-tracked/build.json --output /tmp/lookup-tracked-checks
python experiments/lookup-native/check.py \
  --build /tmp/lookup-acyclic/build.json --output /tmp/lookup-acyclic-checks
python experiments/lookup-gc/verify.py \
  --baseline-build /tmp/lookup-tracked/build.json \
  --candidate-build /tmp/lookup-acyclic/build.json \
  --reference docs/evidence/ascend-decode/lookup-policy-20261028/rfind-cpu.json.gz \
  --output /tmp/lookup-lifetime.json
python experiments/lookup-gc/probe.py \
  --baseline-build /tmp/lookup-tracked/build.json \
  --candidate-build /tmp/lookup-acyclic/build.json \
  --candidate-checks /tmp/lookup-acyclic-checks/checks.json \
  --reference docs/evidence/ascend-decode/lookup-policy-20261028/rfind-cpu.json.gz \
  --gc-start carry --output /tmp/lookup-gc-paired.json
```

`verify.py` 对两份实际装载的扩展分别检查树、GC 标志、构造分配与释放；`probe.py` 复用原计时器，分别保留正常、带 callback、窗口内关闭自动 GC 的样本。另用 `--gc-start fresh` 和新路径测量每样本 full collect 的控制。

本次[固定产物的完整证据](../../docs/evidence/ascend-decode/lookup-gc-20261028/README.md)可独立读回；该审计固定本次 binary SHA，不是任意新构建的通用验收器：

```sh
python experiments/lookup-gc/audit.py \
  --archive docs/evidence/ascend-decode/lookup-gc-20261028 \
  --output /tmp/lookup-gc-readback.json
```

不兼容归档 Darwin CPython 3.11 ABI 的主机追加 `--skip-live-load`，报告会注明没有重新执行扩展。服务器的新构建须重新完成正确性、生命周期、配对测量及实际 NPU 请求验收。
