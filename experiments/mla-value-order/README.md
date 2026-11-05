# 固定真实概率的Value舍入对照

入口只隔离首行layer0/rank0的Value投影顺序。HF概率保持原值，prefix与当前行分别展开；CPU显式关闭mkldnn，与冻结HF一致。实际A3结果、输入归档、设备绑定和ELF摘要见[证据](../../docs/evidence/ascend-decode/mla-value-order-a3-20261029/README.md)。

在对应镜像与挂载环境中准备输入，再在即时确认空闲的A3 die上执行；镜像和挂载使用证据中的实际命令。

```bash
python experiments/mla-value-order/run.py prepare \
  --root /work --source /work/source-v12 \
  --output /work/inputs/mla-value-order.pt
python experiments/mla-value-order/run.py replay \
  --source /work/source-v12 \
  --input /work/inputs/mla-value-order.pt \
  --output /work/evidence/mla-value-order
```

每臂保留32次真实输出并检查逐字节重复；数值比较复用`check_mla.py`原有阈值。范围为固定概率的局部Value计算，完整attention、整模型精度及TPOT另验。
