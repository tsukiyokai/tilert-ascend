# 固定M1的SFA索引顺序干预

承接[TP16三臂结果](../layer68-boundaries-a3-20261029/README.md)，在真实A3 9382物理die12固定原rank12的M1 QL/QPE、KV、页表、长度`[1]/[8206]`及原生SFA入口，仅替换三份实际indices：原生M1顺序、M32 row13顺序、按位置排序。每臂32次完整SFA，未计时。

[实际报告](report.json)显示三臂均逐位复现各自冻结输出，输入未变；固定M1形状与长度，仅换成M32顺序即逐位复现原M32 row13输出。原生/M32顺序只差相邻两项5196/2079，输出只差1个BF16元素，最大绝对差0.00390625。这证明索引顺序足以造成该局部SFA分歧；不把原因归于width形状或HCCL。排序输出自身稳定，但与原生数值不同。

[独立CPU读回](independent-readback.json)重新读取96个实际设备输出，全部匹配冻结参考。源码从固定SHA的DeviceOp方法AST提取原方法体，加载同一原生库；库SHA、实际映射、元数据、契约、空闲检查、命令及退出0保存在[冻结收据](receipts.tar.gz)。[共享归档](archive-readback.json)保留原始inputs/outputs PT与源码，并从归档重读核对全部两PT及报告SHA；[清理](cleanup.json)只退役已结束的本方容器。没有完整模型精度、其他输入或性能通过。
