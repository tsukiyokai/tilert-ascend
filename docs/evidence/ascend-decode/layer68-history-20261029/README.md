# Layer68历史输入核对

在[原生row13首差](../width32-firstdiff-20261029/README.md)基础上，读取同一失败TP16采集的rank0 `serial-00..13.pt`及`wide.pt`，逐文件核对原报告SHA。[14行结果](history.json)中，h68合并残差、新latent/rope及routes均逐位相同；indices存在顺序变化，集合相同。row13两条路径均选择全部14个新增位置，未发现不同的历史KV；原报告的386项冻结前缀检查全部通过。

这排除了该rank已保存历史KV不同的候选解释。h68仍不代替norm的两份实际输入，QL、实际attention结果及TP局部partial还需设备采样；本次没有新的模型或设备执行。

[执行收据](receipts.tar.gz)保留冻结reader、原始输入身份、命令与退出状态。两个准备任务均为runc、2CPU、无设备绑定、退出0。另从原`serial-00.pt`、`serial-13.pt`和`wide.pt`提取22,603,748B的[旧输出控制](controls-receipt.json)，SHA256为`1f3b3d62119d4e887dff91fbabefd397391d787a04414cbe7601e2a290af76fe`，供新采样核对原生输出。原始大PT继续由既有完整归档绑定。
