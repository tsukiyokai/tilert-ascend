# 固定prefix Value缓存

[CPU准备收据](preparation.tar.gz)绑定16份BF16 `[78,4,8192,256]`缓存、checkpoint、冻结prefix与每层投影权重。保持原HF的完整1792列GEMM、BF16舍入和Value选择顺序；rank2正对照逐位匹配既有CPU夹具。总大小20,937,990,800B，CPU准备165.756秒；准备成本和每rank约1.31GB额外存储须随实际接入报告。

[真实9382单die读回](unit-result.json)固定rank2的实际QL/QPE/KV和2048个选择位置，完整QK/概率/Value/PV各重复32次，两臂分别逐位稳定，输入及缓存未变。原展开臂复现原设备输出，与CPU有1/1024项不同；缓存臂逐位匹配CPU。该差异只验证缓存接线，后续整模取舍沿用原精度容差。

[设备收据](device-receipts.tar.gz)含冻结源码、命令、空闲检查、退出0及23个ELF的打开记录。候选一次准备连续Value缓冲，decode只写当前行；目前只支持M1/8193。完整模型精度、多步状态与时延均未验收，未测性能。
