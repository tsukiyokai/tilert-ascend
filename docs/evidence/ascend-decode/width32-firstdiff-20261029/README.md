# 原生M1/M32的row13首差

从[完整修正包TP16采集](../fixed-hf-launch-20261029/README.md#9392完整复验)读取rank0的`serial-13.pt`和`wide.pt`，固定teacher、summary、rank报告、两PT及采样源码SHA。row13消费token24187、位置8205。此次仅在CPU读回已保存的真实A3状态，runc、2CPU、8GiB、无设备绑定，退出0；未重跑模型。

[完整结果](row13.json.gz)区分原始字节、选择顺序/集合及原容差：

| 边界 | row13观测 |
| -- | -- |
| h0～h68 | 逐位相同 |
| layer1起部分indices | 顺序有差异，集合仍相同 |
| layer68输出h69 | 首个hidden字节差，1452项，NRMSE 0.001774，原容差内 |
| layer69新latent | 首次字节差，NRMSE 0.004836，原容差内 |
| layer70新latent | 首次越过原容差，NRMSE 0.012712 |
| layer70 indices | 首次集合差，重叠2045/2048，双方各3个独有位置 |
| row13返回token | 仍相同；最终logits原门槛失败 |

layer68当前行的latent/rope、routes ID与顺序均逐位相同，indices承接layer66的两个位置交换。h68是采样时合并的残差表示，不能代替norm的两份实际输入；rank0当前行相同也不能证明其他行的历史KV相同。现有采样不能把h69差异归因于SFA选择顺序、MoE分组或某个舍入。下一步先核对serial00～13的layer68 KV，再对齐norm两输入/输出、attention及MLP边界，保留各rank实际FP32规约输入/输出和路由权重。layer70仍是下游观测点。

[执行收据](receipts.tar.gz)含冻结reader、命令、真实退出状态和日志；原输入仍绑定既有70,989,101,922B共享归档。重放入口为`python experiments/cann-precision/first_difference.py --capture-dir CAPTURE --teacher teacher-32.json --output row13.json`。reader固定本次rank0/row13，不宣称它是全部32行的最早字节差异；原完整检查已确认row13为最早未过门槛的行。正式精度与性能认证均为false。
