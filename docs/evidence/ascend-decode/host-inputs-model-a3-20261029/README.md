# 合包输入的真实TP16消费者

真实A3 9382、TP16、一个冻结8K前缀，在同一常驻CANN模型中只切换分别准备与packet输入。每臂执行M1首行、完整M32和M32 reset，保存96份实际PT。共享模型和缓存，属于数值试验，没有性能计时。

[独立CPU读回](readback.json)绑定真实容器退出0及summary→16份rank报告→96PT的SHA，逐份读取输入、最终输出和78层新增KV。两臂各阶段的最终logits/hidden/token IDs均逐位复现旧设备参考；两臂的输入和新增KV逐位一致，两次M32 reset也逐位一致。从PT重算的16rank最终输出摘要相同；共有16,496项独立比较，失败0。状态摘要只覆盖最终输出。

每rank另有1,175项现场检查，合计18,800项。其中18,528项逐阶段比较全部冻结前缀，均通过；前缀未再次写入这些PT，独立reader核对完整检查库存和源码，不宣称重新计算现场前缀结果。[摘要](summary.json)及[收据](receipts.tar.gz)保存原报告、预注册契约、命令与实际NPU运行及无设备CPU读回均退出0。主入口3项CPU回归、reader 11项回归含重签收据但篡改单项KV的负例，见[验证收据](validation.tar.gz)。

实际CANN源码SHA为`7e5a9822426468b3e2b0a65da539da79536aac5d13a01c4d8047e8fd0bce47fd`，HostInputs为`f2f2286fb43ef4dbdf09b2bff6588be69b57f39de1855f5a3363a2b6d0ef6034`；main为`209d8ba6d9f9b4cb262ed6bf246e18f4397ba78fa2dcf7a591d7a415341fe1cf`。[加载收据](kernels.json)解析16份strace的79个kernel文件，未解析路径0；成功open不充当执行或性能证明。原始PT、源码和日志的[共享归档](archive-readback.json)为54,318,365B，SHA256为`e500417c4f02c2934cdff16d7a09e15ae5fb4dfbf19256a8216d14ca1ccbf2ab`；从归档重读96PT及17份capture JSON均匹配原始SHA。[清理](cleanup.json)只移除本方已退出0的模型与无设备runc reader容器，原始bind文件保留。

本次证明packet可供该输入的真实eager模型消费。HF精度、实际模型graph、其他七个输入及在线TPOT仍待验收；[组件计时](../host-inputs-a3-20261029/README.md)的20%预测仍为失败，不据模型数值结果升级性能结论。
