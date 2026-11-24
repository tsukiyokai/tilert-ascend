# SFA 索引单位与真实首帧

9382、CANN9.2、同一实际 `_C_ascend.npu_sparse_flash_attention` 的[单die见证](witness.json)使用零query、512个合法索引、其余−1，以及分段常量KV。block参数按1/16/1切换，输出逐位等于各自独立标量参考1/1.9375/1；前后block1逐位一致。[实际运行源码](witness-source.py)与扩展ELF的SHA均绑定在收据中。它证明此接口按块单位解释索引，未测时延。

[冻结设备消费者](device-sfa-consumer.txt)的完整文件SHA为`63001f4e…`，BF16路径硬传block16，输入仍是逐token位置。[真实v4首帧对照](v4-readback.json)中，embedding与第0层新KV逐位相同，hidden1 NRMSE为0.081786，hidden2和最终logits为NaN，实际首预测0、HF首遍为382。HF精确reset尚未完成，此处是诊断，未通过模型精度门禁。改变block参数能否恢复整模型边界，另由单因素重试检验。

整模型候选只应用[一行源码修正](token-index.patch)，将该BF16路径的16改为1；[原文件与候选SHA](runtime-overlay.json)绑定实际字节，其余Python与扩展文件链接同一只读运行包。较新本地上游源码已使用1，本次修正对象是旧冻结包。

v4保留了16rank×32个M1工件与全部prefix精确检查；M32因观测槽硬编码INT64、实际token为INT32而中断。槽现继承真实整数dtype，仍预留32行并严格检查shape/dtype/device；[15项CPU检查](request-dtype-cpu.log.gz)覆盖真实verify夹具、M1→M32、INT32/64及地址/epoch，未替代设备验收。CANN源身份记录补入实际DeviceOperator消费者。

完整原始M1、HF首帧和单位见证已归档到共享NFS的`/trt-a3-evidence-20261028/cann-v4-source5fd0815.tar.zst`，3680685247B，SHA256 `734f8db57758bd0375b73e38941a556c6e6be91fb8b01ec28079e21143d702e4`。源模型采集提交为`5fd08156d8bfbee2fed5e97345e1bf3bd7cc68c5`，不是正式精度或TPOT结果。

## block1 整模型复验

[独立CPU HF](hf-complete.json)已完成32行与精确reset，25347项检查全过，32份trace实际复制SHA匹配。v6使用`03fa67187e572622838b477b60a6d809c781b8a6`与上述单行运行包修正，在同一9382主机完成16rank的512个M1、16个M32及16个reset工件，实际退出1。各rank状态一致，但采集执行检查失败，未通过单例精度。

[首帧对照](candidate-v6-readback.json.gz)中，logits恢复有限，预测382与HF一致；hidden1/2 NRMSE降至0.003595/0.005977。首行1096项仍有438项失败，最终logits NRMSE为0.115986、max-abs为1.732704，不能只凭top1相同判精度通过。rank0的M1/M32、M1/reset及精确reset仍分别有13280/13226/14572项失败。[宽帧/reset读回](candidate-v6-reset.json.gz)确认前3个hidden及前3层选择逐位相同，后续分歧仍待定位。没有新增TPOT测量。

[rank0全部32行首差](candidate-v6-first-reset-difference.json.gz)进一步确认hidden0–4逐位相同、hidden5首次不同，max-abs为0.0001220703125。hidden字段是下一层input norm输出的融合残差，未记录该norm的两输入或归一化输出，不能据此排除更早的分支差异。[后续16rank边界采集](../reset-boundaries-a3-20261028/README.md)把首个已采差异推进至部分rank的layer3本地MoE、TP输入之前。原summary的rank状态一致仅覆盖预测、位置和epoch，不代表中间Tensor相同。
