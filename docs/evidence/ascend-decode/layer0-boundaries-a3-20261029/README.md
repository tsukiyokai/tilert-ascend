# 真实 TP16 第一层边界闭合

9392、CANN9.2、固定镜像 `3892a691…`，复用完整模型首帧的冻结8K prefix、token6500与原运行包。逐层入口仅补齐未启用的workspace/restore字段，[源码差异](source-derivation.json)绑定原版与修正版SHA；数值实现保持冻结。早期入口在第一层前向后因缺字段丢失采集，[原始错误](initialization-error.log.gz)保留。实际[启动](launch.sh)按rank分别跟踪并复用编译缓存；本次不测时延。

[16份报告](capture-summary.json)每rank22项检查全过，包含hidden、索引、新KV对原完整CANN首帧的逐位比较和层对象释放。真实input norm、attention、post norm与MLP边界的连接及rank一致性均通过；实际两处norm的epsilon均为1e-5，gate/up按TP rank分片。原始rank报告保留在 `ranks/`，Tensor与跟踪日志另行归档。

[固定输入干预](result.json)使用匹配的Torch2.12、Transformers5.19、8线程、`mkldnn=False`与原参考权重/算术。input norm逐位匹配冻结HF。把实际attention输出交给原HF post norm及MLP，最终h1与CANN逐位相同；它与原HF h1仍有2960项不同，NRMSE为0.00359479594。实际16份o_proj输入交给参考投影，其BF16汇总也逐位匹配实际attention输出。

本输入上的h1偏差因此收窄到o_proj之前。o_proj和down各有一个rank的一项局部partial不同，但最终BF16汇总均相同；实际o_proj partial的FP32 HCCL结果与CPU rank顺序和不同，BF16输出在16rank均相同。该对照不模拟HCCL树，也不把局部差异解释为h1成因。完整[比较记录](tail.json.gz)保留来源、前提检查和各rank指标。

随后固定全部64head的原HF概率，分别按先展开Value与后投影两图连接原o_proj、HF norm和MLP。[先展开分支](value-result.json)逐位重建原HF h1，rank0两图与既有单die见证的CPU输入输出均逐位一致。后投影分支对HF h1的NRMSE为0.00348409，对实际CANN h1仍有0.00350671残差；attention残差为0.00184487。[完整计算](value-tail.json.gz)因此确认Value顺序会改变本层结果，但单改此顺序尚不能复现实际CANN，下一步核查实际SFA query、KV、索引、输出及QK/softmax舍入。

[SFA边界复验](sfa-capture-summary.json)再次在16rank通过原完整首帧对照。实际query为BF16 `[1,4,512]`、KV为73页×128行、有效长度8193、索引2048个且无负值，scale为0.0625；[输入元数据](sfa-inputs.json)保留每rank原shape/stride/页表与长度。按实际页表只读取有效KV，以实际SFA输出接原UV、o_proj和HF尾部，[最终h1在16rank均逐位复现CANN](sfa-result.json)。UV局部有15rank逐位一致，余下差异未改变该处最终结果。

固定实际query/KV的FP32 scores/softmax/PV候选，对实际SFA输出的context NRMSE为0.00028106–0.00101148，重算概率对HF为0.00452071–0.02901260；接尾部后仍未复现CANN或HF h1。仅将概率转BF16的候选也未闭合。[完整记录](sfa-tail.json.gz)保留逐rank指标。这是两种明确CPU算术的残差，不声称模拟CANN分块softmax；下一步比较QK分数的舍入位置及query吸收投影。

[实际openat记录](kernels.json.gz)与[SFA复验记录](sfa-kernels.json.gz)分别绑定16rank跟踪SHA、设备打开和62个kernel目录ELF的字节SHA，无未解析路径；它们不是kernel执行时间线。原CPU容器因Transformers探测未绑定的NPU库而导入失败，修正仅禁用CPU对照中的NPU后端探测，HF数学源码SHA保持匹配。本方已终止容器均已[移除](containers-retired.json)。当前仅证明第一层边界归因，完整模型精度、八输入和TPOT仍待验收。

[共享归档](archive.json)为105317691B，SHA256 `661f87965c34e8b9d68d9660530324629966a457b385d33fe493b8bc6b84561a`，位于 `/trt-a3-evidence-20261028/layer0-boundaries-source93b0e6c-20261029.tar.zst`。完整zstd解码、280成员清单及共享副本SHA均已核验，包含两次32份层Tensor、32份原跟踪日志、原HF首帧、Value fixture与来源记录。
