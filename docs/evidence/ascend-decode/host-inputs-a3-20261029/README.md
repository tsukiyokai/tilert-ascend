# 合包输入准备的A3对照

真实A3 9382、物理die12，两臂执行当前`CannModel._prepare_inputs`。控制分别准备和上传tokens、length、slots及positions；候选保留固定typed views，以一个pinned packet上传四字段。纯host输入走packet，Tensor/混合输入继续原路径。计时包含fixture的tuple/position生成、实际准备函数及返回检查；构造真实CommonMetadata，但builders为空，不执行权重、模型或TP。

| width | 实现 | 墙钟p50/p95 µs | event p50/p95 µs |
| -- | -- | -- | -- |
| 1 | 分别准备 | 725.280 / 818.148 | 554.520 / 625.480 |
| 1 | packet | 626.012 / 668.223 | 418.970 / 469.460 |
| 32 | 分别准备 | 767.251 / 886.960 | 604.400 / 698.180 |
| 32 | packet | 628.517 / 753.085 | 424.150 / 570.100 |

预注册要求width32双钟p50均至少下降20%、p95不退。实测墙钟下降18.082%、event下降29.823%，`prediction=false`；不改门槛。两width均59/60对墙钟更快，配对节省p50为94.603/137.520µs。保留数据流候选，尚不按该预测准入性能接入；[真实模型消费者](../host-inputs-model-a3-20261029/README.md)另行核对。

两独立进程、五类私有缓存，12对预热、60对AB/BA测量；每臂前完整读写256MiB并同步，压力在双钟之外。profiler和计时graph关闭，未锁频。[原始报告](report.json.gz)保留全部样本、输入签名、地址及实际库/缓存SHA。[独立读回](readback.json)从number/width重建全部288份输入签名，重算统计及预测，核对真实容器和worker均退出0。生产器另检查两臂九种width各32次排队graph消费，共576次；未保存逐replay原始张量，独立reader只核对这些源码绑定的现场断言。

[冻结收据](receipts.tar.gz)保存契约、命令、设备快照、退出及环境预检失败原文：首次缺SDK挂载，第二次缺vendor路径，第三次实际vendor导入需要Triton查询NPU架构；随后在空闲A3执行通过。[验证收据](validation.tar.gz)含8项CPU输入/alias回归、15项reader负例、15项bench回归及CLI检查。默认关闭，可由`bench_cann_mtp.py --packed-host-inputs`选用；完整模型精度及TPOT保持未验收。

[共享归档](archive-readback.json)为55,180B，SHA256为`9a6476c4366d8547512c5f69d059a7571e3ffc825cc888174397f0e7b1a31fd6`，35个成员从归档重读均匹配原始SHA。[清理](cleanup.json)仅移除本方已退出的四个容器，原始bind文件保留。
