# 修正运行包后的整模型尝试

7.150.11.49、9382、TP16、CANN9.2、固定镜像`3892a691…`，源码`3fcfd8b`。SFA设备消费者为block1版本`b7062ab6…`，OPP显式保持overlay优先；同一冻结8K前缀、32个teacher输入及已完成的独立FP32-head HF参考。[启动脚本](launch.sh)、[源码摘要](launch.sha256)、[即时16die空闲记录](before.txt)与实际接收的runtime/SDK/prefix字节摘要均保留。

[最后观察](interruption.json)为68/96分片、加载进度16:59；rank报告与`.pt`工件均为0。[Docker事件](docker-events.jsonl)记录启动后1160秒收到SIGKILL 9，随后die 137、destroy；删除发起方未知。之后16die被另一个VLLM作业占用，本方未在该节点重建作业。它是环境中断记录，没有模型数值或新增TPOT结果。

174节点的[后续准备](checkpoint-174.json)先以[CPU顺序读取](hash-model-view.py)核验109文件、405008283962B，再做设备占用检查。核验用时123.310秒且SHA全部匹配；这是数据与页缓存准备。

## 9392完整复验

180节点以相同源码`3fcfd8b`、镜像和CANN9.2完成[109文件核验](checkpoint-180.json)，用时209.042秒；补齐只读模型视图的两项optional对象。[即时设备记录](before-180.txt.gz)确认16die空闲；[启动器](launch-180.py.gz)、[实际命令](docker-command-180.json)与[环境脚本](launch-180.sh)冻结本次输入。实际模型加载约1077秒，顺序读取准备没有消除加载开销。

[完整读回](full-180-readback.json.gz)绑定576个PT工件和16份原始rank报告SHA。16rank状态一致；每rank的35072项`reset_exact`检查全过。M1与M32仍从第13行出现差异，`serial_vs_wide`与`serial_vs_reset`各582项失败；每rank总计211428项检查、40807项失败。严格入口退出1，采集完整、精度未过。

[首行独立HF对照](first-hf-180.json.gz)为425/1096项失败，预测同为382，最终logits NRMSE为0.0771525；hidden1为0.00359480，hidden6为0.00921656，hidden7升至0.0277551，首次专家路由分歧在layer6。后续用[真实概率的Value重排见证](../mla-value-order-a3-20261029/README.md)区分MLA舍入边界与norm/quant输入差异。当前无新增TPOT结果。

原始工件已在180本机形成70989101922B归档，SHA256 `3188dc0b4a4a1c9b0ec4559e6c85567a9f7f931590b9ad8be4d0a63e358e502e`，完整zstd解码和成员清单检查通过；[共享副本](archive-180.json)已核对同一SHA，位于 `/trt-a3-evidence-20261028/full-fixed-v3-source3fcfd8b-20261029.tar.zst`。本方[完成的容器](container-180-retired.json)已移除，原始目录保留。
