# 常驻TP16首帧对照

真实A3 9382、TP16、一个冻结8K输入，同一常驻模型依次执行原attention与BF16 score/先展开Value组合；各取首个M1和精确reset。[完整读回摘要](result.json)中，两臂16rank全部状态一致、reset逐位一致，首帧HF精度均失败。最终logits NRMSE为0.0771525→0.0608584，首个hidden越界h7→h8，首个专家选择分歧layer6→14。满足预注册的传播预测，未测时延。

[收据包](capture-receipts.tar.gz)含32份rank报告、原始CPU读回、冻结设备/读回源码、实验契约、启动命令、环境与ELF打开记录。实际执行的`run_resident.py` SHA256为`18230d9a38c47c22c4bf67a44bdbcebb3360c04b18a8dbf28398035207361c58`；之后的本地控制流修正及缓存候选接入未用于本次结果。95个ELF均解析，`openat`证明成功打开，未记录执行时延。

[共享归档读回](archive-readback.json)绑定2,971,386,662B原始归档，SHA256为`aa37a79ff52ea7283d3940f7f739096dc4e49a56eefa75ca714495e96b8f8456`；64个首帧/reset PT重新从归档读取，全部匹配原读回SHA。归档另保留停止的逐层运行及单die缓存实验，分别标明范围。

逐层运行在每层准备约48秒、完成16/78层后主动停止，退出137；未取得完整模型结果。常驻对照替代了该路径。M1×32/M32、八输入、graph/SuperKernel与在线P/D继续验收；本轮没有新的TPOT结果。终止容器的退役记录见[清理收据](cleanup.json.gz)。

后续本地producer增加expanded控制臂和cached-prefix候选，[10项CPU入口检查](integration-cpu-check.json)通过，未在设备执行。`--recipe paired --control-attention expanded --candidate-prefix-values /banks/output`选择该对照；冻结执行目录须同时放置`run_resident.py`、`expanded_attention.py`、`routes.py`及以`cached_attention.py`命名的prefix-values `attention.py`副本。helper与bank均检查固定SHA；现有读回仍限定`control/expanded`，缓存候选另需读回接线。当前源码身份见[接入收据](integration-source.json)。
