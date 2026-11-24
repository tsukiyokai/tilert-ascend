# A3 消费空间中的完整 checkpoint 核验

固定 ModelScope `Eco-Tech/GLM-5.2-w4a8c8` revision `67e2b7157ac7d8ce1d15d7251853ce146819ceb9`。实际消费者只读视图 `/models/glm52` 共109文件、405008283962字节，全部大小与完整SHA256匹配固定版本清单；其中96个索引引用的safetensors还检查header长度、每张量形状/字节区间、无重叠/缺失/多余数据及索引覆盖。既有107文件保持原字节，缺失的 `optional/.gitkeep` 与 `optional/quarot.safetensors` 在独立自有目录补齐。

[消费者读回](raw/receipts/checkpoint/consumer-view.json)绑定当前96文件完整读取结果与其余13文件的消费者内实际SHA；[核心文件报告](raw/receipts/checkpoint/summary.json)保留逐文件身份和本次两阶段读取的来源。清单取自仓库保存的固定版本API响应，SHA为 `fd225ae2…`；本次没有重新请求在线API。未把目录存在、`.partial`后缀或下载进度当作完整性依据。

SDK 9.2、NNAL、匹配vLLM/vLLM-Ascend及冻结8K前缀从已有资产经归档SHA校验后落到A3本地盘。实际16个设备均读为9392，独立9.2进程成功导入匹配的 `vllm_ascend_C`；[运行环境预检](raw/receipts/tp16-runtime/)与模型推理分开。本组没有执行整模型数值或性能门禁。
