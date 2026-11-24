# 真实HF概率下的BF16 Value重排

固定首行layer0/rank0的四个head、8193个位置及HF返回的BF16概率`P`，从原始checkpoint和同一冻结prefix重建Value。8192行prefix与新token分开投影，保持原HF的BF16边界；CPU设置同为`mkldnn=False`。[输入身份](input-identity.json)、[实际命令](docker-command.json)与[启动环境](launch.sh)保留。

记`R`为BF16舍入、`C`为latent、`U`为Value权重。比较先展开的`R(P @ R(C @ U))`与后投影的`R(R(P @ C) @ U)`；固定全部概率与位置顺序。[执行源码](../../../../experiments/mla-value-order/run.py)使用同一CPU算术生成两臂参考。

实际9392物理die13、CANN9.2、Torch/NPU2.12上，[两臂各32次](report.json)均逐字节重复一致。先展开与CPU逐元素相同；后投影相对自身CPU仅一项相差`9.313225746154785e-10`，NRMSE为`6.81242e-8`。两种数据流之间510/1024项不同、NRMSE为0.00333698，最大绝对差`7.62939453125e-6`。[openat读回](kernels.json)绑定实际`/dev/davinci13`与三个加载ELF的SHA；完成容器已移除。

[CPU后端对照](cpu-backends.json)记录默认开启mkldnn时同一PV有532项偏离FP64，关闭后全等。既有HF生成器和`inject/step`原已关闭mkldnn；修正的是新用例的设置，早期用例保留在归档中。该见证说明此输入上的Value重排不等价；对hidden1的贡献继续用实际attention/Oproj输入验证。未测时延。

[完整输入、输出和脚本归档](archive.json)位于共享NFS `/trt-a3-evidence-20261028/mla-value-order-source1aee8b7-20261029.tar.zst`，11867054B，SHA256 `b205647ea3e2ae6168278c737e0dfe056ada1f6f2eadcec5cced3294bd593c7c`；zstd完整性与共享副本SHA均通过。
