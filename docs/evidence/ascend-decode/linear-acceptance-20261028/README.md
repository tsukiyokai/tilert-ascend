# 线性接受数独立核验

原在线收据校验只核对接受数、交付和缓存位置之间的一致性。完整8K→1024夹具中，仅令第一轮 `candidates[1] != predictions[0]`，仍可冒报接受4或32个。现在从实际候选与目标预测的首次 mismatch 独立重算接受数、末轮预算裁剪和交付 token；串行路径沿用原检查，树收据拒绝进入这条固定长度线性门禁。

实际执行 `python tests/glm52/check_decode_timing.py --output /tmp/trt-linear-acceptance-check-20261028.json`，28项通过、0失败、0错误、未导入Torch；[收据](cpu.json)绑定实际源码SHA。覆盖 width1/2/4/32、单轮/批量命令、接受/拒绝/混合与末轮裁剪。当前在线CLI仍只接受 width1/2/4，width32是函数接口验证。此结果不包含NPU数值或时延。
