# Attention舍入与误差传播

范围为真实A3的TP16首个M1、一个冻结8K输入，以及首个不同rank的单die重放。未测时延，正式精度门禁未通过。

固定实际QL/QPE/KV、选择顺序及BF16概率，仅把缩放后的score转BF16，概率对HF的最大NRMSE从0.028113降至0.001300，跨rank中位数从0.007736降至0.00002792，达到预设预测。原始报告曾把FP32概率作为控制，混入第二个变量；[单因素重算](score-single-factor.json)使用BF16概率控制，原始报告保留在`evidence/layer0-tail-v7.json.gz`和`v8`文件中。

再按原HF前缀/当前行边界先展开Value，CPU组合逐位重建HF h1。相同组合在[真实TP16](evidence/layer0-expanded-v1-detail.json)把h1 NRMSE从0.003595降至0.001961，16rank一致，21项结构与输入控制检查全过；原CANN的`merged`精确比较失败如实保留。实际attention输入接原HF尾部仍逐位重建该CANN h1，残差留在组合attention内部。helper固定`allow_hf32=False`；TP16安装前的值未单独记录，该组合不作为纯attention单因素结论。

[单die重放](evidence/layer0-expanded-unit-v1/report.json)绑定物理die13、首个不同rank2的实际输入。score、概率与逻辑位置对齐逐位匹配独立CPU；Value展开有小量差异，最终PV仅1/1024项不同，且逐位复现TP16组合。每臂重复32次的范围仅为PV，QK/softmax和Value展开各计算一次。改用FP32 Value投影没有改变PV，停止该精度旋钮。

[FP64干预](evidence/layer0-expanded-fp64.json)仅检查已观察到的Value分歧坐标并集：前缀1188项中，CPU与设备BF16分别有387与998项匹配FP64舍入。把CPU Value替换为设备值，或只替换该并集为FP64舍入，都逐位重建设备PV。因此不把逐位追平CPU作为后续扩层条件；HF计算图及既定阈值保持不变。这一局部样本不支持普遍评价参考后端。

192节点的[前三层复验](node192/evidence/first3-expanded-192-v1-check-v2.json)取得48个PT和16份报告。h1/h2/h3 NRMSE为0.001961/0.004716/0.006684，逐层16rank一致，均在原hidden容差0.01内；h3相对原0.007751只下降约14%，未达到预注册`h3 < 0.0054`的30%改善预测。原CANN逐位比较的hidden、KV及indices变化单列，结构检查全过。初次CPU读回漏列`index_keys`比较名而退出1；修正读回分类后重算，原报告与脚本均保留，未重跑设备或修改数值阈值。

本轮没有整模精度或TPOT通过。后续优先沿原layer-major采集推进完整78层首帧，定位首个越过固定容差或改变离散选择的边界；M1/M32、多输入及在线门禁继续独立验收。当前主KV是BF16，INT8 indexer不能作为开发packed主KV分支的理由。原driver已有MoE及共享indices生命周期，扩层可用逐层factory安装/恢复helper，避免dense专用observer和跨层持有restorer。

原始首层PT、单diefixture/输出、CPU报告、冻结源码、设备打开记录及清理收据保存在[共享归档清单](receipts/attention-rounding-archive.json)：64,838,269B，SHA256 `827cc172eebf2df5be0e88cf97d907c10c78069ecf68d7b20066f5d7a70ca6c5`。`openat`记录只证明设备和ELF被成功打开，不代替执行或计时证据。75个TP16和23个单die ELF均已解析、未解析路径为0；相关容器均在终止后退役。

前三层原始工件另见[192归档](node192/receipts/first3-expanded-archive.json)：292,823,804B，SHA256 `678719e25ec3d2f880e22117d4173ad4b31d2b88e2c037c76f379425d27f47e1`。该节点重新核验109个权重文件、405008283962B；16份trace的75个ELF全部解析，容器退役后的8组NPU进程表均为空。
