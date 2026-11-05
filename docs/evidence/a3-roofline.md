# A3 GLM-5.2 roofline inputs

审计日期：2026-10-16。仓库 `f554fa57d0a69e7a44ac4f3134d0134b4964e848`；模型目录为 `a3-gs:/data/shanshan/trt-asc/models/GLM-5.2-w4a8c8`。容量和精度来自 [实际 safetensors headers](a3-model-bytes.json)，覆盖 96 个文件、235842 个 tensor；未读取 tensor payload。输入 config、量化描述、index 的 SHA256 保存在 JSON 中。header 合计 404765007360 B；index 的 `metadata.total_size=404689509888` 不含 optional `rot.weight` 的 75497472 B。

在仓库根目录重现 header 审计（解析发生在服务器）：

```bash
ssh a3-gs python3 - /data/shanshan/trt-asc/models/GLM-5.2-w4a8c8 \
  < docs/evidence/model_bytes.py > /tmp/a3-model-bytes.json
```

## 流量模型

| 主干部分 | 单 token 逻辑读取，GB |
| --- | ---: |
| Attention 投影、norm、kvB | 14.033071104 |
| 21 个 indexer | 0.218158080 |
| 前 3 层 dense FFN | 0.680214528 |
| 75 层 router | 0.236006400 |
| 75 层选中的 8/256 experts，含量化元数据 | 11.619532800 |
| 75 层 shared expert | 2.837299200 |
| LM head，按 BF16 预转换 | 1.903165440 |
| 其他 norm、一个 embedding row | 0.003870720 |
| 合计 | 31.531318272 |

以上为按 checkpoint 分类的逻辑权重。当前 CANN router 权重为 BF16、输出为 FP32；W4 scale 使用 INT64 carrier，并保留当前 rank 的 FP32 MSD 补偿；普通 W8 scale 为 BF16，Prolog scale 为 FP32。按这些实际表示，M=1 单 rank 权重为 3,606,269,952 B，加 MLA/index KV 后逻辑读取为 3,817,820,160 B。[表示审计](ascend-decode/cann-traffic-source-audit-20261017.json)、[估算器](../../tests/glm52/estimate_traffic.py)和[分项结果](ascend-decode/runtime-traffic.json)绑定执行源码与测量标尺；专家并集及模型 HBM 流量尚未实测。`rot.weight` 为 75.5 MB，仅在 MTP 的 hnorm 前执行；其与 `optional/quarot.safetensors:global_rotation` 的用途见[权重契约](ascend-decode/weight-contract.md)。

8K 下所有主干层均执行 top-2048 attention；`first_k_dense_replace=3` 只区分 FFN。[golden_attention](../../tilert/models/glm_5_2_rocm/ops/flash_sparse_mla.py) 的 `kv_len>topk` 分支要求 indices。按每 rank 的四个 heads 共享一次读取，MLA KV 为 `78×2048×(512+64)×2=184025088 B`。开启 `indexer_kv_dtype=int8` 后，配方指定的 16 层 index cache 每位置为 128 B＋2 B FP16 scale，0/1/2/70/74 五层保持 256 B BF16；一次完整扫描为 `8192×(16×130+5×256)=27525120 B`。[层列表与分配源码](ascend-decode/weight-contract.md)。合计每 rank 4.055886336 GB；此模型假设权重流式读取，不计 L2 收益、MSD scratch 及 activation 流量。

[Huawei CloudMatrix-Infer §3.3](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1) 给出每 die 64 GB／1.6 TB/s、每 package 双 die、每 node 8 package；本机 `npu-smi info` 报每 die 65536 MiB，即 64 GiB。采用十进制带宽，`T_HBM=4.055886336/(1.6×η) ms`：η=1 时 2.53492896 ms，η=0.7 时 3.62132709 ms。该论文的 UB 网络延迟不用于 a3-gs HCCS 的归约预算。

## HCCL 时延依据

[A3 盘古 v2 优化记录](https://wiki.huawei.com/domains/104592/wiki/240629/WIKI2026031610413429) §2 给出 EP AllReduce：384 个 INT64、3072 B，AICPU 27 µs，AIV 12 µs；162 次调用对应 2430 µs 收益。§3 的 EP rank 列表为 6144、6148、…、6204，共 16 ranks、跨节点。此记录未附精确 CANN 版本及原始 trace；plan 用 12 µs 作邻近形状预算参数，不将它标作本机 12 KiB 测量。

本机另有 2026-07-13 的 TP16、1 KiB FP32、CANN 9.1／PyTorch 2.10 测量，稳态 p50 369.4 µs；[本地报告](/home/shanshan/note/log/2026/0323_pg-first-loss/report/a3-16rank-full-link-baseline-20260713.md) 与 `a3-gs:/home/shanshan/hccl-full-link/results/a3-16r-39a0c028-formal01` 保存原始记录。计时包含 Python 与前后 synchronize，故不作为设备归约 α。

TP16 主干的两次 BF16 hidden-state SUM 为 `78×2=156` 次、每次 12288 B。未融合的串行模型为 `T_HBM+156α`；α=12 µs 时得到 4.407／5.493 ms，即 227／182 tok/s。完全重叠时取两项最大值，采样通信、解包、计算、调度和额外重读在实测中追加。

## 本机定标

在空闲的 `trt-asc` 内，使用 CANN 9.2.0 的 `/opt/ascend/cann-9.2.0/toolkit/tools/hccl_test` 源码编译测试副本。MPI 和测试可执行文件需先准备。命令仅测选定机器的 16 个 die：

```bash
HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=200 \
mpirun -n 16 ./bin/all_reduce_test -p 16 -b 12288 -e 12288 \
  -d bfp16 -o sum -t 1 -w 20 -n 100 -c 1
```

M=4 改为 `-b 49152 -e 49152`。`-t 1` 预下发后放行，使用设备 event 计时；它的 warmup/iters 上限与展开模式见 [官方参数说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/devaids/hccltool/HCCLpertest_16_0005.html)。原工具仅打印 root rank 的批均值：`opbase_test/hccl_allreduce_rootinfo_test.cc:107–113` 计算 `total_time/iters`，`common/src/hccl_opbase_rootinfo_base.cc:148–187` 限 root 打印。测试副本须在计时窗外收集 16 rank 批均值并取 MAX，再重复 100 批统计 p50/p95；单次延迟分布另行采样。记录实际 data_size、版本、拓扑，再测 156 次带依赖的 ACL Graph。验收目标与当前测值分别保存。
