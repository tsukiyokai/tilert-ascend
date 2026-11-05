# SFA gather metadata in UB

假设：在 A3 的相同 SFA shape、输入和编译配置下，将 `MergeKv` 对
top-k 和页表的逐项 GM 标量读取改为两次 MTE2→UB 搬运，可减少该算子的
典型延迟。先验收变化输入的逐位输出，再以 profiler 关闭的配对 A/B
检验至少 5% 的 p50 收益；这只是待检验阈值，尚无设备收益记录。

`metadata.patch` 只修改 CANN arch22 的 vector service；原算术、KV
pair 次序、有效范围、因果裁剪、物理页映射、32-row gather ping/pong、
MMAD 和已有同步保持原样。每次 `MergeKv` 将该 AIV 可能读取的 top-k
分片和当前页表搬进已有的 8 KiB `v0ValidSizeBuff`，再等待独立的
`MTE2_S(1)`。不增加 UB 分配、全局 barrier 或跨核 flag。

| 字节区间 | 用途 |
| -- | -- |
| `[0,512)` | 原本地 valid-count 发送 |
| `[512,1536)` | 原两 AIV valid-count 接收 |
| `[1536,2048)` | 原 Vec2 FP32 broadcast scratch，最多 16×32 B |
| `[2048,3072)` | 当前 AIV 的至多 256 个 INT32 top-k 索引 |
| `[4096,8192)` | 至多 1024 个 INT32 页表项 |

页表按当前 kernel 的输入刷新，不跨请求缓存。top-k 分片起点为
`s2Idx*s2BaseSize+s2GmStartOffset`；长度包括奇数尾部可能读取的第二个
pair 成员，并截断于 `sparseBlockCount`。原有索引上界判断先于 UB
读取；页表读取仍先检查逻辑位置与因果界。缓存范围之外沿用原 GM
读取。两片 UB 与 KV ping/pong、原 valid-count 和后继 Vec1/Vec2
scratch 均不重叠，下一次 gather 才覆写本次元数据。

仅在 batch=1、TND/PA_BSND、V_TEMPLATE、非 flash-decode、BF16
KV/输出、RoPE64、head512、一个 KV head、page128、sparse-block1、
preload2、`0<s2BaseSize<=512`、`0<maxBlockNumPerBatch<=1024`
且本分片不超过 256 项时启用。其他配置保留原路径；每次调用先清除
本次缓存启用状态，避免 fallback 读取前次缓存。

## CPU 检查

```bash
python3 experiments/sfa-metadata/check.py \
  --source /home/shanshan/repo/cann/ops-transformer \
  --output /tmp/trt-sfa-metadata-host.json
```

检查从原版和候选头文件抽取实际 C++ 方法，用 g++ 执行带 GM/UB
边界、DMA 完成和 event 配对断言的 host 夹具。比较实际 `MergeKv`
传给未改动 `CopyInKv` 的 pair 和地址；覆盖两 AIV、四个 s2 分片、
width1/8/32、负索引、奇数尾部、页边界、变化输入、fallback 和后继
scratch 覆写。BF16 与不支持的 FP16 分别检查，五个真实代码变异
须全部失败。该夹具不模拟设备 payload 算术、并发或时延。

## 真实 A3 的最小入口

激活 CANN 9.2 与匹配的 Torch/torch_npu 环境，保留实际 CANN 源码
身份。`prepare.py` 只复制四个 SFA arch22 头；`build.sh` 在原有
Prolog/SFA 构建入口前插入该 include 路径，其他 vendor 头继续直接
读取 `OPS_TRANSFORMER_ROOT`。不安装或覆盖全局算子。

```bash
python3 experiments/sfa-metadata/prepare.py \
  --source "$OPS_TRANSFORMER_ROOT" --output /tmp/trt-sfa-metadata-overlay
bash tests/glm52/whole_layer/attention_build.sh /tmp/trt-sfa-control
SFA_METADATA_OVERLAY=/tmp/trt-sfa-metadata-overlay \
  bash experiments/sfa-metadata/build.sh /tmp/trt-sfa-candidate
ASCEND_RT_VISIBLE_DEVICES="$FREE_A3_DIE" PYTHONPATH=. \
  python3 tests/glm52/whole_layer/attention_check.py \
    --library /tmp/trt-sfa-candidate/whole_layer_attention.so \
    --widths 1 8 32 --sequence 8192 --capture --no-timing \
    --output /tmp/trt-sfa-candidate-check.json
```

仅在实时核对为空闲的 A3 die 上运行。上述检查是原生 Prolog/SFA
对同输入 CANN 链的数值入口，尚不能证明配对收益。正式计时需各臂
独立编译缓存和实际 ELF 身份，复原各臂会改写的 KV，以 ABBA 次序
交替测量至少 30 对，并轮换超过 192 MiB 的 KV/工作集；记录事件与
墙钟的 p50/p95/p99、失败和占用。只有设备逐位对照与上述配对测量
成立后，才扩大到实际 decoder、TP16 和在线 TPOT 门禁。
