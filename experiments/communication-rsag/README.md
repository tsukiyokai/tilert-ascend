# RSAG 所有权重排实验

本实验比较同一混合 CUBE/AIV probe 的两种通信实现：控制为 BF16 全互发后在每个 rank 执行现有 FP32 树；候选把输出按 512B 数据行分给 owner，先分发输入、在 owner 上执行相同树，再广播最终 BF16 结果。原 probe 的两次规约、中间 BF16→FP32 CUBE 投影和紧邻消费者均保留。这里没有串行和控制，也没有把控制命名为 HCCL。

可证伪主张是：保持源值每个输出位置一次 FP32 转换、`tree_plan.h` 的结合顺序和最后一次 BF16 舍入后，减少跨 rank 数据及重复求和，可能降低完整 probe 的时延。每次规约的 mixed joins 从 3 次增加到 5 次；流量下降不能推出时延下降，更不能推出 TPOT。比较只针对同一既定 TreePlan 的两臂；该固定树已经不能代表所有 shape 的实测 HCCL 顺序。

## 当前设备证据

根线程已在真实 A3/CANN9.1 构建两臂，完成 TP2/TP4、rows=1/12/32 的 tagged/random、eager/graph、延迟 rank 和连续 parity 门禁：[读回记录](../../docs/evidence/ascend-decode/rsag-a3-20261028/readback.json) 含 12 份 rank 报告、324 条数值记录；162 条跨臂 input/weight/first/output SHA 一致。这是合成完整 probe 数值证据，没有 TP16 或性能结论。[原八文件快照](../../docs/evidence/ascend-decode/rsag-a3-20261028/raw/source/experiments/communication-rsag/) 保留当时源码、旧 README 和旧 manifest 的历史身份，不代表本 README 的后续结论。

新增 TP4 HCCL 特殊 fixture 已暴露实际边界：四 rank 输入 `[1,2^-25,-1,0]` 在 rows1/12 的 HCCL first 与固定树相同、与串行和不同；rows32 的 HCCL first 与串行和相同、与固定树不同，四次 eager/graph 均复现。finite fixture 的 native first 匹配 CPU 树，CUBE 后 output 却与 CPU oracle 不同；edges 的 first 也存在 CPU 差异，两臂模式一致。这些差异全部保留，不能以有限 fixture 宣称全 BF16 语义等价，也不能把 normal gate 的 `pass` 解读成 HCCL 可替换性。`hccl` 字段是独立观测，顶层 `pass` 仍只统计默认 normal gate。性能入口保持既定树，不改算法迎合 HCCL。

## 源码与入口

`prepare.py` 校验原七个通信文件、`tree_plan.h`、`reduce.inc` 及已写协议的 SHA256，再生成 `OUTPUT/{control,candidate}/source/`。只复制这些必要文件和本目录的布局/候选 device 片段，不复制 vendor 树，不读取或修改原二进制。控制直接包含原 `reduce.inc`；候选只改输入/输出位置及分片长度，FP32 栈、leaf 顺序、Add 顺序和 BF16 CAST_RINT 与它相同。

两个 arm 分别使用 `trt_rsag_control`、`trt_rsag_candidate` 命名空间，包括独立的 layout 命名空间、头保护、kernel 名和 C ABI 前缀。C ABI 保留原参数布局：`initialize(rank,ranks,endpoint,heap_bytes)`、`prepare(rows,rank,ranks,blocks,&owner)`、`execute(owner,stream,partial,weight,first_sum,output)`、`destroy(owner)`、`finalize()`、`last_error()`。只有前缀改为 `trt_rsag_{control,candidate}_probe_`。CANN/SHMEM 的公共运行库继续共享；两个 probe 的符号互不替代。`build.py` 验证冻结源码，分别调用原构建脚本，核对 private 导出并记录 ELF、源码、SHMEM 库及编译日志哈希。

在仓库根目录执行 CPU 检查与源码冻结；输出目录必须不存在：

```bash
python3 experiments/communication-rsag/check.py --output /tmp/rsag-source
python3 experiments/communication-rsag/build.py \
  --prepared /tmp/rsag-source \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --shmem /path/to/shmem-24c15d21-source \
  --shmem-library /path/to/libshmem.so \
  --compiler /path/to/CANN9.1/bisheng
```

构建依赖与原 probe 相同：aarch64 CANN 9.1 的 bisheng、host C++ 编译器、AscendC/Matmul 头、SHMEM 对应源码及动态库。保留原 `dav-2201`、`-O2`、`-ffp-contract=off`。`CANN_TARGET_DIR` 可明确指定 SDK target 根目录；不要混用其他 SDK 的 compiler/header/library。构建不初始化设备。编译若失败，真实日志位于每个 arm 的 `build.log` 及 `build/` 下，不改算术规避错误。

`device_check.py` 是单臂数值门禁，不计时。它显式加载所选私有 C ABI，通过 gloo 校验全 rank 配置一致，prepare 后同步设备并全 rank rendezvous，然后在 eager 和 graph 下改变输入、轮换延迟 rank，检查两次规约边界、输入/权重未变及输出前后各 512B 哨兵。另有默认八次 graph 连续 replay：输入在整个批次不变，中间没有全 rank barrier 或设备同步，轮换延迟主机 rank，以实际覆盖 parity 复用。每份输出记录全张量逐字节比较、SHA256、前 32 个 BF16 位模式及加载库身份。使用真正空闲的设备时，由根线程按环境设置可见 die；例如分别启动两个独立 world：

```bash
torchrun --nproc-per-node=2 --master-port=29601 \
  experiments/communication-rsag/device_check.py \
  --prepared /tmp/rsag-source --arm control \
  --endpoint tcp://HOST:29602 --rows 1 12 32 --blocks 4 \
  --output /tmp/rsag-control-check
torchrun --nproc-per-node=2 --master-port=29603 \
  experiments/communication-rsag/device_check.py \
  --prepared /tmp/rsag-source --arm candidate \
  --endpoint tcp://HOST:29604 --rows 1 12 32 --blocks 4 \
  --output /tmp/rsag-candidate-check
```

默认门禁使用 tagged/random 有限 BF16 输入。`--hccl-compare --hccl-fixtures witness finite ties edges` 额外创建 HCCL group，执行设备 `BF16→FP32 HCCL SUM→BF16`，分别记录 first、第二次 FP32 sum、CUBE 后 output、固定树 oracle 与串行 oracle。HCCL 相等与否不预设为门禁条件，所有差异都进入报告；边界见上节。

旧 ELF 的编译 manifest 绑定旧 driver SHA `0bda1919…`；新 driver 不会静默跳过它。复用旧构建时须显式传 `--frozen-driver experiments/communication-rsag/frozen/device_check_v1.py`，程序实际核对冻结原文、编译源码和 ELF 哈希，并分别记录 manifest 中的旧 driver 与本次新 driver 身份。新构建使用当前 driver 时不需要此参数。TP4 特殊 fixture 示例：

```bash
torchrun --nproc-per-node=4 --master-port=29611 \
  experiments/communication-rsag/device_check.py \
  --prepared /tmp/rsag-source --arm candidate \
  --frozen-driver experiments/communication-rsag/frozen/device_check_v1.py \
  --endpoint tcp://HOST:29612 --rows 1 12 32 --blocks 4 \
  --hccl-compare --hccl-fixtures witness finite ties edges \
  --output /tmp/rsag-candidate-hccl
```

## TP16 配对性能入口

`device_bench.py launch` 先写 `registration.json`，再按 ABBA world 轮转；每个 world 是独立 torchrun 进程组，只加载一臂。每 rank 五类缓存目录先创建，再导入 torch/torch_npu，world/rank 路径均独立；源码导出没有 `.git` 也可运行，身份只依据实际源码、编译 manifest 和 ELF SHA。默认 rows1/12/32、24 blocks、10 次 graph 预热、每 world 每 rows 30 个样本、15 个 ABBA cycles，即 30 个相邻 world 对；A0↔B1、A3↔B2 配对。`--abba-cycles 3` 只生成探索统计，不能接受性能假设。

每个样本提交一个包含连续 32 次完整 probe 的 graph，event 与同步 host 区间都按 32 归一化。每次完整 probe 仍包含两次规约及 CUBE；这是连续执行的均摊值，不是一次冷调用或 decoder 单轮时延。样本前改变输入、建立独立 CPU oracle、恢复输出哨兵，并对独立 256MiB 缓冲完整读写，检查首尾值并同步；这些都在计时外。压力只施于批次之前，潜在首调用占比为 1/32，其余 31/32 不能称为 cold；读写压力本身也不是硬件逐 cache-line 逐出的证明。

gloo barrier 在计时外仍可能留下 rank 启动偏斜，因此所有 rank 再共识未来的 CLOCK_MONOTONIC 截止点，计时外短自旋后提交。保留实际 deadline 接收、start、enqueue before/after、同步完成的全部 per-rank ns；不能依据调度目标认定同时启动。event 指标为每样本全 rank 最大 event span/32；host 指标为全 rank 最晚同步完成减最早开始/32。每次收集全 rank 完成后再读取数值，额外 gloo 时间不进入区间。event、graph 和 replay 明确绑定同一个 NPU stream，压力和读值之前的全设备同步覆盖非默认 stream。

预注册接受条件为：每个 rows 的 host/event 都须 p50 至少降低 10%、p95 不回退，且相邻 world 的 p50 配对节省中位数至少 10%；至少 30 对，所有数值、rank 完成和身份检查通过。最坏实际提交时间包络除以 32 必须不超过控制 event p50 的 1%，否则整组标为偏斜无效，保留所有 raw，不剔除异常值。world 前后都读取并解析完整 `npu-smi info`：命令错误、进程表缺失、任何外部/遗留 NPU 进程都会中止；summary 再核对快照原文 SHA 和 idle 表。它仍不能发现恰好出现在两张快照之间的短暂外部任务，根线程需保持设备独占。

```bash
python3 experiments/communication-rsag/device_bench.py launch \
  --prepared /tmp/rsag-source \
  --frozen-driver experiments/communication-rsag/frozen/device_check_v1.py \
  --endpoint-host HOST --ranks 16 --rows 1 12 32 --blocks 24 \
  --abba-cycles 3 --output /tmp/rsag-tp16-pilot
python3 experiments/communication-rsag/device_bench.py summarize \
  --output /tmp/rsag-tp16-pilot
python3 experiments/communication-rsag/test_bench.py -v
```

launcher 不使用 profiler；所有实际 worker 命令、per-rank JSON/增量 JSONL、全 raw、输入/输出逐字节摘要、时钟基准、缓存路径、源库身份和占用快照均保留。新的 Python 入口可以复用先前 ELF，但会显式记录已验证的旧 driver 与新运行源码差异。两臂最终普通输入 oracle 必须分别通过；性能成立也只说明固定 TreePlan 两臂在此合成区间的差异。

## 分区、分配与算术

条件限定为 hidden=6144、rows=1..32、ranks=2..16、连续 BF16 物理 ND、单个 window 内 rank/容量一致且各 rank 规约顺序一致。令 E=6144×rows、B=E/256；owner i 获得 `floor(B/P)+(i<B%P)` 行，开始位置为 `(i*floor(B/P)+min(i,B%P))*256`。M1/TP16 的前八个 owner 各 512 元素，后八个各 256；不存在 384 元素均分。每个 owner 以至多 2048 元素的有效 tile 读取，不读 slot 尾部填充、不用 +0 种子、不做中间 BF16 舍入。UB 为 BF16 2048 元素和五个 FP32 2048 元素槽，共 45056B。

scatter/gather 都有双 parity payload 和独立的双 parity ready words。capacity C 的 scatter slot 为 `ceil(24*C/P)*512` 字节；gather 每个 plane 为 `2*C*6144` 字节。epoch 位于 0；两组 ready words 各用 `2*P*64` 字节，payload 从下一条 512B 边界开始。C32/P16 下候选 window=1577472B，控制=12585472B；控制比原布局增加 448B 对齐填充，算术和三个 joins 不变。所有 payload 起点、步长和有效传输长度为 512B 整数倍。

两臂用 `aclshmem_align(512,bytes)` 分配，随后 `aclrtMemset` 清零并 `aclrtSynchronizeDevice`；`align` 本身不保证清零。host 对内部 scratch 与外部 tensor 检查 512B 对齐。每次 prepare 返回后，driver 必须等所有 rank 的清零和准备完成才允许第一次 execute；本目录门禁实际执行此 rendezvous。

平均每 rank 的 off-rank payload 从 `2*E*(P-1)` 降至 `4*E*(P-1)/P` 字节，P16 为 8 倍缩减；这不包括 ready words。M1 分片不均时单 rank 为 `2*E+2*owned*(P-2)`，即 26624B 或 19456B，不能给每个 rank 都写平均 23040B。控制对应 184320B。该流量恒等式没有硬件带宽或时延假设。

## 五阶段顺序与同 parity 复用

1. AIV0 推进本地 epoch；所有驻留 AIC/AIV 执行 publish 与 join 1。各 rank 的 epoch 只由相同串行调用顺序对齐。
2. AIV 把自己的有效输入片段 put 到各 owner 的本 sender slot；每次异步 put 后立即 `aclshmem_quiet()`，再复用显式 narrow UB。publish+join 2 后仅 AIV0 发布 scatter-ready。
3. owner 的每个 tile 按原树逐 leaf 等待 scatter-ready，读取该 sender 的有效片段，FP32 求和后只舍入一次到本地 gather。MTE3→MTE2 fence 排空 tile 写；publish+join 3 后所有本地 AIV 才可读完成片段。
4. owner 广播自己的 gather 片段到其他 rank；自身已有完成片段，跳过 source/destination 自重叠。每次 put 后 quiet；publish+join 4 后 AIV0 发布 gather-ready。
5. 每个输出 tile 等待所有覆盖它的 owner ready，再从 gather 复制有效元素到外部输出。MTE3→MTE2 fence 及 publish+join 5 完成后才返回原调用者的 CUBE/后续消费者。所有 AIC/AIV 执行相同五个 joins；AIC 不发 SHMEM 操作。

同 parity 的 e→e+2 覆写依赖跨 rank 的两波 ready，而非本地 barrier：sender S 能发 scatter(e+2)，必已完成本地 output(e+1)；它因此见过每个 owner 的 gather-ready(e+1)，各 owner 已读完 scatter(e+1)，也早已读完 scatter(e)。故 S 覆写任一 owner 的旧 sender slot 时，旧读者已退出。owner O 能写 gather(e+2)，必已见到所有 sender 的 scatter-ready(e+2)；每个 sender 都已完成自己的 output(e+1)，也完成 output(e)。故所有目的 rank 的旧 gather 读者均已退出。相同前置关系也保证 ready(e+2) 不会替换尚有读者等待的 ready(e)。证明对任意正整数 e 归纳，不依赖随机调度数量。

SHMEM `24c15d21` 的显式-UB MTE put 将最后一次 MTE3 留给调用者排空；此处每次调用均在 quiet 后才重用 narrow。原 `CommunicationPublish` 的 PIPE_ALL+DDR barrier 位于每个 join 前；远端完成由 quiet→ready→精确 epoch 等待建立。FP32 树的每个 leaf MTE2→V、Add 顺序和 V→MTE2 保护沿用原 reducer；CUBE 后的 NarrowProjection 与两次规约衔接仍为原 probe 源码。

## CPU 证据及边界

`check.py` 真正编译生成两臂的 C++ header/TreePlan，执行 capacity=1/12/32、每个 rows≤capacity、ranks=2..16 的 675 组布局/分片检查，含非整除 owner 长度、512B 对齐、有效 tail 与 window 边界；3840 组 BF16 输入检查两臂及独立轮次归约的 FP32 结合顺序。它另运行 600 个每 rank 八 epoch 的随机异步状态调度。负例实际触发 ready 早于 payload、未 quiet 就复用 UB、旧 scatter/gather 活读者被同 parity 覆盖、live ready 被替换、消费者过早及 epoch 生命周期越界；另验证不一致 epoch 无法满足 EQ、+0 种子改变负零、中间 BF16 舍入改变结果。随机状态检查是抽象协议测试，设备证据限定于上节已记录的 TP2/TP4 条件。

全局准入不可省略。具体反例：P2 时一个 rank 从 epoch 1 开始等待，另一 rank 错把调用推进为 epoch 2；双方分别等待对方精确匹配的 1/2，均无法完成。更严重地，同一 world 若使用不同 capacity，会产生不同 scatter/gather 地址；ready 可能仍相等而 payload 落错位置。原 C ABI 没有跨 rank 参数共识能力，所以 driver 在 prepare 前显式共识 arm、rank 数、rows/capacity、调用顺序、初始 epoch 和运行长度，并在 prepare 后 rendezvous。window 只在单串行 stream/graph 上使用，禁止重叠 replay；销毁顺序为 graph reset→设备同步→全 rank rendezvous→free→finalize。每个 window 的总规约次数必须小于 uint32 wrap，门禁按每次 execute 两次规约校验保守上界；换窗口重新从零开始，不能在活动窗口上重置 epoch。
