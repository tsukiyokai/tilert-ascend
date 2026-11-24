# RSAG 所有权重排实验

本实验比较同一混合 CUBE/AIV probe 的两种通信实现：控制为 BF16 全互发后在每个 rank 执行现有 FP32 树；候选把输出按 512B 数据行分给 owner，先分发输入、在 owner 上执行相同树，再广播最终 BF16 结果。原 probe 的两次规约、中间 BF16→FP32 CUBE 投影和紧邻消费者均保留。这里没有串行和控制，也没有把控制命名为 HCCL。

可证伪主张是：保持源值每个输出位置一次 FP32 转换、`tree_plan.h` 的结合顺序和最后一次 BF16 舍入后，减少跨 rank 数据及重复求和，可能降低完整 probe 的时延。每次规约的 mixed joins 从 3 次增加到 5 次；流量下降不能推出时延下降，更不能推出 TPOT。实际编译、跨 die 可见性、数值与时延仍须 A3 门禁。

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

随后以 4 rank 复测。TP16 配对计时须独立设计，比较完整两次规约+CUBE 区间并记录各 rank raw wall/event；本入口不生成性能结论。门禁默认 tagged/random 有限 BF16 输入；CPU 的 signed-zero、溢出和中间舍入反例尚不等于设备边界证明。若要声称所有 BF16 位模式等价，设备还须检查极值、subnormal、NaN/Inf 的实际 Cast/Add/CUBE 行为。

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

`check.py` 真正编译生成两臂的 C++ header/TreePlan，执行 capacity=1/12/32、每个 rows≤capacity、ranks=2..16 的 675 组布局/分片检查，含非整除 owner 长度、512B 对齐、有效 tail 与 window 边界；3840 组 BF16 输入检查两臂及独立轮次归约的 FP32 结合顺序。它另运行 600 个每 rank 八 epoch 的随机异步状态调度。负例实际触发 ready 早于 payload、未 quiet 就复用 UB、旧 scatter/gather 活读者被同 parity 覆盖、live ready 被替换、消费者过早及 epoch 生命周期越界；另验证不一致 epoch 无法满足 EQ、+0 种子改变负零、中间 BF16 舍入改变结果。随机状态检查是抽象协议测试，不证明设备队列/缓存可见性；设备证据仍为空。

全局准入不可省略。具体反例：P2 时一个 rank 从 epoch 1 开始等待，另一 rank 错把调用推进为 epoch 2；双方分别等待对方精确匹配的 1/2，均无法完成。更严重地，同一 world 若使用不同 capacity，会产生不同 scatter/gather 地址；ready 可能仍相等而 payload 落错位置。原 C ABI 没有跨 rank 参数共识能力，所以 driver 在 prepare 前显式共识 arm、rank 数、rows/capacity、调用顺序、初始 epoch 和运行长度，并在 prepare 后 rendezvous。window 只在单串行 stream/graph 上使用，禁止重叠 replay；销毁顺序为 graph reset→设备同步→全 rank rendezvous→free→finalize。每个 window 的总规约次数必须小于 uint32 wrap，门禁按每次 execute 两次规约校验保守上界；换窗口重新从零开始，不能在活动窗口上重置 epoch。
