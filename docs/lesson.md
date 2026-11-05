# The bitter lesson

定位性能问题时，先确认实际执行、计算结果和计时范围，再用对照区分原因。任务目标见 [plan.md](plan.md)，实验记录见 [probe.md](probe.md)。下文的硬件结论适用于所引证据中的 A3、编译器和运行时版本。

## 1. 确认请求实际执行了什么

主机已有的候选保留在主机，只下载设备新算出的预测；纯整数候选一次批量上传。逐个构造设备 scalar、再把同一候选上传下载，会把准备逻辑变成轮内开销。核对实际 Tensor 调用数与接受状态，再测同配置的设备收益，见 [token staging](evidence/ascend-decode/token-staging-20261028/README.md)。

补丁编译成功后，请求仍可能使用旧安装或旧缓存。为对照组和候选组保存权重、输入、前缀快照、Python 文件、宿主动态库、设备 ELF 的路径与哈希，以及编译器和生效配置，再从请求进程的加载记录核对。验证程序为计算哈希而打开文件，只证明它读过该文件；补丁是否生效要看运行时实际加载了什么。

修改框架前，从真实入口追到实例化的类、方法签名和分派分支。参数解析、导入和 `--help` 通过后，再通过安装器和真实模型对象完成一次执行。使用属性 setter 更新关联状态，保持构造函数、benchmark 与启动脚本的版本相容。

参考实现也要执行到实际 decoder。HF 将专家选择从 gate 移到 MoE 后，头部算术单测仍能通过，完整 forward 才暴露返回协议变化；保留 HF 原路由函数，并用变化输入核对全部层的实际选择，见 [78 层合成 CPU 执行](evidence/ascend-decode/hf-runtime-20261028/README.md)。

宽验证对账先要求完整参考行，再比较数值。空列表会让逐行循环以零检查通过；Python 的 `32.0 == 32`、`1 == True` 也不能替代身份类型检查。固定字段、行号与输入身份，解码后复制独立存储，见 [32 行参考帧正负例](evidence/ascend-decode/graph-frame32-20261028/README.md)。

本项目记录的静态算子缓存键漏掉了 vendor 源码身份，缓存又落在工作目录，曾使 A/B 两组运行同一实现。修改算子、编译选项或被捕获的调用时，分别使用独立工作目录，并核对实际 ELF；不同的 `VLLM_CACHE_ROOT` 仍会共用这层缓存。只改不影响图的宿主逻辑时，才复用经核实的暖缓存。见 [补丁重新测量](evidence/ascend-decode/kernelpatch-remeasured-20261019/README.md)。

环境激活脚本可能重设加载路径。激活后再设置 overlay，保留已有 `PYTHONPATH`，并从实际请求记录核对 OPP 顺序。打包、分派、缓存和 ABI 的具体规则见 [算子包装与加载](reference/ascend-custom-op-packaging.md)。

## 2. 构造能暴露故障的检查

按输入、中间值、状态和输出四层检查正确性。只看有限值、非零值或最终 token，会漏掉部分未写输出、接近零的异常尺度，以及被 argmax 余量掩盖的错误。独立参考从原始权重和量化规则计算，先校准规则，再确定误差界；与被测路径共用同一个 CANN 算子的结果只作交叉检查。

轮宽以本轮剩余容量为准。树验证缩至一行时只有 pending root，不调用零预算起草器；仍按同一状态机处理 EOS、提交与缓存保留。

可在主机判断的请求错误，在占槽和 TP 广播前拒绝。lookup 的缺失 prompt 若到 rank 注入后才报错，会使常驻 TP 命令流失效；单个坏请求的检查须继续验证下一个正常请求，见 [在线接入回归](evidence/ascend-decode/online-lookup-20261028/README.md)。

起草器的角色同时决定模型构造、cache 账本、允许选项和报告标签。删除未执行模型的加载后，仍核对其构造器对共享 head 与全局 RoPE 的影响，再认定实际模型数值等价。

树回执保留每个验证行的 parent，按父路径重建接受，不能把存储行序当成生成顺序。自然接受数与预算/EOS 裁剪后的提交数分别核对。

一条命令可执行多轮。命令计时与逐轮 span 分别核对；只有命令起止时间时，报告命令时长与实际轮数，不把它标为逐轮时延。

为每个检查选择会触发目标故障的输入：

| 故障 | 检查方法 |
| -- | -- |
| 旧状态、漏写、跨轮串扰 | 变化输入、输出哨兵、分配扰动、反复 reset |
| 归约顺序改变 | 各 rank 使用不同的抵消值与舍入边界值 |
| indexer 选择错误 | 输入长度超过 top-k |
| 检查漏报 | 注入漏写、旧状态、错 mask、错提交长度 |

输入变化周期要避开缓冲区单双面的重复周期。FP32 归约检查使用对加法顺序敏感的值；rank 编号组成的小整数和通常无法区分顺序。indexer 的短前缀若全部入选，就只验证了全选路径。每个负例都须被测试拒绝。

定位全模型误差时，固定 teacher 输入，保存第一次分歧处的 hidden、logits、索引、专家和有效 KV。自由生成在第一次分歧后便走入不同上下文，适合检查最终行为；固定轨迹适合比较同一次计算。分别验证串行与宽验证、重复请求、短请求接长请求，以及拒绝后的回滚。局部层检查之后继续检查完整模型，少 rank 检查之后继续检查目标 TP16 规约顺序。参见 [独立参考与教师检查](evidence/ascend-decode/shared-fia-model-20261020/README.md)、[宽度不变性负例](evidence/width-invariance-20261023/round-invariance.json)、[两 rank 组合验证](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-chain-wide-v2-a3-194/summary.json)。

局部诊断要有退出条件。[Attention扩层](evidence/ascend-decode/attention-rounding-a3-20261029/README.md)中，h1误差下降约45%，h3却只改善约14%；第一层逐位闭合不能替代传播检查。已在原容差内的比特差异，先检查是否导致后续越界、索引/路由分歧或宽度不一致，再决定是否继续细分。数学图及阈值保持固定，原CANN精确比较与独立HF容差比较分别报告。

速度准入从请求工件重算输出及统计，并在测时前冻结quality policy原始字节与SHA。[全零1ms反例](evidence/ascend-decode/diversity-gate-20261029/README.md)表明，八输入覆盖、完整token数量和双钟达标仍能容纳退化输出。整段周期和token/bigram计数补充准入，独立数值与会话状态继续分别验证。

## 3. 保留计算中的舍入和编码规则

融合去掉写回时，仍须保留该位置的中间 BF16 舍入。Norm 后量化、路由概率转换、专家乘权与求和，以及 TP 归约的数据类型和结合顺序，共同决定基线计算。项目中的 Norm/Quant 融合改变了 INT8 编码，继而改变完整模型输出，见 [融合舍入实验](evidence/ascend-decode/shared-fia-model-20261020/norm-quant-fusion/README.md)。等价替换沿用原舍入规则；采用更高精度时，另立计算与质量对照。

接入 W4A8 kernel 时，逐项核对有符号解释、打包轴、半字节顺序、物理布局、步幅、scale、offset 和 bias 加入位置。按所用 API 的字节或元素单位计算拷贝长度，再读回原始字节及相邻区域。`int4b_t` 的 `sizeof` 与有效位数不同，张量形状也没有完整描述物理布局。NZ 张量上的 view、copy 和 gather 按各自验证的布局与语义使用；替换其中一种操作时重新核对结果。参见 [W4A8 MSD 与 INT4 路径](reference/w4a8-msd-int4-mmad.md)、[checkpoint 编码与分片契约](evidence/ascend-decode/weight-contract.md)。

每次 A/B 调用前恢复算子可能改写的输入，使两组从相同状态开始。用完整 KV 的 dense PV 替代稀疏读取时，还要保证被 mask 的行是有限值：零乘 NaN 仍会产生 NaN。

CPU 参考要复现硬件实际使用的 scale。所记录的 A3 Fixpipe 路径在反量化前清除 FP32 scale 的低 13 位；CPU 参考使用完整 FP32 scale 时便算出了另一结果。按这条规则修正后，对应的固定输入检查通过。该对照只解释这组差异，其他模型漂移仍需独立定位。见 [参考修正及原始对照](evidence/ascend-decode/gmm2-compact-20261020/README.md)。

保留 head 实际的输出舍入边界。BF16 输出再转 FP32 无法恢复未舍入积；同时绑定执行模式、有效算术与实际导入源，见 [FP32 head 参考](evidence/ascend-decode/fp32-head-reference-20261028/README.md)。实际图须直接对照 HF，或经覆盖相同边界的位等桥连接；两个近似对照各自通过，不保证传递后仍满足同一误差界。

FP32 规约树决定加法的结合顺序，属于所选计算规则。记录并行布局、算法及 `HCCL_OP_EXPANSION_MODE`、`HCCL_DETERMINISTIC`；改变规约树或任一配置后，重新检查数值与接受率。重复运行是否一致、结果是否符合参考，分别验收。

## 4. 先校准计时，再比较性能

随数字保存完整测量条件：模型、精度、物理设备与 rank 数、层数、输入/输出长度、并发、验证宽度、草拟器、预热、样本数、同步位置和统计量。注明执行方式（eager 或 graph）、测量对象（热缓存微基准或真实请求），以及 profiler 是否启用。单 die 的 TP16 分片只能给出该分片的计算成本；重复一个权重 bank 测得的缓存条件，也要与完整模型的权重访问分开报告。

预计算候选同时比较诊断实现与真实执行入口。移走重算会改变驻留存储、访存和调用数量，收益由共同输出边界的完整墙钟确定。[prefix Value三臂对照](evidence/ascend-decode/prefix-value-timing-a3-20261029/README.md)未达到相对展开的预测，且全部慢于原生SFA，因此停止性能接入；数值干预结果单独保留。

先测计时程序的空载开销，再把宿主下发时间与 event 时间放在一起检查。宿主来不及继续下发时，设备 event 区间会包含等待宿主的时间。测 HBM 时轮换超过相关缓存容量的数据，并检查实际写回，排除空 kernel、被编译器删除的工作和常量输出等特殊路径。用 profile 定位耗时，再以同配置、无 profiler 的请求确认端到端变化。

A/B 每次改变一个因素，用同配置重复请求测出运行间波动；效应接近波动时采用交替配对，报告分布、差值和失败情况。测量前确定关注最快可达成本、典型延迟还是尾延迟，始终使用同一统计量。报告性能的环境也要通过正确性检查。

起草器成本对照先固定最长/最短尾部及最近匹配规则，再逐轮比较完整提议。哈希碰撞后的单项检查可能遮蔽更长真实命中，须用独立扫描核查精确策略。分别报告初始化、稳态与高尾成本；纯复制的收益继续接受编辑、歧义短尾和无匹配输入检验，见 [lookup 精确索引与 CPU 对照](evidence/ascend-decode/lookup-radix-20261028/README.md)。

逐轮计时也会改变被测堆。将 GC 起点、callback 和报告保留方式分别纳入对照，先定位回收落在哪个调用，再验收索引表示变化的总成本。[全局 arena 对照](evidence/ascend-decode/lookup-arena-20261028/README.md)减少了节点对象，却使复制与轻度编辑退化；[仅将局部字典的边改为上下文坐标](evidence/ascend-decode/lookup-coordinate-20261028/README.md)才在四类输入获得 p50 收益，编辑负载仍有尾部退化。容器、标签和访问方式分别验收。

原生扩展须核对实际类型声明和热路径。完整包名的构建目录若未匹配 `.pxd`，仍可生成保留 Python 属性分派的 `.so`。检查生成 C、实际字段类型与二进制，再用同一已检查产物测量和安装；见 [lookup 原生执行](evidence/ascend-decode/lookup-native-20261028/README.md)。

局部关闭循环 GC 前，证明节点引用只能指向后代，核对整数键归一化，并保留起草器的自环回收。声明文件中的装饰器可能未生效，须直接检查实际对象标志；分配头、析构和子字典跟踪也会改变，不能把全部差值归为回收暂停。见[节点 GC 对照及完整长尾](evidence/ascend-decode/lookup-gc-20261028/README.md)。

报告 TPOT 时同时保存轮时间、实际接受数和完整输出。先写明计时起止点，再确定分母：若取首个与最后一个输出 token 的回调时间差，且输出数 N > 1，分母就是 N−1 个输出间隔；其他计时口径按原始记录定义分母。说明首 token、bonus token 和最后一轮裁剪如何计数。接受数从逐轮记录读取，验证宽度只表示候选容量。短、长请求分别测量，因为它们的接受分布可能不同。参见 [完整机器重测](evidence/ascend-decode/whole-machine-20261023/README.md)、[宽度梯子](evidence/ascend-decode/width-ladder-20261023/README.md)。

## 5. 用对照定位耗时来源

先核对各部分与整体的计时范围，再解释剩余时间。按 correlation ID、任务 ID 和实际调用链关联事件；阻塞的 `tolist()` 时间窗口会覆盖并发任务，时间包含关系无法识别这些任务的发起者。图内任务则可能只有图级宿主调用，需要沿图内关联继续查。

读取计数器定义后再分配耗时。设备忙碌时间取执行区间的并集；并发管道以及 AIC/AIV 的计数可能覆盖同一段墙钟时间。分别解释活跃、发射、等待与流量，避免把它们相加成一个占比。

时间线上的空洞给出候选原因，受控移除、延迟、替换或冻结输入才检验这个原因。每次改动后都查看总时间：删除前置遍历可能把冷数据读取转移给下一个消费者。消去融合 kernel 的一个阶段时，差值表示它对整条关键路径的影响；若阶段之间有重叠，另行测量才能得到独立阶段耗时。参见 [实际轮时间与宿主等待](evidence/ascend-decode/round-is-device-20261023/README.md)、[PMU 匹配、单位与归属审计](evidence/ascend-decode/layer-major-20261018/mtp-chain-pmu-summary.json)。

用预算做量级检查：预期节省应落在优化所作用的成本范围内。给拟合截距命名为 launch 开销，需要另外测量 launch；把总残差除以 kernel 数只是换算，仍待独立证据解释。推导物理下界时列出必需的数据、算术、带宽和依赖条件，并注明假设。当前实现的管道时间用于描述现状，物理下界用于判断还有多少空间。

## 6. 按数据依赖选择融合范围

先选择能完整负责生产、发布、消费和存储生命周期的片段，写清数据何时可用、谁持有缓冲区、何时能复用，再改变执行方式。片段通过完整路径对照并取得收益后，才扩大到更多算子、整层或整步；每次扩大都重新检查数值、共用缓冲区、同步和端到端时间。

保留真实的数据依赖。Norm 的输出若先供 router 使用，就须在 router 前完成；移进专家执行会改变这条依赖。共享激活的投影还要核对精度与发生阶段。静态权重可以提前准备，动态专家与 KV 地址要等路由或 indexer 产生。

沿真实调用链检查基线已有的跳空组、融合、图捕获和异步执行，再用相同 shape、分片和精度测量候选。TP 切分每个专家的参数，EP 分配专家，两者的活跃专家数、字节和通信不同。多个优化的收益由组合后的测量确定。

融合和流水化的节省须覆盖新增的扫描、重读、量化、规约与通信。使用 DSL、SuperKernel 或 persistent 容器时，从生成代码与实际轨迹确认同步粒度、数据重用或预取距离如何改变，再解释整路径收益。两 rank 的 whole-layer 对照同时改变了融合与通信，只支持组合效果，见 [配对记录及范围](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-timing-m12-v1-a3-194/README.md)。

## 7. 分别检查次序、可见性和活性

A3 MIX 中 AIC 与 AIV 的 block 编号不同。先用探针记录 `GetBlockIdx`、`GetBlockNum` 和 `GetSubBlockIdx`，建立生产者与消费者的对应关系，再添加同步。通知次数与等待次数必须匹配，参与者也必须能获得物理核；常驻同步要避免已占满核的参与者等待尚未获调度的工作。

交接协议须标明写入 pipeline、信号方与等待方，并落实到 pipe 发射顺序、DMA 完成、标量 DCache 处理和跨核会合。在 `TPipe::Reset` 之外仍须完成所需的数据发布与会合；DMA 完成用相应完成条件确认，mixed barrier 后仍须处理消费者所需的标量缓存。跨 pipe 的 `SetFlag/WaitFlag` 由拥有该 pipeline 的核执行。会发生整行写回时，让独立进度标记分占 cache line，避免覆盖其他核的标记。

让同步承担真实数据依赖，再分别检查活性和次序。独立进程的超时与退出状态用于识别挂起，读取字节的比较用于检查数据次序；覆盖变化输入、重复 epoch、缓冲区轮换和边界 shape。移除目标同步后，继续排查其他 barrier 是否掩盖了依赖。`handshake` 的历史更正保留了成功/超时观测，撤回了 mode 编号的解释；后者仍需验证，不能作为设计前提。参见 [MIX 实验](evidence/ascend-decode/mixbrick-20261020/README.md)、[握手实验及更正](evidence/ascend-decode/handshake-20261020/README.md)、[同步原语与资源约束](reference/ascend-launch-overhead-and-sync.md)。

SHMEM 按对端指针映射、设备读写、数值规约、重复通信和完整模型逐级验收。先绑定设备，再加载依赖设备上下文的库；为对称窗口保留所属 stream 和 epoch。关闭时依次停止提交、同步设备、销毁引用窗口的图、让参与 rank 会合、按约定释放窗口，最后 finalize。扩大测试规模后，重新核对规约树、rank 几何和进展条件。具体契约见 [两 rank 层链](../tests/glm52/whole_layer/vendor_hook_tp_chain/README.md)与 [whole-layer 设备接口](evidence/ascend-decode/megakernel-device-20261021/integration/whole-layer-mechanisms/README.md)。

## 8. 维护图、缓冲区与请求的生命周期

ACL graph capture 记录任务；要得到一次完成的 forward，须显式重放并同步，再读取输出。将 token、position、slot、长度、RoPE 和共享索引绑定到捕获时的持续存储，并在对应消费者读取前写入本次的值。记录地址、shape、stride、dtype 和设备，并核对内容已经更新。跨层 selection mask 按生产者的生成代次更新，tensor 地址仅标识存储位置。

图会复用输出和 workspace。消费者在下一次复用前完成读取或复制，链式 MTP 同时保留独立的 seed 与 recycle parent。预热后用毒化值填充未写 KV，确认 capture 前后相关状态不变、重放实际执行、reset 清除有效状态。分别记录图池和模型权重的峰值内存，并计入各宽度图持有的资源。先释放图，再释放它引用的缓冲区或通信窗口。参见 [ACL graph 检查](evidence/ascend-decode/cann-graph-20261017/README.md)、[MTP 链别名与回滚](evidence/ascend-decode/layer-major-20261018/mtp-chain-graph-audit.json)。

编译图的观测按字段写入本轮代次，实际删除 stamp 写回后再执行，确认读回会拒绝；节点计数不能代替这个负例。输出 token ID 也须单独比较，logits 通过不会检查已打包的 ID。CPU AOT 的[持久观测检查](evidence/ascend-decode/graph-producer-20261028/README.md)只证明该机制，NPU/SK 的保留、流次序及正式图数值另验收。

验证 P/D 完整链路时，用同一 request ID 关联各 rank 的 WRITE acknowledgement、prefill 首 token、真实 decode 步、回调 token 与 HTTP 响应；前缀传输完成只是其中一环。接收缓冲区要保留到消费者结束。单机分时验证若在 P 传输后释放模型再加载 D，应保留接收状态，并把模型加载计入 TTFT。按所用 HIXL 版本核对设备放置、监听端口和 allocator-block 注册规则。参见 [真实 handoff](evidence/ascend-decode/pd-live-20261017/README.md)与 [HIXL 设备记录](evidence/ascend-decode/pd-hixl-20261017/README.md)。

## 9. 沿真实提交边界评估草拟器

分别记录 pending token（等待下一步消费的 token）、已提交序列，以及目标 KV 和 MTP KV 的有效长度。Lookup 只更新实际执行的目标状态；切回 MTP 时，先补齐缺失的 conditioned 计算，再推进有效长度。为草拟、验证、接受、回滚和请求结束定义各自的状态转移，最后一轮按剩余输出预算裁剪 bonus。

离线回放沿每个策略自己的接受边界前进。拒绝点后的预测使用了被拒上下文，新分支需要重新计算；copy oracle 仅用于给出上界。suffix 索引只收录已有后继的位置，防止当前尾部遮住历史匹配。修改验证宽度或树预算时，一起更新因果 mask 与有效 KV 长度，并核对 tree attention 的实际调用；提出树候选和 resolve 接受路径只覆盖其中两步。参见 [Lookup 状态契约](evidence/ascend-decode/lookup-contract-20261019/README.md)、[真实轨迹与复制假设](evidence/ascend-decode/lookup-drafting-20261019/README.md)。

先检查真实目标输出的质量和一致性，再比较草拟器的接受率。合成单调复制语料、截断模型、错误 attention 或固定接受长度可能放大速度；模型退化为重复 token 时，复制草拟器尤其容易得到虚假的高接受率。不同工作负载得到相反结果时同时保留两者。加宽候选可能改变活跃专家并集和验证成本，因此逐项计算实际字节与耗时。

## 10. 把机器和环境纳入对照

从运行进程检查 Python、Torch、torch_npu、vLLM、vLLM-Ascend、CANN、NNAL/ATB、编译扩展与 C++ ABI。除了记录版本，还要实际导入模块，检查平台 entry point、`tbe`、SFA 符号、必要 OPP 操作和 `vllm_ascend_C` 的 ABI。Triton、`fla_npu` 或插件冲突按发生问题的环境定位；仅有历史冲突记录时，先复查依赖再决定是否调整安装。

按成本递增验证启动路径：先做不加载模型的导入与参数检查，再跑经过真实 decode 入口的短请求，最后做长测量。定位故障时使用最小仍可复现的配置；CPU 状态检查、单 die 算子、少 rank 协议与整机请求各自回答一类问题。前一级通过后再投入下一项昂贵实验，失败则先读出该级的具体原因。

提交前核对 HBM、进程、die 映射和占用者，确认所需设备仍可用。低利用率和一次盘点都只是当时的读数，清理操作只针对已确认属于本任务的进程或容器，保留其他租户的任务。容器消失或停止时，先查 `docker events`/`inspect`，再区分外部停止、OOM 或其他退出原因。

启动失败还可能来自 NFS `D` 状态进程、残留 NPU 上下文或被占用的 rendezvous 端口。先停止已确认属于本任务的故障进程，让新实验使用独立缓存目录，再核对进程、端口与设备资源；资源仍未释放时，按需重建本次实验容器。故障发生在前一进程时，下一次启动可能只显示内存或通信症状。

把模型加载与 decode 分开计时。先测实际挂载在不同读取方式和并发下的表现，再决定近端暂存、分片读取或并行化；多个 rank 重读慢挂载的同一页面可能互相等待。暂存前检查容量、校验和、可用内存与保留期限。设备等待设定期限，期间继续完成证据核对和可运行的小规模检查。历史条件见 [整机记录](evidence/ascend-decode/whole-machine-20261023/README.md)。

驱动脚本复制到独立文件并校验后启动，运行结束前保持文件内容不变。任务需要独立于 SSH 会话存活时，使用受控的后台启动，保存 PID、日志和退出状态。停止任务按确认的 PID 或容器操作，避免 `pkill -f` 匹配自己的 shell 或其他任务。断连、超时或日志停更后，直接检查进程状态再决定是否重启。

## 工具入口

`VLLM_PLUGINS` 同时筛选平台与通用插件。只允许 `ascend` 会漏掉 `ascend_model`，使主模型使用基础 GLM loader，把 MTP 专用 `rot.weight` 当成未知主干参数。先核对实际 entry point metadata，再在全新进程里检查 `get_model_architecture()` 返回类和真实 `load_weights` 源码；导入成功或只查看注册表不足以证明已解析类的缓存同步更新。[实际对照](evidence/ascend-decode/cann-precision-a3-20261028/README.md)

| 任务 | 入口 | 先核对 |
| -- | -- | -- |
| 整机 decode | [统一启动](../tests/glm52/launch_arm_unified.sh) | 配套源码、模型、设备、TAG |
| 旧环境运行 | [旧启动](../tests/glm52/launch_arm.sh) | 激活脚本、固定路径 |
| 参数检查 | [测量程序](../tests/glm52/bench_cann_mtp.py) | 展开参数、请求记录 |
| 轮计时 | [round_clock.py](../tests/glm52/round_clock.py) | 记录完整、排除诊断请求 |
| 配对统计 | [paired.py](../tests/glm52/paired.py) | 目录约定、实际接受数 |
| 设备盘点 | [逐卡读数](../scripts/free_dies.sh)、[阈值检查](../tests/glm52/precheck.sh) | 进程、die 映射、占用者 |
| A3 MIX 构建 | [构建脚本](../tests/glm52/mixbrick/build.sh) | AArch64、CANN 9.2.0 |
| kernel 正确性 | [INT4 对照](../tests/glm52/routed_int4/check.py) | vendor、shape、构建变体 |

统一启动须设置 `TAG`、`WIDTH`、`TREE`、`OUT`、`MODEL`、`CAPTURE`、`RUNTIME`，核对 vLLM 与 vLLM-Ascend 的配套源码、镜像挂载和独立端口。脚本会删除同 TAG 的输出与 VLLM 缓存，每次用新 TAG 保存证据。它没有设备占用检查，提交前另行盘点；盘点脚本也不取得独占租约。

旧启动依赖 `/workspace/environment/activate.sh` 和固定模型、快照路径，迁移镜像前须核对。设备检查拒绝时退出 2；脚本会重建同 TAG 的输出目录。

读取测量程序的真实参数时，以启动脚本展开结果和请求记录为准，再核对 argparse 默认值。短请求验证入口，长期正确性与性能另测。

`round_clock.py` 从指定目录的 `rank-0.json` 读取 `advance_mtp_n`，只自动排除 warmup，调用者还须排除 profiled 请求。

`paired.py` 使用 `name-1/2/3` 目录，统计请求 p10 的最小值；每 token 换算固定为历史接受数 `227/1023`，新实验须按实际接受数换算。

MIX 构建保留全部编译与链接日志，核对二进制身份和警告。kernel 接入整机前，完成真实 vendor 对照，按已验证的 namespace 与编译 flags 组合运行。

## 调试入口

A3 MIX 的构建入口是现有脚本与 [算子包装说明](reference/ascend-custom-op-packaging.md)。独立程序先加载 `/opt/ascend/cann-9.2.0/set_env.sh`，设备编译使用 `bisheng -xcce --npu-arch=dav-2201`。本项目的同源对照发现，`--cce-aicore-arch=dav-c220` 虽然编译成功，却丢掉了所需 `__schedmode__` 的效果。构建后从运行日志核对 `schemMode`、`taskRation`、`coreDim` 和 `mixType`。OPP 工程生成入口与原始 `<<<>>>` 入口各自使用对应的属性规则，分别核对 `__mix__`、`core_ratio` 与 task-type 声明。

选择 Matmul 调用方式时先确认由哪个核驱动。普通 MIX Matmul 的 KFC 调用由 AIV 驱动；在 AIC 内直接使用 cube-only Matmul 时，须在包含 `lib/matmul_intf.h` 前定义 `ASCENDC_CUBE_ONLY`。宿主 tiling 使用 `GetTiling(AscendC::tiling::TCubeTiling&)` 重载，include 和 `tiling_api/platform/ascendcl/runtime` 等链接依赖由构建脚本维护。定义数据类型的编译宏须同时传给宿主与设备编译单元。

原始 MIX 跨核同步先绑定设备，再取得 `rtGetC2cCtrlAddr`，用 `AscendC::SetSyncBaseAddr` 安装同步地址。随后按 [同步参考](reference/ascend-launch-overhead-and-sync.md)核对 flag 资源、核数与通知/等待的配对规则。

| 目的 | 工具 |
| -- | -- |
| 找到最早失败 | runtime 日志、显式同步 |
| 检查内存与竞争 | `mssanitizer` |
| 断点调试 | `msdebug` |
| 查看执行与代码 | `msprof`、`msobjdump` |
| 读取中间值 | `AscendC::printf`、`DumpTensor` |
| 收集异常 | `ASCEND_LAUNCH_BLOCKING`、`NPU_COLLECT_PATH` |
| 模拟与对照 | `simulator`、`operator_cmp` |
| 定位挂起阶段 | stage bitmask |

runtime 日志在命令前设置 `ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1`。先读最早的原始错误，再追 Python 堆栈；异步 AI core 错误可能在下一算子才报告。探针在 `try` 内显式同步，让本次检查捕获本次调用的失败。

`mssanitizer` 的 `memcheck`、`racecheck`、`initcheck` 和 `synccheck` 需要 `--cce-enable-sanitizer`，mixbrick 的 `SANITIZE=1` 会加入该选项。`synccheck` 在 kernel 退出时报告，挂起时按 stage bitmask 分别开启计算、set、wait，用独立进程、变化数据和超时定位。分别读取生产者中间值与消费者结果，区分写错位置和同步错误。性能使用无插桩版本另测。

`msdebug` 用 `DEBUG=1` 加入设备 `-g` 信息。历史版本曾在按 kernel 名设断点后崩溃；遇到同类现象，先核对调试器版本以及断点、attach、源码位置的支持，再判断 kernel。

设备输出用 `show_kernel_debug_data` 读回，并记录调试写回对执行条件的影响。`tikicpulib` 与 `ASCENDC_CPU_DEBUG` 用于 CPU 检查；设备同步与时延在真机检查。模拟器和算子对照按安装版本的支持范围运行，模拟结果与真机记录分别保存。

`msprof` 的时间线用于定位，`msobjdump` 的反汇编用于核对二进制结构，运行行为用真机记录验证。`ASCEND_LAUNCH_BLOCKING` 辅助同步定位，`NPU_COLLECT_PATH` 保存异常 dump；按匹配 SDK 配置，诊断结束后关闭，再验证正式执行。profile 与无 profiler 的正式计时分别保存。

编译输出直接保存到日志文件，失败时连同警告一起检查；直接接到 `head` 会因 SIGPIPE 中断编译器。框架只报告准备步骤失败时，用它实际传入的参数单独执行该步骤，保存退出码。SHMEM 初始化失败时，打开 `SHMEM_LOG_LEVEL` 并读取 SDK 日志，定位具体失败调用和参数，再选择修复；`-4` 或 `507015` 本身只标识失败，可能对应多个原因。

量化预测与实测变化分别记录。[合包输入](evidence/ascend-decode/host-inputs-a3-20261029/README.md)的20%双钟预测失败，墙钟18.082%的下降仍按原始数据保留；不删除共同CPU成本以改判通过。真实模型接线另做数值对照，结果不升级为TPOT。

核对 AscendC 算子二进制时，同时用 `readelf -sW` 与 `llvm-objdump -d` 检查 `mix_aic`/`mix_aiv` 符号体积、隐式类型截断与链接合并行为（见 [SFA 快速迭代与编译器发现](evidence/ascend-decode/sfa-fast-iter-20261031/)）：
1. 警惕 `using namespace matmul;` 引入的窄类型工具函数：CANN `batch_matmul_utils.h` 中的 `matmul::CeilDiv(uint16_t, uint16_t)` 与 `CeilAlign(uint16_t, uint16_t)` 会静默截断 `int64_t` 参数并产生额外掩码指令与寄存器溢出，算子公共头应提供保留宽度的泛型模板。
2. 控制 MIX 算子单一入口的内联膨胀与 I-Cache 边界：多个分派分支分别内联同一重型流水线（如 `PreloadPipeline` 或 `MergeKv`）会使 `mix_aiv` 突破 64 KB L1 指令缓存并劣化寄存器分配；应合并主循环入口以保证核心流水线仅内联一次。
3. 禁止在双通道 MIX 编译（`dav-c220-cube` 与 `dav-c220-vec` 经 `ld.lld -r` 合并）中对未以 `g_coreType` 作为模板参数的类成员函数使用 `__attribute__((noinline))`：两编译通道会生成同名 `WEAK` 符号并由链接器丢弃其一，导致异构核执行错误指令集触发 `507015` 异常；即便按 `g_coreType` 模板消歧，跨 `Init()` 与主循环切断内联也会阻碍成员常量标量提升。
4. 环状任务缓存（`RunInfo extraInfo[3]`）在跨阶段复用时，所有条件覆写字段须具备完备的 `else` 复位分支；闲置核协同卸载前须先从 `InitCalcParamsEach()` 核对 `gS1Start..gS1End` 的真实连续分核映射，严格区分 3 槽环状缓冲中 `(loop + 2) % 3 == (loop - 1) % 3`（1 拍延迟，如 `Vec1L + ComputeMm2`）与 `(loop - 2) % 3`（2 拍延迟，如 `Vec2`）的真实消费拍点，并将 `SyncAll<true>()` 精确锚定于消费者读取协同槽位的那一拍与槽位生命周期释放点。
