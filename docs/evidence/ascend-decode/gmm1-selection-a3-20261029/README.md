# GMM1 overlay 被插件初始化遮蔽

冻结插件的`bootstrap_custom_op_env()`会在原vendor路径未出现在`ASCEND_CUSTOM_OPP_PATH`时，把它加到首位。实际启动器只放overlay，故初始化后原vendor优先；源码中已有SwiGLU屏障，不代表执行了修补内核。[实际进程与算子读回](selection-readback.json.gz)绑定初始化后的路径顺序、成功`openat`、ELF/JSON的实际SHA、输入与405项权重切片摘要。

同一9392、物理die15、CANN9.2、相同镜像与真实layer3/rank1的M32分发输入，分别在新进程及隔离缓存重放32次。控制臂实际读取原始ELF `2444c7e7…`；显式同时列出overlay与原vendor后，插件保持顺序，修补臂实际读取overlay ELF `2eb81546…`。修补源码仍对应仓库既有[屏障与构建路径](../swiglu-barrier.md)，未作二进制编辑。

控制臂0/32次完整匹配独立CPU，31次重复中仅16次逐位相同，8192行中6642行未达到非零A8量化应有的最大绝对码127；其中17次输出逐位复现现场wide帧。修补臂32次的全部INT8码均与CPU相同，31次重复全部逐位相同，8192行均达到127。FP32 scale每次仍有32/256行与CPU不同，最大绝对差`3.637978807091713e-12`、最大3 ULP，严格入口仍退出1；没有放宽判据或宣告组件精度全过。

[重放入口](../../../../experiments/cann-precision/replay_gmm1.py)仅把旧M1的独立算术推广到实际expert counts，保留INT4切片、MSD校正、FP32 SwiGLU及FP16→INT8舍入。使用源码提交`64180c6418bcc29771fe741f71f74902a1b67876`。`captured_vs_cpu`描述现场，`reference/repeat`描述新重放，两者不混作一个通过条件。未测时延、整模型HF或TPOT。

原始输入、三组输出、报告、实际openat日志、脚本和源码归档于共享NFS `/trt-a3-evidence-20261028/gmm1-selection-source64180c6-20261029.tar.zst`，77148247B，SHA256 `28da0773d108feb5cff1131f1b4b67fe8a048890096e4f44a955a80ab2c9210f`；完整解码、成员清单与zstd完整性检查退出0。长驻容器在算子重放完成后被外部停止，退出137且OOM=false；原控制臂的实际ELF另由同机空闲die上的一次性作业核对。
