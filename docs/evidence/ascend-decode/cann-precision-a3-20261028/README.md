# CANN 首次真实采集失败

9382、TP16、CANN9.2，采集源码为 `d0078a4079bf85c19d4b25d2f801706299317c8f`。[实际读回](failure-readback.json)记录16份 rank 报告的 SHA：均在 `model.construct` 因 `rot` 不属于 `GlmMoeDsaForCausalLM` 而失败，serial 工件为0，尚未执行模型前向或严格 HF 比较。完整109文件的字节核验不能证明当前 loader 接受其参数清单。

容器随后处于退出137、`OOMKilled=false`，启动器退出码文件未写出；停止发起者未确认。保留远端完整日志和失败报告，不把这次运行记为完整采集、数值或时延通过。

## 插件准入修正

原 `VLLM_PLUGINS=ascend` 只允许平台插件，排除了通用模型插件 `ascend_model`。[两个独立进程的实际注册表](plugin-admission/registry.json)分别选中基础 GLM 类和 Ascend 子类；冻结 metadata 确有这两个 entry point。[实际加载方法的 CPU 对照](plugin-admission/actual-loader.json)显示，同一 `lm_head.weight` 与 `rot.weight` 输入由基础类报未知 `rot`，由既有 Ascend loader 正确加载 head 并过滤主模型不用的旋转项。原 MTP 类独立持有并使用 `self.rot`，checkpoint 不改动。

统一启动器在第一次 vLLM import 前设置 `ascend,ascend_model`。独立 precision/server 启动也须携带这份配置并使用全新进程；插件及模型解析均有进程内缓存。原始 CPU 日志压缩保存于本目录。

v3的[实际A3准入](v3-admission.json)完成新消费者109文件完整SHA、9382/16die和真正模型类解析；96/96分片加载完成，随后在首次前向之前因采集器将字符串传给`digest(Path)`接口而退出。[16rank错误和真实退出1](v3-runtime-error.json)保留。库哈希改为显式`Path(path)`；原错误由真实映射的CPU ELF复现，修复后同一映射检查及共12项CPU检查通过，见[日志](mapped-library-cpu.log.gz)。尚无模型前向或精度通过。
