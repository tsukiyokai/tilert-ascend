# Carver 测试的 collection 回归

[PR #1955](https://github.com/tile-ai/tilelang-ascend/pull/1955)的`2d3249b1`新增测试在模块顶层导入Carver。既有`BaseTemplate`字段使用`field(default=auto_infer_current_arch())`，类定义时即探测设备；pytest-forked父进程因此初始化NPU。[对应失败job](https://github.com/tile-ai/tilelang-ascend/actions/runs/40706503123/job/122880076120)的827项测试均报bad-fork异常，新增六项自身通过；另两项设备初始化超时单独保留。

`233047cb`只把该测试文件的Carver导入移到fixture/test执行期。根线程重新执行[CPU检查](cpu_check.py)，[原始输出](root-cpu-run.log.gz)确认旧版collection触发Carver并退出2，新版不触发、退出0且仍收集六项；前后断言、参数与fixture表达式AST相同。实际类定义与架构函数源码在CPU替身下触发`get_arch('ascend')`；[Ruff检查](root-ruff-run.log.gz)通过，生产资源配置代码未变。

[远端快照](remote-status.json)确认当前head的AscendC-PTO与TileLang-Ascend CI均通过。[原生NPU job日志](native-ci.log.gz)记录新增六项全部通过，主集1352项通过、3项跳过，示例153项通过；PR描述已更新并读回核对。[先前9392 JIT](../carver-a3-20261028/README.md)保留SDK重构前的版本范围。输入、脚本和日志的字节SHA见[manifest](manifest.json)；整模型与TPOT继续独立验收。
