# TPOT多样性门禁

同一份八输入、32请求、每请求1024个零、双钟1ms的CPU夹具，旧`passes_target`返回true，新实现返回false。[反例](counterexample.json)及[可重放入口](replay.py)保留原请求字节、摘要和旧实现；[36项检查](cpu-check.json)全部通过，未导入Torch，没有设备计时。

`bench_pd_decode.py`新增显式`--diversity-policy`，原始字节冻结到输出目录并记录来源与SHA。门禁逐项核对请求文件SHA、生产端/客户端IDs、长度与统计，再从1024个实际IDs重算token、bigram和最小周期；允许末尾截断的周期若完整重复至少两次，即拒绝。原八输入、32请求、三次预热及双钟p50/p95≤2ms条件保留。

policy字段为`min_distinct_tokens`和`min_distinct_bigrams`，均为大于1的整数，不给默认值；缺失时仍可测时，`passes_2ms=false`。反例采用2/2，只作测试；正式阈值需根据八个固定输入的串行结果在计时前冻结。整段周期检查是排除退化的必要条件，模型精度、尾部循环与在线会话正确性继续独立验收。
