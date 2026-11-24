# Layer68索引顺序三臂对照

真实A3 9382、TP16、同一冻结8K输入，常驻模型依次执行原生、复制indices、按位置排序indices三臂；只改layer68的实际SFA消费者。每臂M1×14、M32及M32精确reset。[独立读回](readback.json.gz)重算144份PT的SHA、16rank报告、状态与退出值；[摘要](summary.json)保留每rank边界结果。

| row13 M1/M32 | 原生 | 复制 | 排序 |
| -- | -- | -- | -- |
| h69逐位相同rank数 | 0/16 | 0/16 | 16/16 |
| 最终logits逐位相同rank数 | 0/16 | 0/16 | 16/16 |
| 最终logits NRMSE | 0.0327916 | 0.0327916 | 0 |

原生及复制臂的norm68两输入/输出、QL/QPE、有效KV和WUV在16rank均相同；2048个选择位置集合相同，顺序中相邻两项5196/2079交换。最早浮点差异只出现于rank12的SFA输出，2048元素中1个BF16元素相差0.00390625，随后局部attention partial与全局TP输出不同。rank0的首差在TP输出，不能据此归因于通信。排序使本次已记录浮点边界及row13最终输出逐位一致；排序后的M1数值会改变，仍须完整HF、多行及多输入验收。

[契约](contract.json)在启动前冻结。原生首行、row13和M32输出逐位复现旧控制；三臂reset及全局状态通过。[加载收据](kernels.json)解析16份strace中85个kernel文件，未解析路径为0；该记录证明成功open，不证明执行或性能。首次CPU reader误读退出收据字段而退出1；仅修正字段后，用同批PT完成v2独立读回，未改变设备实验或容差。

[共享归档读回](archive-readback.json)核对全部144个PT，归档664,553,745B，SHA256为`c6ea2bc133e09e83020bad65863f155947bbb7b66fcba8172ab47265b4a4ce59`。[冻结收据](receipts.tar.gz)包含实际源码、命令、契约与两版reader；[清理](cleanup.json)仅移除本方已退出容器，原始bind文件保留。本次为数值实验，无TPOT计时。
