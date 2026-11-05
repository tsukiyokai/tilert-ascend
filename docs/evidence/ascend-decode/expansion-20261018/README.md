# AIV reduction wins under the composed set, reversing an earlier result

`a3-84`, machine idle, TP16, source `04a6628`, the frozen 8K prefix,
`--max-tokens 1024 --capacity 9344 --warmup 3 --requests 3`, profiler off.
All three cases run the composed flag set `--target-graph --draft-graph
--draft-projection-tp --compile-target --super-kernel --attention-kernel
gather_fia --fp32-head` with `TORCHINDUCTOR_AUTOGRAD_CACHE=0` and
`VLLM_DISABLE_COMPILE_CACHE=1`. Only `HCCL_OP_EXPANSION_MODE` differs.

| case | mode recorded | TPOT median (ms) | three runs | rounds | A | round (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| unset | `None` | 16.489 | 16.346 / 16.489 / 16.630 | 298 | 3.433 | 56.61 |
| explicit | `AI_CPU` | 16.441 | 16.399 / 16.441 / 16.456 | 298 | 3.433 | 56.44 |
| explicit | `AIV` | 15.978 | 15.927 / 15.978 / 16.051 | 298 | 3.433 | 54.85 |

All three deliver 298 rounds at A = 3.433, so the trajectory is identical and
this is a controlled comparison of the mode alone.
[unset](default.json), [AI_CPU](ai-cpu.json), [AIV](aiv.json).

AIV is 0.463 ms faster than AI_CPU, 2.8%, and 1.59 ms faster per round. Its
spread is also the tightest of the three at 0.124 ms against the unset case's
0.284 ms.

## This reverses the recorded decision

[The earlier control](../cann-graph-20261017/README.md#ai_cpu-versus-aiv-control)
measured AIV at 21.473 ms against AI_CPU's 20.441 ms, 5.05% worse, with
identical trajectories, and the project kept AI_CPU. That run used the
default configuration: eager drafts, `SparseFlashAttention`, a BF16 head and
no compiled target. Under the composed set the sign flips.

The mechanism is a hypothesis, not a measurement. `aiv-comparison.json`
records the AI_CPU reduction as `allreduceAicpuKernel` with `block_num` 0,
so it occupies no AI core and overlaps compute freely, while the AIV
reduction is an `AivKernel` with `block_num` 16, a third of the die's 48
vector cores. Under the old configuration that contention cost more than the
kernel time it saved: `hccl_kernel_union_ms` fell 38.920 → 7.064 but
`task_free_gap_ms` rose 0.257 → 1.823 and the round grew. The composed set
changes what the vector cores are doing — the compiled target with
SuperKernel consolidates vector work into fewer, larger tasks, and the
graphed draft removes the host launches whose jitter AI_CPU was absorbing —
so contiguous vector windows may now exist for the collective to use. A
profile of the AIV case would settle it; none exists, because
`--compile-target` still cannot be profiled.

## The default resolves to AI_CPU

The unset case lands 0.048 ms from explicit AI_CPU and 0.511 ms from AIV, so
leaving the variable unset gives the AI_CPU path. That removes a confound
raised against [the idle-machine comparison](../compose-20261018/README.md):
the `a3-gs` reference set AI_CPU explicitly while the `a3-84` runs left it
unset, and those are now known to be the same path, so the 3.8% between them
is a machine or run difference rather than a mode difference.

New best measured TPOT is 15.978 ms, 7.99× the 2 ms target, from one
environment variable. `precision_verified` remains false.
