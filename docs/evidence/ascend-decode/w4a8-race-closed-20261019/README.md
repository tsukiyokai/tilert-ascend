# The W4A8 non-determinism was a cache, not a race, and width 8 now works

Two defects stacked. The barrier patch this project wrote weeks ago was
correct; a package cache in the static-kernel path silently discarded it at
exactly the captured width, which is why every attempt to verify the barrier
read as "installed but ineffective".

## The defect that hid the other one

`vllm_ascend/compilation/compiler_interface.py:128-139` turns on
`static_kernel_compile` without setting `disable_static_kernel_compile_cache`.
Before compiling, `_merge_and_compile_install`
(`torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/static_kernel.py:385-391`)
asks a cache whether a package for this key exists and installs it if so. The
key, built at `:1071-1092`, covers the dumped operator parameters, the
super-kernel option string and `torch.are_deterministic_algorithms_enabled()`
— and nothing else. Not `ASCEND_CUSTOM_OPP_PATH`, not the vendor headers the
compiler actually reads, not their contents. The index file lands under the
process's working directory, so a run inherits whatever package an earlier run
left there, built in whatever tree that run used.

Static kernels are compiled only for the capture sizes
(`compiler_interface.py:135`), so only the captured width was hijacked. Every
other width took aclnn dynamic dispatch and got the patched object — which is
precisely the pattern that looked like a width-dependent race.

The barrier itself is one line at
`grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h:209`, after the `DataCopy`,
already carried as `tests/glm52/patches/gmm1-swiglu-barrier.patch`. Owner
overlay against in-tree differs in exactly that: `post.h` md5 d8c4c4c2 against
00e14bca, `diff` is `209a210 > PipeBarrier<PIPE_V>();`, `pipeline.h` identical
on both sides, and of seven objects only
`GroupedMatmulSwigluQuantV2_fa3d6d3d….o` differs.

## The differential

Six `target.verify` calls on one frozen input, counting distinct hidden
states. a3-84, 16 ranks, `--compile-target --super-kernel --fp32-head
--draft-projection-tp`, no hoist overlay, `HCCL_DETERMINISTIC` unset.

| condition | static kernel package | width 8 | width 4 |
| --- | --- | --- | --- |
| `--compile-target` only, same script and directory | none | 1/6 | 1/6 |
| compiled in this process, patched vendor first | fresh | 1/6, over 8 processes | 1/6 |
| installed from this directory's own earlier build | cached, patched | 1/20 | 1/20 |
| compiled in this process, in-tree vendor first | fresh, unpatched | 6/6 | 6/6 |
| installed from `snapshot-final`'s index | cached, built elsewhere | 6/6, over 3 processes | 1/6 |

`--super-kernel` is necessary: two arms differing only in that switch give 1
against 6. Putting the unpatched in-tree vendor first breaks width 4 as well,
6/6 — the race's own signature, agreeing with this project's own 64-call matrix
where M=32 was worst. Installing from a cache is not itself fatal: this
directory's own patched package gave 20 repeats with no deviation. So the
carrier is the package's provenance, not caching. Object sizes pair with
behaviour: cached 264128 = unpatched-fresh 264128 ≠ patched-fresh 264088.

## End to end

1024 output tokens, three warmups and three measured requests, `--width 8/4
--target-graph --draft-graph --draft-projection-tp --compile-target
--super-kernel --attention-kernel sfa --fp32-head`. Both width-8 arms run from
`snapshot-final` and both resolve gmm1 to the patched overlay at index 3; the
only variable is this patch.

| arm | rounds | streams | median TPOT | range |
| --- | --- | --- | --- | --- |
| width 8, unpatched | 251, 266, 258 | 3, none matching | 14.4640 ms | 0.8220 |
| width 8, patched | 227, 227, 227 | 1, `68f2bc184553` | 13.0657 ms | 0.8178 |
| width 4, patched | 307, 307, 307 | 1, `68f2bc184553` | 14.5551 ms | 0.1045 |
| width 8, fresh working directory | 227, 227, 227 | 1, `68f2bc184553` | 13.0659 ms | 0.0968 |

−1.4894 ms a token against the same session's width-4 arm, −1.4220 against
this project's recorded 14.4877 baseline. The trajectory gate passes: width 8
emits the width-4 stream. Width 4 is unchanged. Across 36 recorded
1024-token runs on this machine, no width 6 or width 8 arm had ever emitted
`68f2bc184553`. `--expert-parallel` is stable at `fa509e7339c2`, a different
stream — it bought reproducibility and never bought correctness. The broken
width 8 also forfeited its own acceptance advantage: 258 rounds against 227,
at a median within noise of width 4.

## Both named candidates are cleared

The shared expert's `npu_dequant_swiglu_quant` (`shared_experts.py:546`) is
excluded on three independent grounds: 4 groups of 64 consecutive calls at
production shapes, bit-exact each time with
`max_abs_code_difference_from_call_zero` zero throughout; its compiled static
kernel has `blockDim` 1 and `sub_op_with_sync_all` false, so there is no
cross-core synchronisation to lose; and `dequant_swiglu_quant.h:788-802` is
word-for-word CANN 9.2.0's own `:784-799`, so not a local regression.

`apply_gmm2`'s `GroupedMatmul` A8W4 is not an arithmetic problem either: 8
groups of 64 calls, 512 in total across two layouts, `distinct_results` 1 and
`unequal_elements` 0 everywhere. It and `GroupedMatmulSwigluQuantV2` are
indeed the only two fused MoE operators calling `AscendC::SyncAll` inside the
kernel, but the differential puts the carrier on gmm1's barrier.

## Two records corrected

The arms that concluded "the vendor epilogue barrier is installed and makes no
difference" did resolve to patched objects, so that refutation stands **for the
dynamic-dispatch path**. What failed was the static-kernel path. Those arms
also carried a hoist overlay on `PYTHONPATH` and `HCCL_DETERMINISTIC=true`;
this reproduction needs neither.

The super-kernel fusion options are a dead end that cannot be tested in
isolation: `super_kernel_optimize_options` participates in the cache key, so
changing it necessarily forces a fresh compile. Every such arm — `debug-dcci-all`,
`debug-sync-all`, `early-start`, `enable_super_kernel=False`, the blacklists —
came out clean only because it recompiled. They are not evidence.

## What is still open

The cached and freshly compiled packages agree field for field across 245
kernel descriptors except `binFileName`, `kernelName`, `sha256` and the
`obj_files` paths. So "the cached package was built from unpatched source" is
carried by behaviour pairing plus object-size pairing, not by a semantic field
in the artefact.

The width-4 reproducer's non-constant round count was not reproduced; a
higher-sensitivity detector was used instead. Re-reading the existing records
confirms the reproducer exists (`e2e-patched` 322/331/322 against 307/307/307,
all three sharing `68f2bc184553`), but that arm's gmm1 object came from a
different overlay, so its perturbation mixed in a kernel-object change.
