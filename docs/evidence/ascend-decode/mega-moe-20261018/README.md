# CANN MegaMoE on A3: availability and the shape that blocks us

MegaMoE fuses dispatch, Linear1, SwiGLU, Linear2 and combine into one call,
and on A3 it also absorbs the shared expert. Our expert GMM cost is
per-call fixed work rather than traffic
([GMM shape scaling](../gmm-shape-20261018/README.md)), so collapsing a
layer's five or more calls into one is the right shape of remedy.

## It needs no build

`cann_ops_transformer` ships inside CANN 9.2.0 at
`/opt/ascend/cann-9.2.0/python/site-packages`, and in the `trt-asc`
container `from cann_ops_transformer.ops import mega_moe,
get_symm_buffer_for_mega_moe` succeeds with the JIT bindings already
compiled, printing only `ninja: no work to do`. Adding that directory to
`PYTHONPATH` is the whole installation. CANN registers `MegaMoe` for
`ascend910_93` with nineteen inputs, `weight1`/`weight2` accepting `int4`
in three of eighteen dtype combinations and `weight_scales` as `uint64`,
and vLLM-Ascend declares `HardwareCapability.CANN_MEGAMOE` for the A3
profile. The A3 kernel is `op_kernel/arch22/mega_moe_kernel_a3.hpp`, whose
W4A8 path uses `block_epilogue_w4a8post_pertoken_swiglu.hpp` and stages the
activation's two int4 halves through `workspaceInfo.ptrA1Int4` and
`ptrA2Int4`, the same MSD split the built-in GMM performs, but once per
layer instead of once per call.

## GLM-5.2 fits every A3 constraint at EP16

Checked against the checkpoint's `config.json` and the operator's A2/A3
constraint table.

| constraint | required | GLM-5.2 |
| --- | --- | --- |
| `hidden` | 1024–8192, % 512 | 6144 |
| `intermediate_hidden` | 512–3072, % 512 | 2048 |
| `num_experts` | ≥ `ep_world_size`, ≤ 1024, % `ep_world_size` | 256 |
| `num_experts_per_rank` | 1–128 | 16 at EP16 |
| `num_topk` | 1–16 | 8 |
| shared expert | `shared_weight1/2` | 1 |
| scene | A16W16, A8W8-INT, A8W4-INT | W4A8 |
| `ep_world_size` | 2, 4, 8, 16, 32 | 16 |

## What blocks the TP layout

Our MoE is TP16 with every rank holding all 256 experts and the
intermediate sharded to 128. That violates `intermediate_hidden >= 512` and
the multiple-of-512 rule, so no EP setting rescues it. The tiling's
`VALID_EP_WORLD_SIZE[] = {2, 4, 8, 16, 32, 64, 128}` would also reject
EP=1, and no `epWorldSize - 1` or `== 1` assumption appears elsewhere in the
host or arch22 kernel, so that whitelist is one line; but relaxing it does
not help while the intermediate is 128. The intermediate rule is the same
constraint that blocked an earlier 505B attempt at `moe_intermediate 1792`.

Using MegaMoE therefore means moving the MoE from TP16 to EP16, where each
rank owns sixteen experts at the full intermediate of 2048. That also
removes the per-layer MoE AllReduce, since combine performs the reduction:
78 of the 156 reductions a token now pays. The cost is load imbalance —
thirty-two (token, expert) pairs over sixteen ranks put about five experts
on the busiest rank, so its critical path reads 94.5 MB and streams for
82 µs where TP16 reads 9.4 MB and streams for 8.1 µs.

Whether that trade pays depends on one number that is not yet measured: the
fused call's fixed cost on A3.
[The EP sweep](../../../../tests/glm52/bench_mega_moe.py) separates it from
the streaming slope by holding the token count fixed and raising the number
of distinct experts the routing selects, the method that worked for the
built-in GMM. It needs ranks on an otherwise idle machine and has not run.

vLLM-Ascend's own integration records an unfixed W4A8 accuracy problem on
A3 and notes that its MegaMoE constants were adapted to A5, and its config
validator comments that the operator has many problems and was rolled back
internally behind `enable_fused_mc2=2`. Numerical acceptance cannot be
skipped for this path.
