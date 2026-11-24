# GU constant-tiling source and generated-code audit

Candidate `05b3427aa29630c9af16ea829e2ecec8833d31a32615e6a052f6f4e9867ca6b2` specializes only the routed GU `Matmul` tiling in parent `276352762139244a1ddfc8184065a856c2ae75ee2ccad5961af2af93f0dd5e00`. The scheduler remains `CFG_NORM`; runtime expert counts, matrix element order, output layout and every downstream arithmetic boundary remain unchanged. The retained [host tiling](gate-tiling.jsonl) supplies the same 21 constants for ND and NZ weights. M and singleCoreM remain dynamic.

The installed CANN 9.2 files were read from a3-78's SDK. [Installed identities](installed-identities.json) retain each original path, byte count and SHA-256. They describe this SDK version, not an unpinned upstream implementation.

| Obligation | Installed source |
| --- | --- |
| A constant field replaces a tiling read only when its value is not `-1` | `impl/adv_api/tiling/matmul/matmul_constant_tiling_struct.h:25–180` |
| Runtime `SetOrgShape` and `SetSingleShape` continue to update effective M | `impl/adv_api/detail/matmul/param/matmul_shape_info.h:42–58,115–145,188–214` |
| SetTensorA/B reset scheduler state; End releases matrix buffers | `impl/adv_api/detail/matmul/matmul_impl_base.h:560–565,598–603`; `scheduler/base/scheduler_base.h:148–167` |
| The normal scheduler loads the current K tile; the single shape controls its K loop | `impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h:193–240`; `scheduler/iterator/k_loop/k_loop_norm.h:49–79` |
| BasicBlock rejects INT4 and INT8 input types | `impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h:104–109` |
| A3 uses Load3D for A and LoadDataWithTranspose for INT4 B | `impl/adv_api/detail/matmul/feature_trait/matmul_chip_cap.h:70,99`; `stage/split/load_to_l0a/load_to_l0a_loadInstr.h:116–135`; `stage/split/load_to_l0b/load_to_l0b_loadInstr.h:148–174` |
| Small M still requires the installed matrix-pipeline barrier | `impl/adv_api/detail/matmul/stage/compute/mmad_compute.h:77–87` |

Paths in the table are relative to [installed](installed/); abbreviated paths inherit `impl/adv_api/detail/matmul/`. The candidate does not remove `SetTensorA/B`, `IterateAll`, `End`, or the small-M `PIPE_M` rule. That rule tests `(mmadM / 16) * (mmadN / 16) < 10`; for N256 and effective M=`2c`, it includes `c ≤ 7`.

For an expert with `c` routes, GU computes INT4 `[2c,6144] × [6144,256] → INT32[2c,256]`. Logical A/B/C payloads are `6144c`, `786432` and `2048c` bytes; arithmetic is `3145728c` multiply-accumulates in 24 K256 steps. M12/E65 has 96 routes, 48.75 MiB of selected GU weights and 1,560 matrix-step invocations across the AICs. M12/E8 has 6 MiB and 192 such invocations. These are workload formulas, not measured HBM traffic or a claim that real model routes use either distribution. The [work model](work-model.json) retains the equations and resource sizes.

L0 reservations are unchanged: A double buffer 16,384 bytes, B double buffer 65,536 bytes, C single buffer 65,536 bytes. The original L1 reservation is 458,752 bytes. No extra GM or UB is allocated. The original active-expert enumeration, descriptor and five body joins remain intact.

## Actual generated code

[audit_assembly.py](audit_assembly.py) extracts each object's first embedded device ELF, checks machine ID 4137, disassembles with the installed `llvm-objdump`, and compares all four ND/NZ AIC/AIV variants. [The result](assembly-v2/summary.json) records object, embedded-ELF and disassembler hashes. [Instruction excerpts](assembly-v2/excerpts.txt) retain the GU transpose-load and matrix sites with their original addresses; [instruction-word hashes](assembly-v2/instruction-words.json) make the AIV equality independently checkable.

| Device variant | Parent instruction sites | Candidate instruction sites | Difference |
| --- | ---: | ---: | ---: |
| NZ AIC | 13,180 | 11,809 | −1,371 |
| ND AIC | 13,474 | 11,948 | −1,526 |
| NZ AIV | 5,810 | 5,810 | Instruction words identical |
| ND AIV | 5,810 | 5,810 | Instruction words identical |

NZ AIC loses 215 scalar load sites, 152 scalar store sites, 107 jump sites and 17 integer division/remainder sites. ND AIC loses 243, 152, 116 and 18 respectively. Both AIC variants retain two INT4 MMAD sites, two INT8 MMAD sites and four INT32 Fixpipe sites. The GU INT4 transpose loop becomes four adjacent load sites, while the separate Down transpose site remains. Therefore raw opcode-site counts must not be interpreted as dynamic memory operations: unrolling can add sites without adding work, and constant folding can remove untaken paths. The matrix steps, selected weights and full FFN work are unchanged by the source contract.

The compiled intervention is observable. The [complete geometry gate](../ffn-gu-static212-wide-v1-a3-194/summary.json) passes 4,480 raw-byte checks at M1/2/12/32 and 16/24 groups, including 512 immediate-consumer checks and complete INT32[264] active-table comparisons. The separate [same-address full-kernel experiment](../ffn-gu-static-timing-v1-a3-194/summary.json) measures paired median savings of 5.4344 and 6.1665 µs in the two M12/24 E65 arenas, with 60/60 faster pairs. This supports the specialization's local FFN effect; it does not isolate a particular opcode class or establish end-to-end TPOT.
