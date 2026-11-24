# Tree reduction in the TP layer

This private candidate replaces the two rank-ascending FP32 reductions in C2
(`c2b72e46`) with the accepted `communication_tree` postorder reducer
(`a0bd2462`). Its full-layer device result is pending. The reducer alone passed
[three-rank numerical and graph checks](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/communication-tree-three-rank-v1-a3-194/AUDIT.md)
with the original serial probe and FP32 HCCL as controls.

The layer uses the parent's norm, attention, and `381feb` FFN sources without
arithmetic changes. The 664-byte configuration, 928-byte addresses, 48-byte
communication record, stream check, symmetric window, epoch protocol, phase
joins, and caller-owned TopK copy retain their original contracts. The Python
`Library`, `Window`, and `Prepared` interfaces retain their owner checks;
their C symbols use `layer_tp_tree_`. The private device namespace is
`layer_tp_tree_chain`.

Each communication phase starts after the parent resets its `TPipe` arena.
Its 2048-element chunk holds 4096 bytes of BF16 input and five FP32 stack slots
of 8192 bytes each: 45,056 bytes total. The subsequent parent `EndPhase` drains
and resets this arena before FFN allocation. The final reduction is followed
by the parent's `PIPE_ALL` and DDR publication before kernel completion.

`prepare.py` checks both frozen ancestors, copies the accepted tree header,
plan, reducer and communication host implementation byte for byte, and records
the exact namespace-only changes to the layer. `build.py` reuses the C2 compile
commands, including all five routing/Matmul definitions. Dependency output is
the only added compiler option. CPU checks cover all ranks 2 through 16,
source equality, ABI geometry, exports and parent/candidate loading without
SHMEM initialization or mapped NPU devices.

The two layer DSOs have separate host context registries but link the same
SHMEM runtime. CPU co-loading does not authorize two simultaneous initialized
contexts. A device fixture must initialize one selected layer library and
destroy every graph and window before finalization. Numerical acceptance must
compare genuine rank-distinct layer partials with FP32 HCCL, including the
retained residual, caches and ordered TopK output; three-rank reducer success
does not establish TP16 model correctness or full-layer performance.
