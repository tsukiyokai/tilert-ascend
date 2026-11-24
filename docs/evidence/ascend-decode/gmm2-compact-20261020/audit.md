# B114 NZ down GMM active-group candidate

The inspected runtime path is GroupedMatmul_a0be48d01b162435398ed52e1b6182d4, ELF SHA256 fa20701770aa8a575946ad502fcc770184b2a21b4d4e727e4c864aab0faa34d7, key 69325233410: INT8/NZ INT4/BF16, count list, API MSD dequant, AIC:AIV=1:2. This is the original two-half MSD path, not vector dequant or fake-A8W8. Current CANN is 9.2.0 / V100R001C25B114.

Runtime evidence supplied and independently read: a3-194 host /home/shanshan/trt-asc-194/experiments/gmm2-compact-20261020/key-logs/debug/plog/plog-9443_20261020024243437.log.
- line 50500 selects the a0be48d0 object; 51014 launches key69325233410; 51073 names that same actual kernel (the following registration lists are not execution evidence).
- 50557/50593 report baseM64/baseN512; 50571–50573 report final baseM64/baseN512/baseK128.
- 50618 gives operator workspace117440512B=112MiB; 50622 gives executor117441024B, including an extra512B. The extra512B is not borrowed.

Installed source root is /opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul/op_kernel/. Frozen files are in original/; manifest.json binds candidate files. dynamic/grouped_matmul.py:113–129 locates these sources; 273–296 defines their compiler invocation.

Workspace proof: withOffset0/preNZ0 makes mmOutGm=user1 (msd.h:193–209). The sole live user-GM scratch is a ring of32768 half elements/core/slot (45,299,413–446). Under the candidate guard core24/parall4/quantGroup1 its end is 32768*2*24*4=6291456B. Each tile is at most64*512=32768 half; all MM and vector accesses remain in that ring. The observed operator allocation leaves96MiB after the separate16MiB system region. Metadata occupies user1+[6291456,6293536), inside that unused allocation. No xGM2 assumption, system workspace borrowing, ring overlap, or host allocation change is used. The matching local host allocation formula is grouped_matmul_tiling.cpp:2327–2328; local host source is supporting mechanism, not a byte-verified copy of the installed host library.

Candidate changes only:
- grouped_matmul.cpp: AIV0 publishes before the existing pre.Init, with explicit API-dequant if constexpr. Its temporary UB is reset only after MTE3_S. The original pre:213 mixed barrier pairs with msd.h:269 AIC barrier and publishes the table.
- msd.h: replace all-core scalar count traversal at271–293 with stable active (expert_id,cumulative_end) iteration; invalidate consumer DCache after that existing barrier. Every active expert keeps its original ID, m=2*(end-previous_end), offsetM=2*previous_end and original preCount/core assignment.
- MMCompute and VectorCompute, including NZ addressing, scale/bias indexing, FP16 high/low processing, BF16 rounding and flags3/5, are byte-identical. Both AIC and AIV consume the list, so no second dense scan remains in this path.
- Guard: BF16, E256/K128/N6144,0<M<=256,core24/parall4,quantGroup1,offset0,preNZ0,singleTensor1. COUNT comes from this template's key; do not require gmmBaseParams.groupListType==1, which this special host path does not explicitly fill.

Native sparse group-list type2 is not a caller-only replacement for this path: tiling_key.h:50–60 registers A8W4 only with COUNT, while msd.h:272 treats each scalar slot as a count and the loop index as the expert ID. The sparse implementations used by other dtype templates do not change this class.

CPU evidence: cpu-result.json covers102 M8/64/256 distributions, all24 per-core schedules equal,86652 tiles/arm. It compares actual ID/address/tile/ring reuse/wait order and verifies the arithmetic suffix is unchanged; it does not emulate CANN arithmetic or device visibility. Same A but experts0..7 versus248..255 remains a stale-table witness. Reusing a mutated activation between arms is invalid; existing CANN pre modifies x in place.

Artifacts: gmm2-active-groups.patch SHA256 5532e3438a0cf7d246a12c28773827f73efe6602561555abecbd8e5230eb581b; cpu_invariants.py and cpu-result.json; no NPU execution by this audit.

