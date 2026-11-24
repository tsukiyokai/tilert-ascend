# Frozen M1 versus M32, first eight layers

The actual rank-0 M1 and M32/SFA artifacts agree bit for bit for hidden,
residual, merged boundaries, following-layer inputs and newly written cache
rows: 68 tensor pairs. Hidden/residual alone cover 3,145,728 BF16 values.
Their NRMSE and maximum absolute difference are zero at every layer 0–7;
FP64 cosine calculations are 1 within 2.3e-16 arithmetic rounding.

All 32 selected-token sets are identical at every layer: 524,288 index
entries, 100% per-query overlap. The order differs in a few positions:

| Layer | Different ordered entries | Same set for all 32 rows |
| --- | ---: | --- |
| 0 | 0 | yes |
| 1 | 40 | yes |
| 2 | 6 | yes |
| 3 | 6 | yes |
| 4 | 6 | yes |
| 5 | 6 | yes |
| 6 | 0 | yes |
| 7 | 0 | yes |

The first ordered-index difference is layer 1, query 1, column 1153:
M1=967, M32=1063. There is no hidden/residual/cache first difference.

## Inputs and execution identity

Checkpoint config, quant description and weight-index hashes agree. Both
reports bind the same 8K prefix, request/generation, original prompt bytes,
compact prompt IDs and teacher SHA `82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655`.
The 404 GB checkpoint payload was not rehashed by this comparison.
Both use FP32 TP reductions, deterministic AI_CPU HCCL and a 200 MiB HCCL
buffer. The actual `_tp_reduce` AST is identical. Package versions agree.

Old M1 ran on a3-gs; M32 ran on a3-194. Capacity changes 8320→9344,
width 1→32, layer count 78→8 and `TASK_QUEUE_ENABLE` 1→unset. The old head
was BF16; the new flag selects FP32, but neither comparison uses final-head
outputs. The rotary source adds only two `.contiguous()` calls to derived
cos/sin tables; its exact diff is saved.

The native CANN package is 9.2.0/B114. CPU Div calibration modifies the CPU
reference, not these eager native runner calls. The old run did not save a
specific loaded DynamicQuant ELF receipt, so identical loaded quant binaries
are not asserted. GMM overlays differ: old recorded barrier ELF `2eb81546…`,
new production ELF `baaa18e2…`, with OPEN/ACCESS evidence for the new run.
`identity.json` retains full hashes, mapped libraries and these differences.

The existing M32 audit checks hidden/residual/indices across all 16 ranks.
The old full-M1 audit checks final logits across ranks and within-rank
handoffs; it does not establish first-eight-layer hidden equality across
ranks. This comparison reads old rank 0 only. It connects that M1 reference
to the new M32 outputs at layers 0–7, not the remaining 70 layers or teacher
successor acceptance.

## Artifacts and reproduction

`result.json` contains raw per-layer metrics, per-query overlaps, first
differences and tensor-file hashes; `summary.json` is the small result.
`compare.py` runs on CPU with backend autoload disabled and confirms
`torch_npu` was not imported. No tolerance gate or NPU call is used.

Raw tensor files remain on servers. The copied old rank-0 subset is
20,945,401 bytes under the experiment's `old/` directory on a3-194.
Run `bash command.sh` from a3-194 host to repeat this exact comparison.
