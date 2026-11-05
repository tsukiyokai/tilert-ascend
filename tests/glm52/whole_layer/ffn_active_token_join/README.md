# Active-expert FFN with token-owned output

This candidate combines active-expert enumeration and the 408-byte body
descriptor with token-owned route reconstruction, ordered top-8 accumulation
and the BF16 output epilogue. The body ends with one publication and mixed-core
join before resource reset. Instrumented builds are rejected.

The parent is the frozen `f7621e49` active-descriptor FFN. The output arithmetic
and final synchronization are copied from the verified `4417fd15` token-join
implementation. `prepare.py` checks both source manifests and preserves the
parent's 33-buffer public interface, descriptor layout, dispatch and all
projection stages. The token-output helper adds 3,072 UB bytes; the combined
resident allocation is 116,480 bytes, including the 1,056-byte active table.
No additional GM storage is allocated.

The [M12/24 device gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-token-join212-m12-v1-a3-194/summary.json)
passes 496 raw-byte boundary comparisons over eight changing inputs against
both the native orchestration and official bridge references. Every check
includes the complete INT32[264] active table. The frozen DSO is
`8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84`.
The [remaining seven geometries](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-token-join212-wide-v1-a3-194/summary.json)
also pass, completing all 3,968 raw-byte checks at M=1/2/12/32 and 16/24 groups.

The [paired benchmark](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-token-join-timing-v1-a3-194/summary.json)
compares this DSO with `f7621e49` using identical
33-buffer addresses and 16 weight banks. CPU probes compile against both
actual frozen contracts and verify every pointer, launch argument and POD
offset on x86 and ARM. The original numerical, sentinel and event-window
checks remain unchanged. At M12/24 with 65 active experts, the two owner
arenas give candidate/parent medians of 216.462/223.297 us and
211.493/227.337 us. Median paired savings are 6.843/15.850 us, with all
60 pairs faster. All 19,200 raw-byte boundary, 360 event-tail and 696
full active-table checks pass. These are local synthetic FFN measurements,
before TP reductions and whole-model execution.
