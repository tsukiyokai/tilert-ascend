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

Device numerical and paired performance validation are required before this
composition can replace either frozen parent. Independent component gains
do not establish its performance.
