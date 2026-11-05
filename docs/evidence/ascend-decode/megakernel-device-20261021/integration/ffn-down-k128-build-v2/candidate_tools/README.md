# Down K128 with one L0B buffer

This private candidate changes the accepted `bbd52e42` FFN's Down geometry from M32/N1024/K64 to M32/N1024/K128. The installed CANN 9.2 constant tiling API supplies six named fields. The runtime host tiler first generates and verifies the complete accepted `c207230a` record, applies those fields, and verifies the complete `c2b73af9` record. Both records are 200 bytes and identical between ND and NZ weights.

`static_down_plan.cpp` is a host constant provider compiled with Bisheng for dav-2201. Its 24-byte PODs contain `baseK`, `depthA1`, `depthB1`, `stepKa`, `stepKb`, and `dbL0B`. They are linked into the host DSO and read by the host tiler. The planner disables its unused INT32-output quantization reserve; the device Matmul configuration remains unchanged. Copying the whole constant tiling record would be incorrect: even the accepted control plans have compile-time `transLength=131072`, whereas the accepted runtime record has `transLength=0`. The other 176 runtime bytes remain unchanged.

| Field | Parent | Candidate |
| --- | ---: | ---: |
| baseK | 64 | 128 |
| depthA1 | 4 | 2 |
| depthB1 | 12 | 6 |
| stepKa | 2 | 1 |
| stepKb | 2 | 1 |
| dbL0B | 2 | 1 |

The device consumer remains `CFG_NORM` with dynamic `SetSingleShape(2 * count, 6144, 128)`. For one expert, six N1024 tiles cover all 6144 columns. Counts 1–16 use one M tile; counts 17–32 use two. K128 requires one INT32 MMAD per output tile, replacing two K64 steps. Each dot product has magnitude at most `128 * 64 = 8192`, so regrouping these integer products introduces no rounding or overflow. All later FP16 MSD reconstruction, BF16 route/combine boundaries, shared branch, five body joins and terminal publication retain their parent statements.

The resource contract is L0A operand 2 KiB, L0B operand 64 KiB, L0C 128 KiB, and L1 397312 bytes: `2 * 32 * 128 / 2 + 6 * 1024 * 128 / 2`. The CANN `TBufPoolL0Base` selects ping-pong jointly from `dbL0A` and `dbL0B`; setting B to one buffer also stops A ping-pong. `Allocate/EnQue/DeQue/Free` still issue both directions of the MTE1/M event protocol. This reduces K-loop work but can remove useful load/compute overlap. No UB or GM allocation is removed. The routed body uses 116480 UB bytes; the complete FFN's preceding norm stage peaks at 195584 bytes.

The public contract remains 33 scratch tensors, a 408-byte body descriptor and a 112-byte shared input POD. The private namespace is `tilert_ffn_down_k128`; the C ABI is `ffn_down_k128_launch`. Source normalization must recover every parent C++ device consumer exactly. CPU tests check all counts 1–32, the original NZ full-K stride, both complete tiling records, six-field isolation and the original ABI. The build verifies that the planner object is a relocatable AArch64 host ELF, checks its linked ND/NZ PODs, runs both actual host tilers, and co-loads parent/child Torch schemas without initializing NPU.

Device validation is pending. The required gates are the original Decoder norm/MLP oracle on M1/2/12/32, hot eight-expert routing with counts 17/32 under both 16/24 groups, and a same-owner comparison against `bbd52e42` with private immutable Down tilings. A positive standalone result must then be tested in the same 609 + TopK-copy whole-layer chain against N512; results from different comparisons are not additive.
