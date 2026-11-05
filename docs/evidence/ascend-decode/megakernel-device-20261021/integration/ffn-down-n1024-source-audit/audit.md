# Down geometry and paired MSD reconstruction

The next isolated candidate is M32/N1024/K64 Down tiling from the accepted c90 complete FFN. Installed CANN 9.2 accepts both ND and NZ and produces the same complete 200-byte record, `c207230aafb11e2c2f825cd9a2ea70147ee7733046b8b2957593c9c2bc3e8009`. The host probe's accepted control reproduces c90's `bb8ef448…` record. [Raw host results](host/summary.json), [source identities](summary.json).

## Exact loop change

The parent [Project](parent/body/source/routed_int4/routed_body_native.h) at lines 176–200 supplies logical M=`2*c`, N=6144, K=128 to both `SetOrgShape` and `SetSingleShape`, then calls `IterateAll`. Down uses CFG_NORM; the separate static GU type is unchanged.

The installed [IterateAll](installed/matmul_impl.h) at lines 118–135 calls `Iterate` followed by one `GetTensorC` for each MN tile. [Iterate/GetTensorC](installed/matmul_impl_base.h) at lines 692–720 route to `ScheduleOnce` and `GetResult`. [ScheduleOnce](installed/scheduler/base/scheduler_norm.h) at lines 119–148 advances MN once and performs the complete K reduction before returning. Lines 193–240 contain that K loop. [KLoop](installed/scheduler/iterator/k_loop/k_loop_norm.h) at lines 49–57 and 94–111 uses `ceil(singleCoreK/baseK)` and advances one K tile per iteration.

| Expert count | Parent C results / MMAD calls | Candidate C results / MMAD calls |
| --- | ---: | ---: |
| 1–16 | 12 / 12 | 6 / 12 |
| 17–32 | 12 / 12 | 12 / 24 |

Every M12 expert has `c <= 12`, so the candidate removes six C-result extractions per nonempty expert. M32 hot experts are the direct counterexample to a universal reduction. The candidate adds a second K iteration inside each result, including K bookkeeping, L0 allocation/enqueue/dequeue and split-data decisions, but halves the number of result iterations when `c <= 16`. At `c > 16`, it adds a second M block instead.

[GetResultImpl](installed/scheduler/base/scheduler_base.h) at lines 405–427 invokes the nonsequential C copy once, then frees that L0C result. [CopyOutNZ2ND](installed/stage/copy_cube_out/copy_cube_out_fixpipe.h) at lines 154–185 and `GetDstOffset` at 205–235 retain the GM row stride 6144, with offset `m_tile*baseM*6144 + n_tile*baseN`. This is the actual no-callback path. [MLoop](installed/scheduler/iterator/m_loop/m_loop_norm_base.h) at lines 140–171 limits each last block to the remaining valid rows. Thus the M32 second block writes rows 32 through `2*c-1`; it must not write the physical padding through row 63 when `c < 32`.

Two K64 pieces update one INT32 accumulator before Fixpipe. Each complete signed INT4 dot has absolute value at most `128*64=8192`, so the partition adds no INT32 overflow or floating-point boundary. Every original FP16 MSD, BF16 route/output, probability-zero skip and ordered slot combine remains outside this unchanged integer projection.

The L0 budgets are A 2,048 B, B 65,536 B including double buffering, C 131,072 B; reported L1 remains 397,312 B. This does not imply reduced HBM traffic. [CopyCubeIn](installed/stage/copy_cube_in/base/copy_cube_in_norm.h) at lines 84–110 caches L1 tiles, and `SplitA`/`SplitB` in [scheduler_norm.h](installed/scheduler/base/scheduler_norm.h) at lines 244–276 cache L0 tiles. K64 changes cache keys and transfer geometry. The exact dynamic misses require device measurement. [MmadCompute](installed/stage/compute/mmad_compute.h) at lines 77–81 retains the small-M `PIPE_M`; for `2*c < 16`, both configurations still execute twelve such checks/barriers. No barrier saving is assumed.

## Alternative: pair high/low MSD reconstruction

The independent alternative changes only Down's token-owned reconstruction in [output_native.h](parent/source/output_native.h). Each 1536-column token tile processes eight original slots, calls `LoadScale`, then reconstructs high and low partials through two [ScaledPartial](parent/body/source/routed_int4/routed_body_native.h) calls at lines 272–295. Each half preserves INT32→FP32, channel-scale multiplication, FP16 RINT, FP32, then the original `16*high + low + correction` ordering. Activation scale and all BF16 boundaries follow unchanged.

A paired implementation could fetch both rows into one 3072-element INT32 buffer, convert both together, perform the two disjoint scale multiplies, narrow both to FP16 and widen both together. A two-block DMA uses block length 192 units of 32 B and source gap `768*c-192` units; the second block starts exactly `c*6144` INT32 elements after the first. Keep the existing 1536-element scale vector and use separate paired staging/value/half buffers. This adds 30,720 B UB, raising the routed phase from 116,480 to 147,200 B, below the unchanged norm peak of 195,584 B.

Per route/tile, this could remove one DMA command, two Set/Wait event pairs, three vector commands and four vector pipe barriers without changing payload bytes. There are `32*M` such pairs. The FP16 rounding must remain independent for both halves; moving `16*high+low` before narrowing is invalid. Signed zero, zero-probability NaN suppression, slot-zero seeding and BF16 scale/add boundaries must remain exact. The larger live UB region and different vector schedule may offset the saved commands.

## Choice and falsification

N1024 is first because it changes one host tiler call, adds no GM/UB and leaves device arithmetic source identical. Its precise adverse case is `c > 16`; its uncertain cost is the changed K/L0 schedule. Paired MSD is second because it removes vector command overhead but adds live UB and a new two-row DMA. Neither command counts nor the earlier successful N512 experiment establish N1024 performance.

The N1024 device gate must use the actual original Decoder norm/MLP, full prepared 200-byte tiling checks, all production bytes and immediate consumers at M1/2/12/32 and 16/24 groups. Hot M32 routing must exercise the second M tile. Paired timing compares c90 and the candidate with the same other 32 scratch addresses and weights, preserving each arm's own body[7] TensorImpl and full tiling bytes. Two owner placements and at least 30 alternating event samples per arm decide the candidate. A slowdown or byte mismatch rejects the corresponding claim; no result here predicts a layer or TP16 latency.
