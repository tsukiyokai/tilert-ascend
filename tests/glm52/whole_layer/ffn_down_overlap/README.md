# Down projection and route dequantization overlap

This independent candidate starts from composed library SHA256 `3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655`. It retains the full INT32 down-projection buffer, all 30 output boundaries, all six body barriers, both outer `EndPhase` calls and the original numerical helpers. It tests tile publication and consumption while CUBE continues the down projection. Active-expert tables, descriptors, token-owned output and tail-join changes are excluded.

## Data and numerical contract

The host checks the generated down tiling: base M/N/K=64/256/128, single-core M/N/K=64/6144/128. For an expert with c routes, the original matrix shape is M=2c, N=6144, K=128, with c at most 32. One M tile and 24 increasing N tiles cover the full result. Each AIC keeps its original increasing nonempty-expert ownership. It calls `Iterate` and `GetTensorC(c, 0, false)`, preserving the full-GM row stride and tile offsets used by the SDK's default `IterateAll` implementation.

The two associated AIVs consume columns `256*t + 128*subblock` through `+128` for every route of the same expert. They call the original `StoreRoutes` helper. High/low INT32 partial reconstruction, FP16 partial boundaries, correction and activation-scale order, BF16 route output, original top-8 slot placement and ordered combine remain unchanged. The helper performs no cross-element reduction, so its 128-column span changes ownership and memory operations without changing elementwise arithmetic. Original route outputs are written even for zero-probability slots; `CombineTokens` retains its zero-probability skip and slot-zero signed-zero seed. The outer routed BF16, scaled BF16 and shared-add BF16 boundaries are retained.

## Credit and lifetime contract

Mode2 uses ready flag 8 and free flag 9. CANN defines one AIC publication as releasing both AIV waits, and requires both AIV publications before releasing one AIC wait. These flags avoid Matmul's reserved 0–7 and mixed `SyncAll`'s 11–14. The SDK counter range is 0–15; this protocol permits at most four unconsumed tiles.

The AIC sequence number continues across every owned nonempty expert. Before publishing tile q≥4 it consumes one free credit. After the last tile it drains `min(total_tiles, 4)` remaining credits. Each AIV waits once and acknowledges once per tile, so an AIC with E owned experts performs exactly 24E ready publications and 24E free waits. Empty experts contribute no operations. The two AIVs each perform 24E ready waits and 24E free publications. All counters are drained before barrier 4 and before a later graph replay.

Ready is published on `PIPE_FIX` after the full-GM copy. A free credit is published on `PIPE_MTE3` only after the original route helper has completed its input reads and BF16 stores. Shared output stores execute before the AIV consumer loop; the unchanged barrier 4 therefore joins down, routed and shared publication. Barrier 5 and both outer phases remain. No ring-buffer storage is introduced: full down GM is retained, and peak resident UB remains 112,352 bytes, including shared-store allocations whose C++ scopes have ended.

## Build and gates

The public interface retains 12 bridge and 20 body tensors, the original shape/alias/owner validators and the original system-workspace behavior. The new Torch namespace is `tilert_ffn_down_overlap`, C ABI `ffn_down_overlap_launch`, and library `ffn_down_overlap.so`.

Frozen build-v2, library SHA256 `1d4220181c3a0235109d663aa348c4b435d9e5f91bdb8e82e0201e82c8214822`, compiled and linked with CANN 9.2.0 and co-loaded with the parent under Torch/TorchNPU 2.12 without initializing an NPU. Nine targeted CPU contract tests passed on both x86 and AArch64. All four ND/NZ GU/down host-tiling checks passed. Its 71 source identities and six artifact identities were verified after transfer to a3-194. [Build evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-194/ffn-down-overlap-build-v2/handoff.json) includes the exact SDK/documentation excerpts and hashes.

The M12/24 A3 device gate passed eight candidate and 16 reference changing-input replays. All 480 retained-boundary comparisons were byte-exact against the separate native stages and the official numerical chain. See the [device report](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-overlap212-m12-v1-a3-194/summary.json). Remaining geometries are not established by this gate.

The [same-address paired measurement](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-overlap-timing-v1-a3-194/summary.json) rejects build-v2 for performance. At M12/24 with 65 active experts and 16 weight banks, candidate/parent medians are 336.3173/236.9028 us and 336.0761/234.6792 us for the two owner arenas. Paired savings are -99.411368 and -101.369270 us, with zero wins in 60 pairs. All 19,200 raw-byte checks and 360 event-tail checks pass. The retained composed parent remains selected.

The new schedule increases per-route elementwise helper invocations from four to 48. Activation-scale DMA count changes from `2*sum(min(expert_count, blocks))` to `48*active_experts`: at M12 with 96 routes and 65 active experts, that is 192 versus 3,120 small copies. Expert-local consumption also replaces the original balanced route ownership. These costs can outweigh the overlap; retained INT32 input and BF16 output byte counts alone cannot predict the outcome.

```bash
python3 build.py /workspace/tests/glm52/whole_layer/ffn_composed/build-v2 \
  /workspace/tests/glm52/whole_layer/ffn_down_overlap/build-v2
```

CPU tests enumerate both-consumer credit interleavings, counter draining, zero-expert transitions and compact high/low address ownership. Source checks require unchanged arithmetic, public validators and phase joins. The device gate must compare all 30 boundaries for changing inputs/routes at M=1/2/12/32 and 16/24 groups, including hot-to-mixed transitions. Performance requires same-address pairing against the frozen composed parent; smaller launch count or local overlap alone does not establish a gain.
