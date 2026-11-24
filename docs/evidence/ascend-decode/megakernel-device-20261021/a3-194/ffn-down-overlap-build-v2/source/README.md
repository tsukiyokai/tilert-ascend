# Down projection and route dequantization overlap

This independent candidate starts from composed library SHA256 `3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655`. It retains the full INT32 down-projection buffer, all 30 output boundaries, all six body barriers, both outer `EndPhase` calls and the original numerical helpers. It tests tile publication and consumption while CUBE continues the down projection. Active-expert tables, descriptors, token-owned output and tail-join changes are excluded.

## Data and numerical contract

The host checks the generated down tiling: base M/N/K=64/256/128, single-core M/N/K=64/6144/128. For an expert with c routes, the original matrix shape is M=2c, N=6144, K=128, with c at most 32. One M tile and 24 increasing N tiles cover the full result. Each AIC keeps its original increasing nonempty-expert ownership. It calls `Iterate` and `GetTensorC(c, 0, false)`, preserving the full-GM row stride and tile offsets used by the SDK's default `IterateAll` implementation.

The two associated AIVs consume columns `256*t + 128*subblock` through `+128` for every route of the same expert. They call the original `StoreRoutes` helper. High/low INT32 partial reconstruction, FP16 partial boundaries, correction and activation-scale order, BF16 route output, original top-8 slot placement and ordered combine remain unchanged. The helper performs no cross-element reduction, so its 128-column span changes ownership and memory operations without changing elementwise arithmetic. Original route outputs are written even for zero-probability slots; `CombineTokens` retains its zero-probability skip and slot-zero signed-zero seed. The outer routed BF16, scaled BF16 and shared-add BF16 boundaries are retained.

## Credit and lifetime contract

Mode2 uses ready flag8 and free flag9. CANN defines one AIC publication as releasing both AIV waits, and requires both AIV publications before releasing one AIC wait. These flags avoid Matmul's reserved0–7 and mixed `SyncAll`'s11–14. The SDK counter range is0–15; this protocol permits at most four unconsumed tiles.

The AIC sequence number continues across every owned nonempty expert. Before publishing tile q≥4 it consumes one free credit. After the last tile it drains `min(total_tiles, 4)` remaining credits. Each AIV waits once and acknowledges once per tile, so an AIC with E owned experts performs exactly24E ready publications and24E free waits. Empty experts contribute no operations. The two AIVs each perform24E ready waits and24E free publications. All counters are drained before barrier4 and before a later graph replay.

Ready is published on `PIPE_FIX` after the full-GM copy. A free credit is published on `PIPE_MTE3` only after the original route helper has completed its input reads and BF16 stores. Shared output stores execute before the AIV consumer loop; the unchanged barrier4 therefore joins down, routed and shared publication. Barrier5 and both outer phases remain. No ring-buffer storage is introduced: full down GM is retained, and peak resident UB remains112,352 bytes, including shared-store allocations whose C++ scopes have ended.

## Build and gates

The public interface retains12 bridge and20 body tensors, the original shape/alias/owner validators and the original system-workspace behavior. The new Torch namespace is `tilert_ffn_down_overlap`, C ABI `ffn_down_overlap_launch`, and library `ffn_down_overlap.so`.

```bash
python3 build.py /workspace/tests/glm52/whole_layer/ffn_composed/build-v2 \
  /workspace/tests/glm52/whole_layer/ffn_down_overlap/build-v1
```

CPU tests enumerate both-consumer credit interleavings, counter draining, zero-expert transitions and compact high/low address ownership. Source checks require unchanged arithmetic, public validators and phase joins. The device gate must compare all30 boundaries for changing inputs/routes at M=1/2/12/32 and16/24 groups, including hot-to-mixed transitions. Performance requires same-address pairing against the frozen composed parent; smaller launch count or local overlap alone does not establish a gain.
