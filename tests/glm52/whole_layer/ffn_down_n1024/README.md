# Down M32/N1024/K64

This private complete-FFN candidate changes the c90 N512 parent's Down host tiler to `SetFixSplit(32, 1024, 64)`. Its namespace is `tilert_ffn_down_n1024` and its C ABI is `ffn_down_n1024_launch`. All device arithmetic, dynamic expert counts, GU tiling, original shared activation, five body joins, terminal publication, 33 scratch tensors and the 408-byte descriptor remain unchanged.

The installed CANN 9.2 host tiler accepts both ND and NZ weights and emits the same 200-byte record: SHA-256 `c207230aafb11e2c2f825cd9a2ea70147ee7733046b8b2957593c9c2bc3e8009`. The host preparation rejects every other record. Logical dimensions remain M64/N6144/K128. Each expert still executes `SetOrgShape(2*c, 6144, 128)` and `SetSingleShape(2*c, 6144, 128)`.

For `c <= 16`, the normal scheduler produces six C tiles with two K64 accumulations per tile. The parent produces twelve C tiles with one K128 accumulation each. Thus the candidate removes six `GetTensorC`/Fixpipe calls per expert while retaining twelve MMAD calls and the same INT32 output bytes. For `17 <= c <= 32`, a second M tile restores twelve output copies and raises MMAD calls to twenty-four. Small-M `PIPE_M` checks remain. L1/L0 buffer reuse also changes; reduced output-call count does not establish a speedup or reduced HBM traffic. [Installed-source audit](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-n1024-source-audit/audit.md).

The resource limits are L0A 2,048 B, L0B 65,536 B including double buffering, L0C 131,072 B, and L1 397,312 B. Routed-body UB remains 116,480 B; the complete FFN's norm phase remains the 195,584 B peak. No GM or UB allocation is added.

Build from the immutable c90 tree with CANN 9.2 and the same Torch 2.12 toolchain:

```bash
python3 build.py /workspace/tests/glm52/whole_layer/ffn_down_n512/build-v2 /workspace/tests/glm52/whole_layer/ffn_down_n1024/build-v2
FFN_DOWN_N1024_PARENT=/workspace/tests/glm52/whole_layer/ffn_down_n512/build-v2 python3 -m unittest discover -s . -p test_prepare.py -v
```

`build.py` compiles sequentially, checks both parent and candidate host tilers, co-loads both real Torch libraries without initializing an NPU, and retains source hashes, commands and logs. `expected-tilings.json` contains both complete Down records for device-prepare checks. Device acceptance must compare the original Decoder norm/MLP and immediate consumers for M1/2/12/32, 16/24 groups, including hot experts with `c=32`. The prepared body slot 7 must match its own 200-byte record before capture and after replay.

Paired timing must use c90 as control, both owner allocations, 30 event samples per arm and the existing fresh-write/raw-byte checks. The other 32 scratch addresses and weights are common; body slot 7 stays private to each tiling owner. No device result is claimed by the host-only tests.

The frozen build-v2 library is `bbd52e425a1e94bb199e464e3bf6001c4420e8238afe993e8741e38af50b77ef`. All 99 sources and six artifacts were verified locally and on a3-194. Five source/layout/ABI contracts passed on x86 and ARM; actual Torch 2.12 co-loading preserved both schemas without initializing NPU. The compiled host tilers reproduce the complete parent/candidate Down records and identical GU bytes. [Build evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-n1024-build-v2/summary.json).

The [original-Decoder gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-down-n1024-wide-v1-a3-194/summary.json) passes M1/2/12/32 at 24 groups, with eight changing inputs and both synthetic layer weight sets. All 704 byte checks, including 256 immediate-consumer checks, and 72 complete Down-tiling checks pass. This gate retains the original norm and MLP as its oracle.

The [hot-expert gate](hot/README.md) also passes M17/32 at both 16 and 24 groups. Each run checks 32 changing-input cases across two layer weight sets, with all tokens selecting the same eight experts. All 480 byte checks, including 128 immediate-consumer and 128 routing-metadata checks, and 36 complete tiling checks pass per run. The actual expert counts exercise both Down M32 tiles.

The [paired M12/24 comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-n1024-timing-v1-a3-194/summary.json) saves 7.933/6.371 us against N512 across two owner arenas, with all 60 pairs favoring N1024. Candidate/control medians are 201.439/209.362 and 199.486/205.869 us. The workload uses 65 active experts, 16 independent weight banks, 16 calls per graph and 30 replays per event. All 19,200 numerical, 360 event-tail, 696 active-table and 20 full-tiling checks pass. The 32 common scratch addresses and two private Down-tiling owners stay fixed. This result applies to M12; it does not cover the second M tile required by hot experts at M17/32.

The [609 local-layer composition](../layer_down_n1024/README.md) retains the original TopK copy and passes both eager and official graph gates at M1/2/12/32. Each passes 3,488 byte comparisons and 136 full-tiling checks. Its [matched whole-layer timing](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/layer-down-n1024-timing-v1-a3-194/summary.json) saves 1.435/2.487 us per indexed/shared two-layer pair against N512, with 51 of 60 pairs faster. Candidate/control medians are 991.699/993.182 and 996.440/998.994 us. Both arms use the same 609 binary and production addresses; only the two immutable Down-tiling pointers differ. All 6,660 numerical, 600 event-tail, 844 complete-tiling and 388 complete-record comparisons pass. This one-bank, identity-TP whole-layer effect is smaller than the 16-bank standalone FFN effect above.
