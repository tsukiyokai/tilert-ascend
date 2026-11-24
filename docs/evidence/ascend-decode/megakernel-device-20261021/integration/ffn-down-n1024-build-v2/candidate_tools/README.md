# Down M32/N1024/K64

This private complete-FFN candidate changes the c90 N512 parent's Down host tiler to `SetFixSplit(32, 1024, 64)`. Its namespace is `tilert_ffn_down_n1024` and its C ABI is `ffn_down_n1024_launch`. All device arithmetic, dynamic expert counts, GU tiling, original shared activation, five body joins, terminal publication, 33 scratch tensors and the 408-byte descriptor remain unchanged.

The installed CANN 9.2 host tiler accepts both ND and NZ weights and emits the same 200-byte record: SHA-256 `c207230aafb11e2c2f825cd9a2ea70147ee7733046b8b2957593c9c2bc3e8009`. The host preparation rejects every other record. Logical dimensions remain M64/N6144/K128. Each expert still executes `SetOrgShape(2*c, 6144, 128)` and `SetSingleShape(2*c, 6144, 128)`.

For `c <= 16`, the normal scheduler produces six C tiles with two K64 accumulations per tile. The parent produces twelve C tiles with one K128 accumulation each. Thus the candidate removes six `GetTensorC`/Fixpipe calls per expert while retaining twelve MMAD calls and the same INT32 output bytes. For `17 <= c <= 32`, a second M tile restores twelve output copies and raises MMAD calls to twenty-four. Small-M `PIPE_M` checks remain. L1/L0 buffer reuse also changes; reduced output-call count does not establish a speedup or reduced HBM traffic. [Installed-source audit](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-n1024-source-audit/audit.md).

The resource limits are L0A 2,048 B, L0B 65,536 B including double buffering, L0C 131,072 B, and L1 397,312 B. Routed-body UB remains 116,480 B; the complete FFN's norm phase remains the 195,584 B peak. No GM or UB allocation is added.

Build from the immutable c90 tree with CANN 9.2 and the same Torch 2.12 toolchain:

```bash
python3 build.py /workspace/tests/glm52/whole_layer/ffn_down_n512/build-v2 /workspace/tests/glm52/whole_layer/ffn_down_n1024/build-v1
FFN_DOWN_N1024_PARENT=/workspace/tests/glm52/whole_layer/ffn_down_n512/build-v2 python3 -m unittest discover -s . -p test_prepare.py -v
```

`build.py` compiles sequentially, checks both parent and candidate host tilers, co-loads both real Torch libraries without initializing an NPU, and retains source hashes, commands and logs. `expected-tilings.json` contains both complete Down records for device-prepare checks. Device acceptance must compare the original Decoder norm/MLP and immediate consumers for M1/2/12/32, 16/24 groups, including hot experts with `c=32`. The prepared body slot 7 must match its own 200-byte record before capture and after replay.

Paired timing must use c90 as control, both owner allocations, 30 event samples per arm and the existing fresh-write/raw-byte checks. The other 32 scratch addresses and weights are common; body slot 7 stays private to each tiling owner. No device result is claimed by the host-only tests.
