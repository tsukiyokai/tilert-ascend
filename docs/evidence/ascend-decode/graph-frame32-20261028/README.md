# Width32 CPU frame evidence

The real 78-layer HF model runs 96 teacher steps as cold A/B/A sequences of
32 rows. The FLOAT fixture uses hidden/latent rank 16, vocabulary 32,
native MLA, FP32 head, and zero attention/FFN outputs. It has changing
routes and nonzero KV/RoPE/C8 rows. Cold32 does not exercise 8K or top2048.

Each frame has 628 fields: 79 hidden states, 78 diagnostic logits, 78 index
selections, 78 routes (three dense `None` entries), 312 KV fields, final
logits/hidden, and token IDs. Short index rows use explicit lengths and
`-1` padding. All 32 rows and source/input/arithmetic/checkpoint identities
are required. The unmodified strict comparator accepts 25,216 checks per
frame; 32 reset rows pass exactly. All 19 rejection cases pass, alongside
99 cache-length checks and 629 scalar oracles. The raw ledger contains
297,632 checks: 256 deliberately fail in numerical negative cases; zero
unexpected checks fail. All 55 experiment cases pass.

The current reusable receiver additionally rejects empty/non32 references and
bool/int/float identity aliases. The frozen codec admits an empty reference
with zero checks; its numerical receipt remains unchanged. The separate
`receiver-readback.json` checks the current receiver against all 96 saved HF
rows, 60,288 field rows, 629 recomputed scalar oracles, and 15 rejection cases.
It performs no HF construction or forward and copies decoded row storage.

The packet/default observer sources and reports preserve the DCE trap:
`has_side_effect(.default)` does not protect an OverloadPacket before AOT.
Explicit `.default` survives the tested CPU Dynamo/AOT paths. The CPU save
implementation verifies tracing only. Opaque attention/MoE consumers need
separate capture-time observation of actual choices. NPU producer execution,
callback lifetime, static-kernel/SuperKernel compatibility, deployed framework
version, actual graph inputs, and the bridge to the performance graph remain
unverified. `precision_verified=false`; no CANN precision gate is passed.

`manifest.json` records raw and compressed hashes. The archive includes all
96 HF `.pt` files, three frames, checkpoint payloads, bound reference/HF
sources, the complete report, and losslessly compressed strict records.
`audit.json` records full readback. [Reproduce or audit](../../../../experiments/graph-evidence/README.md)
with the preserved sources; the 96-step receipt belongs to the frozen code.
