# Original Decoder graph indexer diagnostic

The frozen overlay observes the C8 M1 graph with a provided residual for replays 0–2. It retains the original `decoder_graph.py`, real DecoderLayer 6/7 calls, original FFN, descriptor aliases and accepted attention libraries. [handoff.json](handoff.json) records the exact command, deployed paths, image and source identities. `decoder_forward.py` is the fixture helper; the actual Decoder class belongs to the separately hashed `deepseek_v2.py`.

The vendor indexer keeps two device clones before in-place RoPE. Other taps retain references. Native taps retain the indexed workspace from capture; the shared successor executes normally and does not replace that reference. The diagnostic reads only after the original comparison has failed and its complete fixture has been saved, while graph objects and workspace owners remain live. It adds no GMM synchronization or operator replacement.

Intermediate evidence is accepted only when all of these checks pass:

- The complete plan, both sets of 26 output/cache boundaries, case identity and every recorded live-input witness reproduce the original failure.
- Both tapped hidden tensors match their own input-norm boundary, and both slot vectors match the current plan.
- Each arm's retained INT8 key and FP16 scale exactly match its own current cache destination.

The report orders the key path as hidden → WK → LayerNorm → RoPE → Hadamard → dynamic quantization, with the query path and original weights also checked. CPU FP64 WK and FP32 RoPE formulas are auxiliary. The diagnostic handles only the exact original numerical RuntimeError after these acceptance checks; it records `original_graph_pass=false` and permits the helper ranks to finish. Other failures propagate.

[cpu-report.json](cpu-report.json) records 10 passing contracts in the actual Torch 2.12 / torch_npu 2.12 image, verification of 793 overlay files and 7,953 runtime files, four attention schemas and four CLI controls. No davinci devices were mapped, and NPU initialization remained false. These checks do not establish device arithmetic or graph correctness.

The full frozen overlay is `/tmp/vendor-hook-decoder212-graph-index-v3` locally and the matching task directory on a3-194. The repository retains its source and manifests; runtime packages, native libraries and large failure tensors remain in their separately identified locations. Earlier v1/v2 overlays remain unchanged.
