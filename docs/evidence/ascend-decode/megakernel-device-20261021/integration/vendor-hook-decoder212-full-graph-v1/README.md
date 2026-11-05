# Native attention and complete FFN in the original Decoder graph

The actual Decoder layers 6 and 7 passed the official ACLGraphWrapper comparison for M=1/2/12/32, both absent and provided initial residuals, and eight changing inputs: 16 captures, 128 explicit replays and 3,264 raw-byte checks. All 16 Gloo participants completed; only rank 0 used the single NPU. See the [device acceptance summary](../vendor-hook-decoder212-native-full-graph-wide-v1-a3-194/summary.json).

The native arm uses indexed attention `8deff750`, shared-indexer attention `64108f6f`, and complete FFN `381febf9`. The reference retains the original attention and FFN. Both arms execute the original Decoder class (`deepseek_v2.py`, SHA-256 `96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d`). The helper `decoder_forward.py` and the FFN adapter are separate files, identified in the [manifest](manifest.json).

The FFN adapter preserves the original norm and MLP module call sites. Its complete native FFN runs at the post-attention norm call; the following MLP consumes the same normalized owner once, resolves the original runner once, and calls its original final-reduction helper once. Capture reserves belong to one layer, device, stream and exact width. Graph reset precedes pool release. Replay does not re-enter Python. The four routed NZ banks retain their addresses and complete physical bytes.

The first TP boundary receives an explicit BF16 post-reduction fixture input; the second exposes the local FFN partial. This result validates the original host graph and local native bodies, without executing a TP16 reduction or measuring model TPOT. A local identity chain, where the attention partial directly feeds the FFN, is a separate experiment.

The [Torch212 CPU report](cpu-report.json) records 18 contracts, actual host helper imports, native schema co-loading and CLI controls with no mapped or initialized NPU. Sources and launcher inputs are frozen by [frozen-inputs.sha256](frozen-inputs.sha256); binary identities and external artifact paths are in [binary-artifacts.json](binary-artifacts.json).
