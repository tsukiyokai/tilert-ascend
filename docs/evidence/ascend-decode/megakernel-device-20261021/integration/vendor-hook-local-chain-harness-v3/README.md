# Local layer-chain fixture

The eager device gate passed M=1/2/12/32 with eight changing inputs per width: 32 cases and 3,488 raw-byte comparisons. See the [device audit](../vendor-hook-local-chain-wide-v1-a3-194/summary.json). Each case compares the separate input norm, attention and complete FFN parents with two local-layer launches using the actual synthetic Decoder6/7 weights and cache metadata.

Both TP reductions are identities. Layer6's attention partial feeds its FFN directly; layer7 receives layer6's combined output and separately retained residual. Layer6 copies its selected indices into the persistent owner consumed by shared-indexer layer7. This fixture calls the component entries, not `DecoderLayer.forward`.

[The CPU report](cpu/report.json) records ten contracts, four CLI checks, 808 overlay hashes, 7,953 runtime hashes and four loaded libraries in the real Torch/TorchNPU 2.12 environment. No NPU was mapped or initialized. [The archive record](archive.json) identifies the frozen sources and externally retained binaries; the complete manifest remains unchanged.
