# Original Decoder graph fixture

The fixture captures the original DecoderLayer 6 then 7 through the official
`ACLGraphWrapper`. Both arms retain the official FFN, its normal GMM disposal,
and the corrected storage-preserving INT32 NZ aliases. One arm installs native
attention `5f8868bf`; the shared-indexer library is `64108f6f`. No TP16 numerical
reduction is performed: each attention partial is recorded before substituting
an independently changing BF16 post-AR1 input, and each FFN partial is recorded
at the second reduction boundary. Layer 7 consumes layer 6's local output and
its separately retained residual.

The actual `DeepseekV2DecoderLayer.forward` resides in the pinned
`deepseek_v2.py`, SHA `96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d`.
The manifest's `original_forward_sha256` instead identifies the retained
fixture helper `decoder_forward.py`, SHA
`c802ae9371103bb75c1f8f5df39e3c9b2e49d77fd3a86ef23ad5d58781fcb207`.

[The CPU report](cpu-report.json) records four contracts, four CLI controls,
784 overlay hashes, 7,953 runtime hashes and both native libraries loaded into
the actual Torch 2.12 / Torch-NPU 2.12 image. The container had no davinci device
mapping; NPU initialization remained false. These checks do not establish graph
execution on the device. [The frozen manifest](manifest.json) and
[handoff](handoff.json) identify the complete source snapshot and proposed run.

Each width has separate captures for an absent initial residual and a supplied
BF16 residual. GPU sequence lengths, positions, independent main/indexer page
maps, three slot producers and hidden inputs change between replays. The host
sequence limit stays at the safe upper bound 8192; the page tables retain 64
columns. Persistent INT64 main slots are refreshed before replay.

Capture retains strong device owners for all observed boundaries and both
returned tensors. The unchanged `TPCuts.reduce` also records a CPU Python stack
with `traceback.extract_stack(limit=10)` during capture; it does not read tensor
values to CPU. Tensor snapshots are taken after explicit execution. The initial
capture result is not treated as executed data.
Every explicit replay follows fresh cache restoration and output poisoning,
then compares both graphs with an eager original-Decoder oracle. Checks include
the actual inner capture stream, native reserve ownership, unchanged Python
call counters, ordered top-k, full caches, both reduction cuts and residuals.
Output and pre-AR1 hashes must change across inputs and contain nonzero values.
Graphs are reset before their workspace owners are released.
