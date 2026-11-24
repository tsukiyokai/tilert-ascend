# Layer-resident teacher verification

The runner preserves actual TP16 reductions while keeping only one decoder
layer resident. For a fixed teacher sequence, each layer consumes its own
prefix KV and the preceding layer's hidden/residual rows; no later-layer
output feeds an earlier layer. This permits layer-major execution without
holding the complete model on the device.

Shared index selections add a state dependency. A producer supplies the
consumer layers' complete query selection. With one M32 chunk, a single
mask owner survives layer replacement. Two M16 chunks would overwrite that
mask before their consumers execute; the runner exposes only M1 and M32.
The [original CPU witness](result.json) also identifies two omitted runtime
initialization fields. Its source snapshots describe the pre-change runner.

The [implemented runner and current results](m32/README.md) cover actual
M32/SFA execution through eight layers, with dense, MoE and C8 producer and
consumer paths. Peak Torch allocation is899MiB/die under a2GiB quota.
Shared-FIA8 remains the next control, followed by78 layers and the M32
FP32 head against32 frozen successor labels. A complete same-width SFA run
is the separating control if that teacher check fails.

This checks numerical execution and state ownership. It does not measure
Graph execution, full-model residency or TPOT.
