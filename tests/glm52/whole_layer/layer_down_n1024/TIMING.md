# N512 versus N1024 in two complete local layers

Both timed arms use the frozen 609 local-layer kernel and its current-stream
host shim. Each indexed/shared pair includes the same retained TopK copy.
Only the Down tiling changes: N512 uses the c90 preparation's complete 200-byte
record; N1024 uses bbd's record. The comparison does not include TP collectives.

The two layers share all input, weight, cache, norm, attention and FFN payload
addresses between arms. Each layer has one private read-only Down-tiling Tensor
per arm. All prepare owners remain alive until both graphs are reset. Extra
scratch from the N512 preparation is used only for an owning-Torch numerical
gate outside capture; its addresses do not enter either timed graph.

The frozen 609 C++ headers define the record check. All 98 Config fields and
116 Addresses fields are decoded. The 664-byte Config must be identical; the
928-byte Addresses record may differ only in `body.scratch[7]`, an eight-byte
field at offset 624. Raw comparison also checks padding. Every gate verifies
both complete tilings, so sharing a tiling or silently reverting to N512 fails
even when all numerical outputs remain equal.

Two retained scratch arenas use the same two synthetic Decoder weight sets and
persistent caches. Before capture, eight changing inputs compare original
input norm plus DynamicQuant against the existing native norm output API;
the native reference writes directly to the fixed owners. Separate c90
`forward_out` calls validate its original prepare objects and all retained FFN
boundaries. Each graph contains 16 identical two-layer pairs. Input, metadata
and current-slot cache updates are idempotent within one timed graph; the
second layer reads the first layer's output and residual, while the first
layer keeps independent external inputs.

Each arena records 30 alternating pairs of device-event samples, with 30 graph
replays per event. Cache restoration, fresh-output poison, warmups, CPU
snapshots and tiling checks remain outside events. Eight post-capture inputs
compare all 109 boundaries. Every sample checks output, residual, both active
tables and retained TopK; after timing, both arms again compare all 109
boundaries to the native stages. This is one weight bank with two layer weight
sets, not model-wide HBM traffic or TP16 latency.

CPU and device results are recorded with the frozen bundle. No performance
result is inferred from standalone FFN timings.
