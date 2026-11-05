# Persistent top-k destination

This candidate changes the accepted `609c973e` local layer so that its indexed
selector writes directly to the persistent indices used by later shared-indexer
layers. The current layer's sparse attention reads the same address. The existing
selector publication, mixed-core join, cache writes and FFN remain unchanged.

The existing `shared_topk` address slot is required by both entries. The owner is
an aligned INT32 ND `[M,1,2048]` view of the active prefix of the original
`[32,2048]` allocation. It has separate storage from every norm, attention and FFN
argument and remains alive across capture and replay. Host records stay at
664/928 bytes. The private library has distinct `layer_topk_*` C exports.

The indexed attention arena retains its original size and offsets. Its old
top-k slice becomes unused; the numerical comparison must observe the retained
destination and verify that the unused slice and inactive persistent rows remain
unchanged. The separate parent still selects into its arena and copies into the
persistent owner. The candidate requires no copy between layers.

The hypothesis is that writing the final owner directly removes one copy task
and its active-prefix traffic without changing either layer. Validate two
consecutive indexed/shared layers against the separate parents with changing
inputs, caches, lengths, pages and positions, then use actual ACLGraphWrapper
replays. Time matched graph workloads with the same owners and output consumers;
the removed copy is part of the candidate change. No performance result is
available yet. Both TP cuts remain identity.
