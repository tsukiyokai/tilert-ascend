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

Writing the final owner directly removes one copy task and its active-prefix
traffic. The numerical gates compare two consecutive indexed/shared layers
against the separate parents with changing inputs, caches, lengths, pages and
positions. Both TP cuts remain identity.

[The CANN build](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/layer-topk-owner-build-v2/summary.json)
passes the no-undefined link, actual Torch212 library load and eight host
geometry cases without mapping or initializing an NPU. The library SHA is
`6a326292287a17254b48376400e1b852093ebfe10fbf228938336693402aa528`.
The [eager two-layer comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-topk-owner-wide-v2-a3-194/summary.json)
passes M1/2/12/32 with eight changing inputs each and 3,488 raw-byte checks.
The candidate makes zero top-k copy calls; the unused internal slice and
inactive persistent rows remain unchanged in every case. With the accepted
N512 FFN preparation, the [official graph comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-n512-graph-wide-v1-a3-194/summary.json)
also passes all four widths, 32 replays and 3,488 raw-byte checks.

The [paired M12 graph timing](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-pair-timing-v1-a3-194/summary.json)
rejects this candidate. With N512 in both arms, two scratch arenas give
candidate/control medians of 1002.09/987.94 and 999.36/990.01 us per two-layer
pair. Paired median regressions are 14.26 and 9.09 us; all 60 pairs favor the
original layer with its copy. All 3,488 eager/captured boundary checks, 600
event-tail checks and 372 full tiling checks pass. The workload repeats two
synthetic weight sets; it does not measure whole-model TPOT.
