# N512 preparation for the local layer

The local layer uses the accepted `c90b3666` FFN host preparation and its
M64/N512/K128 Down tiling. The direct-Top-K numerical fixture uses `6a326292`;
the retained performance baseline uses `609c973e` with its Top-K copy. The
separate reference uses the same N512 FFN and the same prepared buffers.

The frozen source comparison permits this composition: all 49 embedded FFN
C++ files match the N256 parent. After namespace normalization, the N512
parent changes only host tiling; the device arithmetic and schedule match.
The device fixture checks all 200 Down tiling bytes after preparation and
after each reference or candidate call. Both local TP reductions are identity.

The [eager local-layer comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-n512-wide-v1-a3-194/summary.json)
passes M1/2/12/32 with eight changing inputs each: 3,488 raw-byte checks and
136 complete tiling checks. The [official ACLGraphWrapper comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-n512-graph-wide-v1-a3-194/summary.json)
passes four captures and 32 changing-input replays with the same check counts.
The candidate makes no persistent top-k copy; unused internal indices and
inactive persistent rows stay unchanged. Both gates use synthetic parameters
in the original Decoder6/7 classes and identity TP reductions.

The standalone N512 FFN has passed original-Decoder numerical and paired
performance comparisons.
The first timing fixture compares `609c973e` plus its top-k copy with
`6a326292` direct output, using N512 in both. Each arm receives identical
config/address bytes and tensors. Two separately retained scratch arenas share
the original persistent index owner, caches and two weight sets. Graphs repeat
16 indexed/shared layer pairs; 30 replays form one event sample, with 30
alternating sample pairs per arena. This one-bank workload tests the top-k
change under repeated weights; it does not establish whole-model HBM latency.

That [paired comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-pair-timing-v1-a3-194/summary.json)
rejects direct Top-K output: paired median regressions are 14.26 and 9.09 us
per two-layer pair, with no candidate wins in 60 pairs. The original `609c973e`
layer with N512 remains the timing baseline.

The `native-stages` timing mode uses the recorded Torch212 `norm_quant_out`
library and the accepted attention/FFN entries. These three launches write the
same owners as the whole layer, with no norm-output copies. Before capture,
eight changing-input comparisons check this native norm path against the
original norm module and dynamic quantization. The graph comparison then
checks 109 boundaries and times three native stages against one `609c973e`
launch per layer. The independent original-norm oracle retains its output
copies outside the timed graphs.

The [M12 native-stage comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-native-timing-v3-a3-194/summary.json)
passes all 5,232 eager/native/captured boundary checks, 600 event-tail checks
and 404 complete tiling checks. Whole-layer/three-stage medians are
986.78/1006.82 and 999.74/1020.19 us per two-layer pair. Paired median savings
are 20.22 and 21.21 us, with 60 wins in 60 alternating pairs. Both arms retain
the same Top-K copy. These are local, single-bank measurements with identity
TP reductions and two synthetic Decoder weight sets.
