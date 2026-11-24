# N512 preparation for the local layer

The persistent-top-k layer uses the accepted `c90b3666` FFN host preparation
and its M64/N512/K128 Down tiling. The layer library stays at `6a326292`.
The separate reference uses the same N512 FFN and the same prepared buffers.

The frozen source comparison permits this composition: all 49 embedded FFN
C++ files match the N256 parent. After namespace normalization, the N512
parent changes only host tiling; the device arithmetic and schedule match.
The device fixture checks all 200 Down tiling bytes after preparation and
after each reference or candidate call. Both local TP reductions are identity.

The standalone N512 FFN has passed original-Decoder numerical and paired
performance comparisons. Local-layer numerical, graph and timing acceptance
are required for this composition.
