# Decoder graph index-cache difference

The original eager arm and vendor graph agree at C8 M1 with an initial residual, replay 2. The native graph differs at `layer6.index_cache[7,125,0,46]`: original code 20, native code 21. The physical slot is 1021, exactly the current token's indexer destination; its logical position is 8189 and the actual key length is 8190. Both FP16 cache scales are 0.0222625732421875. The other 25 recorded boundaries, including input normalization, ordered top-k, both outputs and retained residuals, are byte-equal.

[coordinate-audit.json](coordinate-audit.json) records the coordinate mapping, exact comparisons and source hashes. Its CPU RNG reconstruction uses Torch 2.9.1 and does not reproduce every Torch 2.12 producer byte; it is auxiliary. The device diagnostic instead compares the recorded live-input hashes directly.

The saved failure has no WK, LayerNorm, RoPE, Hadamard or dynamic-quantization intermediates. A correct destination and equal scale do not identify which arithmetic boundary first differs. [The graph-index diagnostic](../vendor-hook-decoder212-graph-index-v3/README.md) reads retained producers before graph teardown and requires the original full failure to recur.

The original result is preserved in [failure-summary.json](../vendor-hook-decoder212-graph-int8-wide-v1-a3-194/failure-summary.json). The 42.6 MB tensor dump remains outside the repository; its path and SHA-256 are in the coordinate audit.
