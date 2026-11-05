# Complete FFN with original shared quantization and token scheduling

This candidate applies the CANN 9.2 shared activation contract to the frozen
`ffn_token_terminal` library (`27635276...`). It retains token-owned MSD input
splitting, token-owned output merging, the active-expert table, and local kernel
exit. The five internal producer/consumer joins remain in place.

The shared activation uses scalar `amax / 127`, a guarded reciprocal, vector
multiplication, and the INT16 RINT / FP16 NONE / INT8 NONE cast sequence selected
by installed CANN tiling key 30013. The public descriptor stays at 408 bytes,
with 33 scratch tensors and 116,480 bytes of maximum resident UB.

`prepare.py` verifies every frozen parent source and records the activation
diff. `build.py` reuses the parent compilation commands and checks schema
compatibility without initializing the NPU. The original DecoderLayer norm and
MLP provide the numerical reference. Four immediate graph consumers check
publication of the combined, routed, shared, and retained-residual outputs.

Device correctness and paired latency measurements are pending.
