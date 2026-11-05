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

Library `c39acf92728c8a3f09ee0b66988bed4cf0e312a15285a68c02d2594969a348af`
passes the [original FFN device comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-original-shared-terminal-wide-v1-a3-194/summary.json):
both layer weight sets, M1/2/12/32 and eight changing inputs each. All 448
arithmetic boundaries and 256 immediate-consumer checks match raw bytes.
Four original NZ weight banks keep their allocation and contents. The test
uses synthetic parameters and one physical A3 die, with explicit post-AR1
inputs and local pre-AR2 outputs.

The paired benchmark compares this library with original-shared library
`342d9653`, so both arms use the same corrected shared-activation arithmetic.
It retains two independently allocated arenas, all 33 shared addresses within
each pair, poisoned-output checks and the existing event window. Each arena is
prepared by the corrected Torch entry, which retains its tiling owners and
active-expert table for validation.

The [M12/24, 65-active-expert timing](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-original-shared-terminal-timing-v2-a3-194/summary.json)
uses 16 weight banks, 16 calls per graph, 30 replays per event and 30 alternating
pairs per arena. Candidate/parent medians are 211.408/219.639 us and
211.261/221.010 us. Median paired savings are 8.239/9.763 us, with all 60 pairs
faster. All 19,200 numerical, 360 event-tail and 696 active-table checks pass.
Integration into the complete Decoder graph remains pending.
