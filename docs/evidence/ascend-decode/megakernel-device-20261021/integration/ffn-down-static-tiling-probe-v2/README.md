# Down K128 single-buffer host probe

The installed CANN 9.2 compiler emits an M32/N1024/K128 plan with one L0B buffer for both ND and NZ INT4 weights. Six emitted constant plans cover the accepted N512 and N1024 controls and the K128 proposal. All 92 checked geometry/resource fields in the four control plans match their actual host-runtime records. [Compiler audit](summary.json), [probe source](probe.cpp), [commands](command.json).

Relative to the accepted M32/N1024/K64 record, the K128 plan changes six fields: `baseK` 64→128, `depthA1` 4→2, `depthB1` 12→6, `stepKa` 2→1, `stepKb` 2→1 and `dbL0B` 2→1. L1 remains 397,312 bytes, L0B uses 65,536 bytes, and L0C remains 131,072 bytes. The compile-time planner disables vector dequantization because this Down path produces INT32 and does not call `SetQuantVector`.

The [typed host probe](runtime_record.cpp) obtains the accepted 200-byte `c207230a` record through the installed host API and assigns only these six public `TCubeTiling` fields. ND and NZ yield the same complete proposed record, SHA-256 `c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999`. The [record audit](runtime-record-summary.json) reconstructs every byte from the compiler-derived values and the compiler-reported field offsets. All other bytes remain unchanged. In particular, runtime `transLength` stays zero; the compile-time API emits 131,072 for both accepted controls as well as the proposal.

The installed [L0 pool implementation](../ffn-down-static-tiling-probe-v1/installed/impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h) couples A/B ping-pong through `(dbL0A - 1) & (dbL0B - 1)`. Setting `dbL0B=1` therefore removes both operands' ping-pong overlap in this path. For `c<=16`, the proposed K tile would reduce Down MMAD calls from twelve to six per expert while retaining six output-tile copies. For `17<=c<=32`, the corresponding counts are twenty-four to twelve MMAD calls and twelve output copies. The latency effect depends on the lost overlap and requires a device comparison.

Both probes ran in CPU-only containers on a3-78 with no mapped NPU devices. These results establish host preparation and compiler-derived resource arithmetic. A complete candidate kernel, original-Decoder numerical comparison, hot-expert coverage and paired device timing remain required. [Runtime environment](runtime-environment.json), [installed source identities](../ffn-down-static-tiling-probe-v1/source-files.json).

The preceding [dynamic host API probe](../ffn-down-l0b-host-audit-v1/summary.json) returned the existing K64 record when K128 or single buffering was requested. The [first constant-plan probe](../ffn-down-static-tiling-probe-v1/fields.json) retained the default vector-dequantization reservation; this version removes that unused 49,152-byte reservation in the planner.

Recompute the audits from this directory with `python3 audit.py` and `python3 audit_runtime_record.py`.
