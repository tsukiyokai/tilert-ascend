# FFN norm publication

This candidate removes the final `SyncAll<false>()` from the private norm helper in static-GU FFN library `381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2`. Its caller immediately performs `PIPE_ALL`, DDR publication, a mixed-core `SyncAll`, and a TPipe reset before Gate and shared quantization consume the normalized values. No consumer or allocation occurs between the two original joins.

The per-row norm operations, local completion and TPipe resets remain unchanged. This helper is private to the composed FFN; its caller supplies the cross-core publication. The five routed-body joins, static GU tiling, shared key30013 arithmetic, terminal completion, 33 tensors and 408-byte descriptor remain unchanged.

Acceptance requires the original DecoderLayer norm/MLP oracle at M1/2/12/32 with both layer weight sets and immediate graph consumers, followed by same-address timing against the static-GU parent. No latency result is established by removing the source-level join alone.
