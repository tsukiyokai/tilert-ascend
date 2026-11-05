# Static GU with expert-ready activation

This composition applies the validated expert-ready schedule to static-GU FFN library `381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2`. Each AIC publishes a complete expert GU product to its two AIVs with mode2 ready flag 8. Both consumers acknowledge through flag 9; at most four publications remain outstanding, and all credits drain before the original second body join. Expert counts below the AIC-group count retain the globally balanced activation path.

The static GU tiling, five body joins, shared key30013 arithmetic, terminal completion, 33 public tensors, 408-byte descriptor and 116480-byte UB requirement are retained. Mathematical helpers and the Down/output phases are unchanged. No extra GM or UB storage is allocated.

Acceptance requires the original DecoderLayer norm/MLP comparison at M1/2/12/32 with both weight sets, poisoned immediate graph consumers, and a same-address timing comparison with the static-GU parent. The inherited checker retains historical shared arithmetic and is not this composition's acceptance oracle. The separate static-GU and expert-ready timing results do not establish the combined gain.
