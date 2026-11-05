# Token-output tail join

This independent candidate removes two redundant tail synchronizations from token-output build-v2, library SHA256 `e92a9cbe4d11307a8ea8beae20b5cbe845be10f9aba5280cbc3f438b049ed041`. It removes body barrier 5 and the second adjacent outer `EndPhase`. The first outer `EndPhase`, body barriers 0 through 4, every local producer fence and all arithmetic remain unchanged. Active-expert and descriptor variants are not included.

## Synchronization contract

After barrier 4, each token/column owner produces route, routed, scaled and combined BF16 outputs, completing its local publication fences. The body then executes only `NoTrace::Begin`, `FinishFinal` and `Store`, which are empty, and returns. No consumer or buffer reuse occurs between token output publication and the first outer `EndPhase`. All body-local objects leave scope before that phase performs `PIPE_ALL`, DDR synchronization, mixed `SyncAll<false>` and `pipe.Reset()`. An early CUBE worker can wait there while a VECTOR worker finishes; reset cannot pass the mixed join.

The second outer `EndPhase` immediately follows the first with no intervening statement. It has no new producer, consumer or allocation to protect. Removing it does not change the first phase's publication or reset obligation. Profile builds are rejected: `StageTrace::FinishFinal` and `Store` have device effects, so this proof is specific to the uninstrumented `NoTrace` body.

The CANN 9.2 source audit checks that `TBuf` and its `TQueBind` base have no device-work destructor, and that mixed `SyncAllImpl` drains local pipelines before AIV MTE3/AIC FIX coordination. The candidate does not assume that leaving a `TBuf` scope reclaims its allocation. Peak UB remains 115,424 bytes; no GM or UB storage is added.

## Interface and validation

The library is `ffn_token_join.so`, with Torch namespace `tilert_ffn_token_join` and C ABI `ffn_token_join_launch`. Its 12 bridge and 20 body buffers, alias checks, tiling owners and 30 retained output boundaries are identical to the parent. The initial system-workspace selection and later ordinary `SetSysWorkspace` call retain their original behavior.

`test_contract.py` checks exact source restoration after the two deletions, unchanged numeric helpers and validators, compile-time rejection of profiling, the retained publication/reset sequence, arbitrary three-worker arrival orders and inherited token/column ownership. These are CPU contract checks; target-device numeric and timing results require a separate gate.

```bash
python3 build.py /workspace/tests/glm52/whole_layer/ffn_token_output/build-v2 \
  /workspace/tests/glm52/whole_layer/ffn_token_join/build-v1
```

The generated checker differs from the frozen parent's checker only in its candidate Torch namespace. Performance must compare against the token-output parent at identical full-scratch addresses with fresh-write checks, independently of the older composed baseline.
