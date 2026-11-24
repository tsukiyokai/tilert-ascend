# Token-output tail join

This independent candidate removes two redundant tail synchronizations from token-output build-v2, library SHA256 `e92a9cbe4d11307a8ea8beae20b5cbe845be10f9aba5280cbc3f438b049ed041`. It removes body barrier 5 and the second adjacent outer `EndPhase`. The first outer `EndPhase`, body barriers 0 through 4, every local producer fence and all arithmetic remain unchanged. Active-expert and descriptor variants are not included.

## Synchronization contract

After barrier 4, each token/column owner produces route, routed, scaled and combined BF16 outputs, completing its local publication fences. The body then executes only `NoTrace::Begin`, `FinishFinal` and `Store`, which are empty, and returns. No consumer or buffer reuse occurs between token output publication and the first outer `EndPhase`. All body-local objects leave scope before that phase performs `PIPE_ALL`, DDR synchronization, mixed `SyncAll<false>` and `pipe.Reset()`. An early CUBE worker can wait there while a VECTOR worker finishes; reset cannot pass the mixed join.

The second outer `EndPhase` immediately follows the first with no intervening statement. It has no new producer, consumer or allocation to protect. Removing it does not change the first phase's publication or reset obligation. Profile builds are rejected: `StageTrace::FinishFinal` and `Store` have device effects, so this proof is specific to the uninstrumented `NoTrace` body.

The CANN 9.2 source audit checks that `TBuf` and its `TQueBind` base have no device-work destructor, and that mixed `SyncAllImpl` drains local pipelines before AIV MTE3/AIC FIX coordination. The candidate does not assume that leaving a `TBuf` scope reclaims its allocation. Peak UB remains 115,424 bytes; no GM or UB storage is added.

## Interface and validation

The library is `ffn_token_join.so`, with Torch namespace `tilert_ffn_token_join` and C ABI `ffn_token_join_launch`. Its 12 bridge and 20 body buffers, alias checks, tiling owners and 30 retained output boundaries are identical to the parent. The initial system-workspace selection and later ordinary `SetSysWorkspace` call retain their original behavior.

`test_contract.py` checks exact source restoration after the two deletions, unchanged numeric helpers and validators, compile-time rejection of profiling, the retained publication/reset sequence, arbitrary three-worker arrival orders and inherited token/column ownership.

The frozen candidate `4417fd158e48a2da4d59e61fec48ebd0dd6e113022f13cbeb4f01d596eead757` passed the A3 device gate for M=1/2/12/32 and 16/24 core groups: 64 candidate and 128 reference changing-input replays, with all 3,840 retained boundary byte comparisons exact. The references are the separate native stages and the official numerical chain. See the [M12/24 report](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-join212-m12-v1-a3-194/summary.json) and [remaining geometries](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-join212-wide-v1-a3-194/summary.json).

At M12/24 with 65 active experts, 16 distinct weight banks and identical 32-buffer addresses within each pair, the candidate saved paired medians of 11.445 and 7.748 microseconds in the two owner layouts; all 60 pairs favored the candidate. Candidate/parent event medians were 215.294/226.728 and 221.440/229.205 microseconds. The gate passed 19,200 boundary checks and 360 event-tail checks. Each arm has 30 independent event samples per owner, with 30 graph replays inside each sample. These are full local FFN measurements against the token-output parent; attention, TP communication and model TPOT are outside this scope. See the [paired timing report](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-join-timing-v1-a3-194/summary.json).

```bash
python3 build.py /workspace/tests/glm52/whole_layer/ffn_token_output/build-v2 \
  /workspace/tests/glm52/whole_layer/ffn_token_join/build-v1
```

The generated checker differs from the frozen parent's checker only in its candidate Torch namespace. Performance must compare against the token-output parent at identical full-scratch addresses with fresh-write checks, independently of the older composed baseline.
