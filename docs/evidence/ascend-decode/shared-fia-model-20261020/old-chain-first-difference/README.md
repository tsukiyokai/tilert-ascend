# Historical chain-w8 first difference

The saved record locates the first difference before host commit/delivery. It does not establish whether the device target prediction differed, a readback buffer changed, or the target consumed different cross-round state. Equality of token6500 with the initial output token is not causal evidence.

Three mechanisms require distinct evidence:

| Mechanism | Required evidence |
| --- | --- |
| Target arithmetic on the intended causal prefix | Actual input/metadata and matching valid KV, plus immediate target normalized hidden, logits/top2 and scores of2524/6500. |
| Prediction/commit/delivery buffer error | Device argmax before and after conditioned MTP, then accepted-span and output-offset mapping. |
| Rejected-tail or cross-round state | Per-cache valid-prefix identity, write coverage, live positions/slots/sequence lengths and query0 selected IDs; a matching-prefix control if these differ. |

## Observed boundary

The local file `/tmp/lookup-offline/chain-w8-rank0.json` matches witness SHA `3aa6cce191de24a06a1131f45db3c93f30716ffe5024ad0c23ca4f8c7524d671`. Its original server reference is `a3-84:trt-asc:/workspace/experiments/lookup-20261019/chain-w8/rank-0.json`. It declares FP32 head, compiled target Graph, super-kernel, draft Graph and conditioned Graph. [result.json](result.json) retains the first two raw round records for requests3/4/5 and the scalar consistency receipt.

| Round (one-based) | Pending | Main before / written / retained | Recorded predictions | Accepted output offsets |
| --- | ---: | --- | --- | --- |
| 1 | 6500 | 8192 /8200 /8195 | `[382,262,576,31989,20047,13865,2240,6539]` | offsets1–3=`[382,262,576]` |
| 2 | 576 | 8195 /8203 /8196 | `[6500,323,374,25902,25902,393,315,576]` | offset4=`6500`, expected2524 |

Round2 candidates are `[576,1887,6500,10528,279,1156,3950,13]`. The first draft1887 mismatches prediction6500, so `matched_drafts=0`, `natural_accepted=1`; output6500 is the recorded query0 bonus, not an accepted later draft. Query0 is physical position8195 and should see valid keys through8195. Before this round, the valid new main-cache rows are8192–8194 for `[6500,382,262]`; pending576 has not yet been committed to KV. Round1 also physically wrote rejected rows8195–8199. Correct round2 execution overwrites8195–8202 and excludes future rows from query0.

All227 rounds of each measured request satisfy scalar candidate matching, acceptance, position and delivery recurrence. Callback tokens equal returned tokens. All six complete histories (three warmup plus three measured) are identical after removing clocks. This excludes a mismatch introduced only after the recorded predictions; it does not validate actual per-layer KV contents or device replay buffers.

The exact historical `mtp_round.py` is available at Git commit `168934ff8841b684e5d730bbbfef19200688bab2`, file SHA `ec1e321e758dd4ffb228832159a13289c01de7c2ff17ccee261ef63c76d9a16e`, matching the record: verify at119, conditioned MTP at121–123, host readback at124–126, resolve at127. Thus the JSON predictions were collected after conditioned execution. No pre-conditioned device argmax/logit snapshot exists in this record. The current server-tree copies of `mtp_round.py`, `cann_model.py` and the benchmark differ from their recorded hashes; they are not substituted for historical source.

## Minimum additional trace

Only the first two rounds are needed: (1) live target input IDs, positions, slots, SFA/indexer lengths and Graph replay identity; (2) target token_ids plus query0 normalized hidden, top2 and scores of2524/6500 immediately after verify; (3) the same token_ids after conditioned MTP to detect mutation; (4) hashes of valid new rows8192–8194 in each main/indexer cache and query0 selected IDs with causal threshold8196; (5) accepted tokens and output offsets, already present here. Raw target snapshots must be independent copies. These distinguish a late buffer change from an already divergent target result; separating arithmetic from incorrect prefix state additionally needs a verified matching-prefix control. No device work or product changes were performed.
