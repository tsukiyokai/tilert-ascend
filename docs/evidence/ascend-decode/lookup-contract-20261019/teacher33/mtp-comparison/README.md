# Teacher33 candidate replay

This CPU replay uses the trusted 8192-token prompt and 33 output IDs, including the prefill output. It reuses the 30 audited MTP4 windows and the actual indexed lookup implementation with suffix lengths 1–8. It does not open the old 1024-token trajectory or execute a model.

## Contract

Let `g_i` be the pending output. Main KV length is `8192+i`, the main query starts there, and MTP seed row `p=8191+i` pairs its parent hidden with `g_i`. The pending token was already emitted. Lookup context includes `prompt + g_0..g_i`; longest suffix wins, with the latest occurrence at that length. Short proposals are padded with `g_i`.

The proposal is fixed before successor labels are accessed. If `m` consecutive drafts match, the natural new output count is `m+1`: matching drafts followed by a correction, or a bonus when all drafts match. Commit IDs start at `g_(i+1)`, and the remaining output budget clips the count. Physical verification retains its fixed width.

At `i=0`, MTP candidates `[6500,382,262,9897]` commit `[382,262,576]`, not the already emitted 6500. At `i=26`, four matching candidates `[323,3796,198,262]` commit `[3796,198,262,1105]`; the bonus 1105 is available in teacher33. These follow `mtp_state.py:185` and `mtp_round.py:100`; lookup padding follows `mtp_round.py:147`, and suffix selection follows `lookup_draft.py:23` in the saved source.

## Results

| Frozen output budget after prefill | Method | Rounds | Target verify rows |
| --- | --- | ---: | ---: |
| 30 new outputs, complete common prefix | MTP4, original or sharded | 14 | 56 |
| 30 new outputs, complete common prefix | Lookup8 | 26 | 208 |
| 32 new outputs | MTP4, strict interval | 15–16 | 60–64 |
| 32 new outputs | Lookup8, complete | 28 | 224 |

MTP visits pending offsets `0,3,4,5,8,9,11,13,16,19,20,21,23,26`, then stops at missing window 30 with two outputs remaining. Window 31 is also absent. No missing candidate is synthesized. The original and sharded candidates agree exactly at every visited offset; their differences at offsets 2, 12 and 17 are not visited.

The common-prefix mean complete MTP round cost must be below `26/14 = 1.857143` times the lookup round cost to win. For the full window, the corresponding threshold is 1.75 for the worse tail bound and 1.866667 for the better bound. These are cost budgets, not measured savings. MTP additionally executes one initial seed, two chain rows and four conditioned rows per round: `6R+1` draft rows. Target verify rows alone omit this cost.

## Evidence boundary

The replay is exact for the saved candidate IDs under a teacher oracle. The MTP records were conditioned with M1 teacher states, so they do not prove that a live M4 target and conditioned MTP pass produce the same subsequent candidates. Wrong-prefix target predictions are unavailable and are never filled with teacher labels. One short window does not establish prompt-set acceptance, latency or a runtime choice rule. The prior audited candidate JSON is reused; its large original device tensors are not reaudited here.

Run on 194: `/opt/trt-asc/venv/bin/python3 /workspace/experiments/teacher33-policy-20261020/replay.py`. Raw rounds and hashes are in `result.json`; compact counts are in `summary.json`. `manifest.json` binds the local files.
