# Capturing the target-conditioned pass: −2.979 ms a round

Alternated A B A B A B in one session, paired pass by pass, judged on each
arm's lowest round p10. Width 8, the three gmm1 patches, cache fix on, same
tree, same overlay read back from every arm's own record.

| pass | off | on | delta |
| --- | --- | --- | --- |
| 1 | 55.880 | 52.717 | −3.162 |
| 2 | 55.163 | 52.365 | −2.798 |
| 3 | 55.143 | 52.166 | −2.977 |

**−2.979 ms a round, −0.661 a token.** Three passes, same sign, spread 0.365 —
18× the 0.163 ms between-run floor.

All four token digests are `68f2bc184553`, width 4's reference stream, so the
gain costs nothing in trajectory. Best TPOT **11.7715 ms**, from 12.6037.

And all six arms report `status=measured`, `agreement=True`, no rank errors —
the first runs in this project to pass its own gate, which had been failing on
a missing field.
[the gate](https://github.com/tsukiyokai/TileRT-Ascend/commit/837f7dd)

## What had to be fixed for the flag to do anything

`--conditioned-graph` existed and was a no-op at any verify width but 4,
because the width was named twice:

- `enable_graph` built `CannMtpGraph(self, 4)` whatever the verify width;
- `_graph` returned that graph only when `width == 4`.

Fixing the first alone left the graph captured at the right width and never
replayed — `captures: 1, replays: 0`, and all sixteen ranks said so in
`summary.json` while the A/B measuring it read a 0.047 ms difference. The
selector now compares against the graph's own width and there is no literal
left to disagree with.

With both fixed: `captures: 1, replays: 1362` — one capture, one replay per
round over six requests, which is what `validate_local` asserts.

## What the size of the gain says

The phase hook had charged this pass 42.877 ms. Most of that was the verify's
device work, which it blocked on: the verify's own 2.196 ms is below its
weight-read floor of 3.886, so that figure was dispatch and this one was
waiting.
[the correction](../phase-split-20261019/README.md)

But 2.979 ms of it was real and is now gone — the pass's own eager dispatch of
one MTP layer. So both readings were partly right, and the control separated
them where neither the hook nor the arithmetic could.
