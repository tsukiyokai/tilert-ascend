# Width 8 is the optimum, bracketed on both sides, and rows are nearly free

## The four arms, all `status=measured` with a zero exit

| width | chain | verify | round floor | acceptance | TPOT |
| --- | --- | --- | --- | --- | --- |
| 6 | 1.746 × 4 | 39.496 | 48.76 | 4.125 | 12.342 |
| 8 | 1.707 × 6 | 39.677 | 52.26 | 4.507 | **12.047** |
| 12 | 1.684 × 10 | 41.460 | 60.74 | 4.871 | 12.995 |
| 16 | 1.744 × 14 | 43.338 | 70.29 | 5.356 | 13.847 |

The marginal token, from the measured floors and acceptances:

| | cost | tokens | a token | against the average | |
| --- | --- | --- | --- | --- | --- |
| 6 → 8 | +3.50 | +0.382 | 9.17 | 12.05 | pays |
| 8 → 12 | +8.48 | +0.365 | 23.25 | 12.99 | does not |
| 12 → 16 | +9.54 | +0.485 | 19.69 | 13.85 | does not |

So the optimum is exactly 8, and the earlier reading that put it there was
right for a reason it could not have known: that reading came from the
configuration whose captured width ran a stale static-kernel package.

## Rows are nearly free up to 8 and cost 0.45 above it

| | the verify, a row |
| --- | --- |
| 6 → 8 | +0.090 ms |
| 8 → 12 | +0.446 ms |
| 12 → 16 | +0.470 ms |

**The verify costs the same at six rows and at eight.** So its 39.5 ms is not
the work of eight rows — it is a fixed cost per forward, and the knee at 8 is
where the rows start to matter. That is the sharpest statement yet of what the
2 ms target has to remove: not arithmetic, not bandwidth, but whatever a
forward pays once regardless of how much it carries.

A chain step is 1.712 ms independent of width, so the same holds there: three
arms with 4, 6, 10 and 14 steps give 1.746, 1.707, 1.684, 1.744.

## The fit, and where it fails

Fitted on 8, 12 and 16: `round(w) = 1.712(w − 2) + 0.458w + 41.753`. It
predicts 52.26, 60.94 and 69.62 against 52.26, 60.74 and 70.29 — and 51.34 at
width 6 against a measured 48.76, because the 0.458 a row it assumes does not
hold below 8. The linear form is an interpolation between 8 and 16 and not a
law.

Acceptance carries across configurations: the token digest is `68f2bc184553`
in every arm at every width, so the same tokens are accepted whatever the
kernels underneath.

---

# The round is linear in width, and widening past 8 does not pay

Four arms in one session, the last repeating the first as the chain's own
drift control. Best configuration: three gmm1 patches, conditioned graph,
cache fix. Every arm `status=measured`, no rank errors, exit 0.

| width | chain | verify | round floor | round p10 | rounds | acceptance | TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1.707 × 6 | 39.677 | 52.261 | ~52.3 | 227 | 4.507 | **12.047** |
| 12 | 1.684 × 10 | 41.460 | 60.743 | 62.280 | 210 | 4.871 | 12.995 |
| 16 | 1.744 × 14 | 43.338 | 70.286 | 71.739 | 191 | 5.356 | 13.847 |

A chain step costs 1.712 ms and does not depend on the width — three arms with
6, 10 and 14 steps give 1.707, 1.684, 1.744. The verify grows 0.458 ms a row,
measured twice at 0.446 and 0.470. So

    round(w) = 1.712(w − 2) + 0.458w + 41.753 = 2.169w + 38.330

which predicts 52.26, 60.94, 69.62 against measured 52.26, 60.74, 70.29.

## The criterion

Widening costs **2.169 ms a unit of width**, so it pays only while acceptance
grows faster than 2.169 / TPOT = **0.187 tokens a unit**.

| | acceptance slope | verdict |
| --- | --- | --- |
| 4 → 8 | 0.294 | pays |
| 8 → 12 | 0.091 | does not pay |
| 12 → 16 | 0.121 | does not pay |

Acceptance carries across configurations: the token digest is `68f2bc184553`
in every arm at every width tested, so the same tokens are accepted whatever
the kernels underneath. That is what lets the width-4 acceptance of 3.332,
measured on the older configuration, be used here.

So the optimum is above 4 and below 12, and width 6 is the one untested point.
Its round is predicted at 2.169 × 6 + 38.330 = 51.34, which beats width 8 only
if acceptance at 6 is above 51.34 / 12.047 = 4.262 — against 4.507 at width 8
and 3.332 at 4. Linear interpolation gives 3.92, so width 8 should hold, but
the interpolation is the weakest link and one arm settles it.

## What it says about the two halves of the round

The chain's cost per unit of width, 1.712 ms, is twelve times the verify's
0.458. A speculative round buys its extra candidates almost entirely in the
chain, and the chain is serial single-row forwards whose cost is a fixed
overhead rather than work: one row costs 1.712 against 0.458 for one more row
inside the verify, **3.7×**.

That is the same ratio the phase split gave from a different direction — a
single-row MTP layer at 1.731 against 0.501 a layer inside the eight-row
verify. Two independent measurements of the same fixed cost.
