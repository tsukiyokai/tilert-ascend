# The round has a 56.7 ms floor, and the tail is worth 0.111 ms a token

TPOT mixes two quantities that can move in opposite directions — how long a
round takes and how many rounds a thousand tokens need — and the four-patch
episode moved both at once. The run's own record already carries one
`advance_mtp_n` entry per round with its own `elapsed_ns`, so the round can be
read directly, with no profiler and therefore no profiler-induced starvation.

`round_clock.py` in this directory reads it.

## Six requests, two independent runs, one floor

| run | rounds | TPOT | round median | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| this baseline | 227 | 12.9709 | 57.616 | 56.757 | 60.840 |
| | 227 | 12.8348 | 57.487 | 56.789 | 59.160 |
| | 227 | 12.7836 | 57.204 | 56.676 | 58.809 |
| the earlier width-8 run | 227 | 13.7257 | 59.991 | 56.944 | 67.690 |
| | 227 | 13.0657 | 57.948 | 56.728 | 61.332 |
| | 227 | 12.9079 | 57.381 | 56.595 | 60.769 |

The p10 sits between 56.595 and 56.944 in all six — a spread of 0.349 ms, or
0.6%, across two runs built in different directories on different source. The
median ranges 57.2 to 60.0 and the p90 ranges 58.8 to 67.7, so everything that
varies between requests lives in the tail and the floor does not move.

Holding every round at its own p10 would be worth
`(57.2 − 56.7) × 227 / 1023` = **0.111 ms a token**. That is the whole prize
in the variance, and it is not where the work is.

## What the floor costs against the target

Width 8 accepts 1023 tokens in 227 rounds, so 4.507 a round, and a round may
therefore cost 9.013 ms at a 2 ms TPOT.

| | |
| --- | --- |
| round now | 57.2 ms median, 56.7 floor |
| over budget | 6.3× |
| weight floor, per rank per round | 3.886 ms, 43% of the budget |
| left for everything else | 5.127 ms allowed against 53.3 now, **10.4×** |

Width 8 improves the budget's composition, not only its size: at width 4's
acceptance the weight floor was 57% of the round's allowance and here it is
43%. So the target is a 10.4× on work that is not the weight stream, against a
weight term that now has room.

## Why this replaces the earlier framing

The arithmetic this project has been using — a 6.866 ms round at acceptance
3.433, with host launch, collectives and weight stream each large enough to
break it — was computed at width 4 before the hoist and before width 8 worked.
Both of its premises have moved. The figures above come from one run's own
record at the configuration that now holds the best deterministic TPOT.
