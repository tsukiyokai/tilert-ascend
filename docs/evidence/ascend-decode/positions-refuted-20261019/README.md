# Inconclusive: the arms differed by run as much as by patch

## 2026-10-19, corrected the same hour

A third arm in a later chain ran the **same configuration** as the baseline
here — barrier-only overlay, cache off, width 8 — from a different working
directory, and read a different floor:

| | p10 of a round | mean |
| --- | --- | --- |
| baseline, first chain | 56.757, 56.789, 56.676 | 56.741 |
| the same configuration, later chain | 57.110, 56.875, 56.839 | 56.941 |
| with the patch, first chain | 57.033, 56.846, 57.177 | 57.019 |

Two identical configurations differ by **+0.201 ms**. The patch was credited
with +0.278, so the effect is 1.39× the spread between runs of the same thing,
and against the configuration that ran under comparable conditions the patch
is **+0.077** — nothing.

The error was mine and it was specific: a round's p10 holds to 0.6% *within* a
run, and I treated that as reproducibility *between* runs. It is not. The
between-run floor for a three-request arm's p10 mean is about 0.2 ms, so an
effect that size needs the arms alternated within one session rather than run
one after the other.

So this page's conclusion does not hold. The patch is neither shown to hurt
nor shown to help, and the revert stands only on the usual default — an
unproven change earns no place. What follows below was written before that
third arm existed.

---

# The metadata copies sit next to the idle without causing it

A controlled test of the one piece of the idle attribution that could be
controlled. It failed, and that bears on the rest of the attribution more than
on the patch.

## The claim under test

Ranking the device's steady-state idle by the operator that ends at each gap's
start put four families at 66.714 of 97.962 ms, 68%. Two of them were ours:
`aclnnInplaceCopy` over 31 gaps at 318.4 µs each, and
`aclnnInplaceFillScalar` over 26 at 315.4 — the host-to-device copies and
scalar fills a forward issues to tell the device where it is.
[the idle](../device-idle-20261019/README.md)

The change removed three of them: `seq_lens` already carries `position + width`
on the device, so positions and slots can be derived from it with a persistent
offset vector, leaving one copy of a single integer and no per-forward host
allocation.

## The measurement

Width 8, one tree, both arms resolving gmm1 to the barrier-patched overlay at
index 3 of their own recorded search path, differing by this patch alone.
Judged on a round's tenth percentile, the statistic that holds to 0.6% across
six requests in two independent runs.
[the round clock](../round-clock-20261019/README.md)

| | p10 of a round | TPOT |
| --- | --- | --- |
| baseline | 56.757, 56.789, 56.676 | 12.9709, 12.8348, 12.7836 |
| with the patch | 57.033, 56.846, 57.177 | 13.2023, 12.8156, 13.1831 |

Every value with the patch exceeds every value without it — complete
separation, three against three, a sign test at 0.05 one-sided. The floor
rises 0.278 ms a round, 0.062 a token. The TPOT medians, 12.8348 against
13.1831, have overlapping ranges and would not have settled it.

Reverted.

## Why it went the wrong way

The patch traded two host-to-device copies of a small integer tensor and one
host `arange` for three device kernel launches — a subtraction, an add into
`slots`, a widening copy into `positions` — and a round at width 8 runs nine
forwards, so twenty-seven more launches.

A `copy_` from a CPU tensor into a device tensor does not synchronise the
device. It was never a host round trip. The premise assumed it was.

## What this costs the rest of the attribution

(Written before the correction above, and weakened by it: the test could not
resolve an effect of this size, so it did not refute anything. The attribution
remains unsupported because adjacency is not causation, which was true before
any of this was measured.)

Gap adjacency is not causation, and this is the piece of that ranking a
controlled experiment could reach. It came back negative: those operators
precede the idle without producing it.

So `aclnnIndex` at 34.938 ms and `aclnnArgMax` at 13.706 carry no more weight
than these two did. The 68% figure should not be spent against, and the rope
gather located behind `aclnnIndex` is a lead about where an operator is, not
evidence that removing it buys time.

What survives from that page is what did not depend on adjacency: `EVENT WAIT`
is 99.9% overlapped and not a lever, device-side `Cast` is 2.86 µs a span, and
the idle is in gaps of 100 µs and more rather than in launch-rate dribble.
