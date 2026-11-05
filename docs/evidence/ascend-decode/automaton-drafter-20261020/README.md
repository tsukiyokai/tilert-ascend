# The automaton drafter, end to end

2026-10-20, a3-84, sixteen ranks, width 32, same machine for both arms.

    base       tpot 11.2312  rounds 194  accepted/round 5.273  round 63.869 ms
    automaton  tpot 10.8510  rounds 169  accepted/round 6.053  round 72.066 ms

Acceptance **5.273 to 6.053**, up 14.8%, inside the 1.11 to 1.27 the CPU
sweep predicted. Rounds fall 194 to 169 for the same output. TPOT improves
3.4%.

The first lever in this project to improve TPOT end to end at all.

## Why 14.8% on acceptance buys 3.4% on TPOT

Round time rose 63.869 to 72.066 ms, 12.8%, which eats most of it. The cause
is structural rather than host cost. Both arms verify at width 32, but the
hash-tuple drafter frequently returns an empty or short draft -- it gives up
when no tail of length `shortest` or more has a continuation -- and a round
that verifies four tokens is cheaper than one that verifies thirty-two. The
automaton backs off along its suffix links instead of giving up, so it almost
always fills the width.

So the automaton buys acceptance with more expensive rounds, and at width 32
the trade is close to even. Which names the next experiment: a smaller width
should keep most of the acceptance, since the gain comes from matching better
rather than from drafting longer, and give the round time back.

## The width sweep the decomposition asked for

    arm                     tpot_min  rounds  accepted   round
    base    chain8   w32     11.2312     194     5.273  63.869 ms
    automaton        w32     10.8510     169     6.053  72.066 ms
    automaton        w16      9.6360     183     5.590  58.588 ms
    automaton         w8     10.6425     225     4.547  52.787 ms

Width 16 is 14.2% better than the baseline, and it works the way the
decomposition said it would: acceptance stays above the baseline's at 5.590
where width 32 reached 6.053, and the round becomes cheaper than the
baseline's own, 58.588 against 63.869. Width 8 overshoots -- acceptance falls
to 4.547 and the cheaper round does not pay for it.

So the gain is not from drafting longer. It is from matching better, and once
the match is better the cheapest width that still carries it wins.

## The control, which moves most of the credit to the width

    arm                     tpot_min  rounds  accepted   round
    chain8           w32     11.2312     194     5.273  63.869 ms
    chain8           w16      9.8913     206     4.966  52.742 ms
    automaton         w8     10.6425     225     4.547  52.787 ms
    automaton        w12      9.3794     188     5.441  55.424 ms
    automaton        w16      9.6360     183     5.590  58.588 ms
    automaton        w32     10.8510     169     6.053  72.066 ms

Narrowing `chain8` from 32 to 16 is worth 11.2312 to 9.8913 on its own. So
the original baseline's width was simply a bad operating point: going from 16
to 32 costs 63.869 against 52.742 in round time and buys only 4.966 to 5.273
in acceptance, which does not pay. Most of the 14.2% I attributed to the
drafter was the width.

The drafter's own contribution is what is left at equal width: 9.636 against
9.8913 at width 16, 2.6%. Taken together, 11.2312 to 9.3794 is 16.5%, of
which about 12 points are the width and about 3 to 5 the drafter.

Width 12 beats 16 for the same reason width 16 beats 32: acceptance 5.441
against 5.590 is a small loss and 55.424 against 58.588 is a larger gain. The
optimum is interior -- 8 loses too much acceptance at 4.547, 32 costs too much
round at 72.066 -- so it is a real extremum rather than an edge.

## On the label

The arm reports `drafter lookup`, because the minimal patch to this tree's
bench left its report line unconditional. The label is wrong and the swap is
not: acceptance 6.053 against 5.273 and 169 rounds against 194 cannot come
from the same drafter.

## What was measured on the CPU first

    edit rate   chain8  chain32  automaton     tree    swap
         0.00   29.035   32.033     32.230   32.171   1.110
         0.02   10.295   10.819     12.176   12.132   1.183
         0.05    4.859    4.965      6.020    6.302   1.239
         0.10    2.579    2.590      3.282    3.680   1.272
         0.20    1.495    1.496      1.877    2.208   1.256

The column that mattered for sequencing is `tree` against `automaton`: they
are within half a percent at the low edit rates, so most of the tree's gain is
available without tree attention masks or position ids -- which is why this
was measurable today and the tree is not.
