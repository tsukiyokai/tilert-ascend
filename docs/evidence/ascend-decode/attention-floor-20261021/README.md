# Attention is already better than a dense implementation of itself

2026-10-21, a3-84, one die. The same 10.3 MB paged cache, 9344 positions by
576 wide.

    sum over positions        22.316 us   449 GB/s   1.74 ms a round
    score, softmax, weigh     95.271 us   105 GB/s   7.43 ms a round
    the capture's operator    85.048 us   121 GB/s   6.63 ms a round

The vendor operator is 11 percent faster than a plain dense implementation of
the same attention in torch.

## Correcting the gap this was meant to size

The reasoning was: the operator runs at 121 GB/s where a contiguous read of
the same bytes reaches 380, so six of its 6.63 ms a round are waste and six is
what separates the best projected TPOT from 2.x. The first half is arithmetic
and the second half does not follow -- the operator does not only read those
bytes, it scores them, takes a softmax and weighs them, and doing that in
torch costs 95.271 us. A pure read is 22.316; the work between is the
difference, not an overhead.

So attention's 6.63 ms a round is near its own floor given the algorithm, and
the levers that would change the algorithm are closed: fewer picks moves it
five percent, pre-gathering the selection is 35 percent slower, and lower
precision is a quality change rather than an optimisation.

## Where that leaves the target

    fixed term after the merges     24.84 ms
      attention                      6.63
      MoE grouped matmul             5.20
      the rest                      13.01
    what 2.9 ms needs at width 24   15.80 ms

The 10 ms would have to come out of "the rest", which is what the five
boundary merges already swept. So on everything measured, 2.x ms is not in the
mechanisms identified so far: the honest best is 3.73 ms, and closing the
remaining gap needs a mechanism this project has not found rather than a better
measurement of one it has.
