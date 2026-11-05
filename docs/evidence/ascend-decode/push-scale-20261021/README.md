# The one-sided push wins small messages and loses large ones

2026-10-21, a3-84, two ranks, hidden 6144, float32.

    rows      KB   hccl us   push us   factor
       1      24     189.3      27.4     6.90
      12     288     258.1      26.1     9.91
      64    1536     118.6      54.7     2.17
     256    6144     120.7     172.9     0.70

The advantage peaks at the decode shape -- twelve rows, 288 KB -- at 9.91x, and
inverts by 6 MB.

## The mechanism, and what it means for the lever

HCCL spends 189 to 258 us on a message of 24 to 288 KB: that is rendezvous, not
transfer, and it is flat in the size. The push skips the rendezvous and costs
26 us for the same payload. Past a megabyte HCCL's transfer path takes over and
the push, which moves bytes one-sidedly without that machinery, falls behind.

So the lever's size depends on which regime the engine's collectives are in.
The capture says 20.5 calls a round at 858 us each. Two-rank HCCL at the decode
shape is 258 us, so 858 is either a much larger message or the sixteen-rank
cost of the same one. If the latter, the push's advantage should hold and the
lever is near the 14.7 ms end; if the former, the engine is already batching
into the regime where the push loses, and the lever could be negative.

That distinction needs sixteen ranks. What two ranks establish is the shape:
**the push is a latency optimisation, so it pays only where the collective is
latency-bound**, and batching collectives -- which the engine appears to do,
20.5 calls where a layer-wise reduction would give 156 -- moves them out of
that regime.

## Which reframes the lever

If the engine batches 156 layer-wise reductions into 20.5 calls, then undoing
that batching and pushing each one-sidedly is the configuration to test: 156
pushes at 26 us is 4.1 ms a round against 17.56. The one-sided path and the
batching are alternatives, not complements, and the arm as queued tests the
push *against* the batching rather than instead of it.
