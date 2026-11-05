# Complete FFN terminal completion

This isolated variant of the active-descriptor/token-output composition replaces
only the final global entry's `EndPhase` with local `PIPE_ALL` and DDR completion.
It retains the ordinary `TPipe` destructor, all five body joins, all arithmetic,
the 408-byte descriptor and the 33-buffer interface. Peak UB remains 116,480
bytes. The callable body is unchanged; this completion policy requires that all
output consumers run in a later task on the same stream or wait on its event.

The checker retains both reference paths and all 31 boundary checks. It also
queues copies of route, routed, scaled and combined outputs immediately after
the candidate inside the captured graph. Every replay poisons those outputs
and consumer destinations before running changing inputs. Device correctness
and latency have not yet been measured for this variant.
