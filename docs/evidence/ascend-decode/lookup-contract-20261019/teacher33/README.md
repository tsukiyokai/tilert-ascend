# Copy proposals on the frozen teacher window

A3-194 server CPU; the exact 8192 input IDs and 33 frozen outputs, including
the first output already emitted by prefill. Each policy follows its own
commit boundaries. Proposals use only the visible context; future labels
score an already fixed proposal. Pending-token padding and final bonus
clipping follow the runtime.

| Width | Rounds for 32 new outputs | Planned verification rows |
| --- | --- | --- |
| 8 | 28 | 224 |
| 16 | 28 | 448 |
| 32 | 28 | 896 |

Every width commits one token in 25 rounds, two in two rounds and three in
one round. The actual indexed drafter, independent reverse scan and existing
replay agree at every boundary. This short window provides no acceptance
benefit from widening. It does not represent later code-copy regions or
establish target execution time.

[Summary](summary.json) binds source and input hashes;
[raw rounds](result.json) include source positions, candidates and commits.
Run [replay.py](replay.py) on the server beside the frozen `source/` files.
No Torch/NPU import or old 1024-token trajectory is used.

[Matched MTP4 replay](mtp-comparison/README.md) covers a common prefix of
30 new outputs: MTP4 needs14 rounds/56 target rows, lookup8 needs26/208.
MTP wins this replay only when its mean complete round costs less than
1.857 times the lookup round, including seed, chain and conditioned work.
The MTP candidates were recorded under M1 teacher conditioning; live
M4-conditioned candidate equivalence and actual round costs remain to be measured.
