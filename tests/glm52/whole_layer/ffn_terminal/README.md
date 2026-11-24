# Complete FFN terminal completion

This isolated variant of the active-descriptor/token-output composition replaces
only the final global entry's `EndPhase` with local `PIPE_ALL` and DDR completion.
It retains the ordinary `TPipe` destructor, all five body joins, all arithmetic,
the 408-byte descriptor and the 33-buffer interface. Peak UB remains 116,480
bytes. The callable body is unchanged; this completion policy requires that all
output consumers run in a later task on the same stream or wait on its event.
The [source audit](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-terminal-source-audit/README.md)
verifies the parent's 83 source hashes, final DMA publication, completed
matrix operations, empty trace calls and retained earlier joins.

The checker retains both reference paths and all 31 boundary checks. It also
queues copies of route, routed, scaled and combined outputs immediately after
the candidate inside the captured graph. Every replay poisons those outputs
and consumer destinations before running changing inputs. The
[M12/24 A3 gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-terminal212-m12-v1-a3-194/summary.json)
passes all 560 raw-byte checks, including 64 immediate-consumer comparisons,
across eight changing inputs and both reference paths. The
[remaining seven configurations](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-terminal212-wide-v1-a3-194/summary.json)
pass another 3,920 checks. Together these cover M=1,2,12,32 and 16/24 block
groups with 4,480 raw-byte checks and 512 immediate-consumer comparisons.
The [matched-address M12/24 comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-terminal-timing-v1-a3-194/summary.json)
saves median 8.834/8.019 us against the active-token-join parent in the two
owner arenas; all 60 pairs are faster. Candidate/parent medians are
211.257/220.077 us and 211.925/219.951 us. All 19,200 raw-byte checks,
360 event-tail checks and 696 full active-table checks pass. These results
use 65 active experts and 16 weight banks; they measure a local synthetic
FFN with no TP reduction or model TPOT.
