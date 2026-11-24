# Active-expert FFN with body argument descriptor

This candidate applies the validated descriptor substitution to active FFN
build-v1, library SHA
`6a6241da7d7a5e2e4a2ba9396d20b902f0ca19fa2e4af224a7e7da41a3dc6513`.
The native namespace is `tilert_ffn_active_descriptor` and the C entry is
`ffn_active_descriptor_launch`.

The 408-byte argument holds six weight addresses, 21 body addresses,
64-byte DispatchInput, 112-byte SharedInput, M and group count. Host
construction captures addresses and geometry; routing data is produced
on device. The active-list field names body slot 20. The external active
ABI and all 33 scratch tensors retain their original types and ownership.
No additional GM buffer is allocated relative to the active parent.

Reversing the descriptor field substitutions must recover the active
body's numerical statements byte for byte. The six body barriers, bridge
boundaries, outer EndPhase calls and first system-workspace selection
remain unchanged. The checker retains all 30 numerical boundaries and
the complete INT32[264] active table, including unused slots.

Run `python test_prepare.py -v` for the source/ABI contract. Set
`FFN_ACTIVE_DESCRIPTOR_PARENT` to another copy of the same frozen parent
if needed. `python build.py PARENT NEW_OUTPUT` verifies source hashes,
replays the parent's compiler commands, checks both actual Torch schemas
and co-loads the two libraries without initializing an NPU.

Separate descriptor and active-list timings do not establish their
combined effect. Device acceptance requires the unchanged byte gates;
performance requires a direct matched-address comparison against active
FFN, including the same active-table initialization and graph replay.

Build-v1 has SHA
`f7621e49a818a77d7377246f74c7feaa9a5b8cb9a7b81feacc4cf195d1bd1696`.
Four source/POD tests passed on both x86 and ARM; the original active and
combined libraries co-loaded without NPU initialization. The
[M12/24-group device gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-descriptor212-m12-v1-a3-194/summary.json)
passes eight candidate and sixteen reference changing graph replays.
All 496 raw-byte checks pass, including the complete active table.

`prepare_bench.py` derives a comparison against active FFN from frozen
variant timing-v3. Both arms share the same 33 tensor addresses. Each
checked replay and timed warmup starts with active-table word 0 zeroed
and all remaining words set to one; both arms must replace the entire
table. Original owning-Torch validation, numerical references and event
windows are retained. Four CPU tests passed on x86 and ARM, including
32 recorded C ABI calls across M1/2/12/32, 16/24 groups and ND/NZ.
All five actual libraries co-loaded on ARM without initializing an NPU.

The [A3 M12/24-group paired comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-descriptor-timing-v1-a3-194/summary.json)
with 65 active experts and sixteen independent weight banks saves
19.813/19.851 us against active FFN in the two owner arenas. All 60 pairs
are faster. Combined/active medians are 224.177/243.945 us and
226.389/246.341 us. All 19,200 numerical boundary checks, 360 event-tail
checks and 696 complete active-table checks pass. This measures the
complete rank-local FFN; it does not isolate body setup or include TP
reductions, attention or model TPOT.
