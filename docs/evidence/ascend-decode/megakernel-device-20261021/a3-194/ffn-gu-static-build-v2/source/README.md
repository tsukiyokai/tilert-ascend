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
