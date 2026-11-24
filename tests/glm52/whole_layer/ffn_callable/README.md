# Callable rank-local FFN

`prepare.py` extracts the validated composed FFN into `source/native.h` without
changing its numerical statements. The parent is composed build-v2, SHA
`3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655`.
The independent wrapper uses `tilert_ffn_callable` and `ffn_callable_launch`;
the original library and default path remain available.

`ffn_callable_device::Forward<format>(pipe, input, residual, gamma, gate, bias,
weights, bridge_addresses, body_addresses, shared, rows, blocks, epsilon)`
accepts the same 12 bridge and 20 body buffers and weight addresses as the
parent. The caller initializes SoC and mixed synchronization state once,
provides one selected 16 MiB system workspace for the full call, and supplies
a drained, reset `TPipe`.
Every mixed core must enter the call, including cores without row or matrix
ownership. Only GM values may survive from a preceding stage.

The original bridge boundaries, body setup, six body barriers,
publication after body return, and BF16 epilogue remain in their original
order. The function returns after the original final `EndPhase`: output stores
are published and the pipe is reset. An FFN TP reduction may then consume the
output. Neither attention nor either TP reduction is implemented here.

On CANN 9.2 `dav_c220`, `SetSysWorkspace` assigns only when the reserved pointer
is null. The frozen standalone wrapper selects bridge scratch slot 9 first;
the later call with body scratch slot 8 does not switch that selection. Both
16 MiB allocations remain in the ABI. The selected region must remain alive
through every stage that uses it. A caller embedding this function after
another stage must establish that the already selected workspace covers the
combined lifetime and capacity; another `SetSysWorkspace` call alone cannot
establish a switch. The [SDK and disassembly evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-system-workspace-source-audit/summary.json)
records this behavior without changing the frozen implementation.

`build.py PARENT_BUILD NEW_OUTPUT` verifies and copies the frozen sources,
extracts the body, renames the host/device namespace and C symbols, and replays
the parent's recorded compiler/linker commands. The parent and candidate are
CPU-loaded together and their schemas are compared. Source, statement, command
and library identities are retained. Device equivalence must be checked using
the generated `check.py`; unchanged statement bytes and successful compilation
are separate from device validation.

The [M12, 24-group device gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-callable212-m12-v1-a3-194/summary.json)
passes 16 dual-reference cases and all 480 raw-byte boundary comparisons,
with eight candidate and 16 reference changing-input replays. The library SHA
is `6dad131d70300e6775aa43f46cf034895d00398a1d805f698b42984c61743d89`.
This gate covers callable equivalence at that geometry; it contains no timing
or TP reduction measurement.
