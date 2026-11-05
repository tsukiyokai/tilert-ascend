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
selects the bridge system workspace, and supplies a drained, reset `TPipe`.
Every mixed core must enter the call, including cores without row or matrix
ownership. Only GM values may survive from a preceding stage.

The original bridge boundaries, body workspace switch, six body barriers,
publication after body return, and BF16 epilogue remain in their original
order. The function returns after the original final `EndPhase`: output stores
are published and the pipe is reset. An FFN TP reduction may then consume the
output. The body system workspace remains selected; a later matrix stage
selects its own system workspace. Neither attention nor either TP reduction
is implemented here.

`build.py PARENT_BUILD NEW_OUTPUT` verifies and copies the frozen sources,
extracts the body, renames the host/device namespace and C symbols, and replays
the parent's recorded compiler/linker commands. The parent and candidate are
CPU-loaded together and their schemas are compared. Source, statement, command
and library identities are retained. Device equivalence must be checked using
the generated `check.py`; unchanged statement bytes and successful compilation
are separate from device validation.
