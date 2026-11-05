# FFN body argument descriptor

This independent candidate changes how the frozen composed FFN supplies
arguments to its routed/shared body. The parent is uninstrumented composed-v2,
library SHA `3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655`.
The candidate uses `tilert_ffn_descriptor` and `ffn_descriptor_launch`.

The external C ABI, Torch preparation/validation, all 32 scratch tensors and
their ownership are unchanged. A 392-byte kernel argument contains six weight
addresses, 20 body addresses, DispatchInput, SharedInput, M and group count.
The host launch wrapper fills only addresses and immutable geometry; it never
reads routing values. The three dispatch addresses refer to bridge outputs
produced later in the same invocation. The argument is copied by value into
the device kernel and passed to the body by const reference.

This adds 56 kernel-parameter bytes for DispatchInput, formerly constructed
inside the kernel. It allocates no GM scratch. The 112-byte SharedInput and
all three references to its bridge/body producers keep their original values.

## Field equivalence

| Original body parameters | Descriptor source |
| --- | --- |
| GU/down weights, correction and scales | Six original weight addresses |
| Restored rows/scales, counts/starts/source/slot/probability metadata | Original body slots 10 through 16 |
| Stacked input, GU product, middle codes/scales, down product, route output | Original body slots 0 through 5 |
| Final routed output | Original body slot 9 |
| GU/down tiling and system workspace | Original body slots 6 through 8 |
| Dispatch BF16 input, IDs and probabilities | Bridge slots 0, 5 and 4 |
| Shared input and geometry | Original SharedInput, M and group count |
| Expert count and trace | 256 and null, as in the parent |

`prepare.py` replaces only parameter identifiers in the body's numerical
statements. Reversing those substitutions must recover the parent bytes
exactly. Helpers, rounding, matrices, six body barriers, bridge boundaries and
both outer EndPhase calls retain their original statements. The later
SetSysWorkspace call remains at its original position; CANN 9.2 selects only
the first non-null system workspace, so this is not a workspace-switch change.

## Build and validation

Run `python test_prepare.py -v` for source equivalence, host-validator
preservation, original-call parameter mapping, and a portable C++ address/POD
check across M1 through M32 at 16/24 groups. `FFN_DESCRIPTOR_PARENT` can point
to a different location of the same frozen parent.

`python build.py PARENT_BUILD NEW_OUTPUT` verifies the parent library and
source identities, creates a private source copy, and replays the original
compiler/linker commands serially. It co-loads the parent/candidate libraries
on CPU, checks Torch schemas, and preserves commands, source and artifact
hashes. The generated checker changes only the native Torch namespace.

The intended compiler effect is fewer simultaneous scalar argument loads or
spills at body entry. The compiler may reproduce the original instruction
sequence; successful compilation does not establish that effect. Inspect
the actual binaries before interpreting performance. Device acceptance uses
the unchanged production-byte gates. Performance requires same-address
whole-kernel event timing in both arenas; the profiled setup interval is not
an expected saving or a TPOT result.
