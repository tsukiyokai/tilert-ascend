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
|---|---|
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

The CANN 9.2 build is `build-v1/ffn_descriptor.so`, SHA
`f874bee1c83cf2e7549ed472d74a373706c7f475592223cd6e1d670b7e38a143`.
Five source/POD tests passed on x86 and ARM. The M12/24-group device gate
passed eight candidate and sixteen reference changing graph replays, with
480 raw-byte comparisons against both separate native and official numerical
chains. This covers one rank-local FFN, excluding TP and model latency.
The [device report](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-descriptor212-m12-v1-a3-194/summary.json)
records the tested library identity.
The [remaining seven configurations](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-descriptor212-wide-v1-a3-194/summary.json)
also pass. Together they cover M1/2/12/32 at 16/24 groups, NZ routed
weights and BF16 shared scales: 64 candidate and 128 reference changing
graph replays, with all 3,840 raw-byte boundary comparisons passing.

The actual uninstrumented device binaries differ. `assembly_audit.py` counts
the following candidate-minus-parent static instructions over each complete
entry function:

| Entry | All instructions | LD/LDP | ST/STP/STI | Direct X30 memory operands |
|---|---:|---:|---:|---:|
| NZ AIC | -14 | -3 | -31 | -31 |
| ND AIC | +27 | +21 | -20 | 0 |
| NZ AIV | +11 | +9 | -3 | -9 |
| ND AIV | +11 | +9 | -3 | -9 |

Both binaries contain zero SYS_CNT reads. These counts establish a compiler
change; they do not count executed loads, identify memory stalls or measure
body-entry latency. The [build evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/a3-194/ffn-descriptor-build-v1/assembly-audit.json)
retains the disassembly identities and counting script.

## Matched-address timing

`prepare_bench.py` derives a two-arm benchmark from frozen FFN timing-v3.
The `fused` result label means descriptor; `separate` means the accepted
one-launch parent. Both use the same 32 buffers within each of two original
owning-Torch arenas, with no extra allocation. The original owner validators,
floating NaN and bounded integer poison, finite references, per-bank changing
input checks, post-event output checks and full post-timing checks are retained.

The frozen `timing-v1` uses sixteen distinct equal-valued weight banks,
sixteen independent FFN calls per graph, thirty alternating event pairs per
arena and thirty graph replays per event. Copies and poison are outside the
event windows. Four CPU tests passed on x86 and ARM; separate C++ translation
units used the real parent and candidate headers to verify all arguments
across M1/2/12/32, 16/24 groups and ND/NZ. The four actual libraries co-loaded
on ARM with NPU initialization false. `bench_run.sh preflight OUTPUT` performs
that CPU check; `bench_run.sh device OUTPUT` is the device timing entry.

The [A3 M12/24-group comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-descriptor-timing-v1-a3-194/summary.json)
with 65 active experts saves 5.192 and 2.423 us in the two owning arenas,
with 30/30 wins in each. Candidate/parent medians are 235.801/240.962 us
and 234.768/237.178 us. All 19,200 numerical boundary comparisons and
360 post-event output checks pass. Both arms share all 32 scratch addresses.
This measures the complete local FFN with synthetic weights; it does not
isolate body setup or establish a model TPOT gain. Active-expert enumeration
was tested separately, so its saving cannot be added to these results.
