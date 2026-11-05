# Private communication tree: CPU build

The private tree reducer compiled with CANN 9.2 and co-loaded with the unchanged
serial probe in the actual Torch 2.12 image on a3-78. No NPU was mapped or
initialized. Device arithmetic and graph replay are separate gates.

| Artifact | SHA256 |
| --- | --- |
| Candidate DSO | `a0bd2462e2f999280720fa5060466159a2eab13ce5541639ddc7afef35a45ebf` |
| Accepted serial DSO | `1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f` |
| Frozen source manifest | `22d408d22afcccfce6941a0a9cd5c2cdc3e38c53110401f21cb46832b4b79716` |
| Actual CPU build report | `b857a54c47aa61deb5429e81ee84651fdec7e05f840343d7251f64c41b6961b3` |

The [build report](build.json) records 22 frozen source files, 859 compiler
dependencies, 68 SHMEM headers, the three SHMEM DSOs, the compiler binaries and
80 mapped dependency libraries. The image identity and exact no-device Docker
command are in [the command record](communication-tree-build-v1-command.json).
The private C ABI has six `communication_tree_probe_` exports, distinct from
the parent exports in the same process. Neither library's initialization
function was called during the CPU check.

Five source/ABI contracts passed on both x86 and ARM. The actual C++ plan
matched an independent round recurrence for all 15 rank counts from 2 to 16,
including exact expression identity, every leaf once, `n-1` additions and a
five-slot maximum. It also passed 3840 random finite BF16-valued calculations
and 2160 CPU special-value comparisons. See [the plan result](plan-contract.log)
and [the ARM test log](python-contract.log).

The change retains the parent transport, ready signals, epochs, window layout,
2048-element tile distribution, all barriers and the dependent CUBE operation.
The reduction uses five FP32 slots instead of a running accumulator and one
term. Its phase-local UB is `2048*2 + 5*2048*4 = 45056` bytes (44 KiB). No
production communication or C2 source is changed.

[The installed selector/device audit](../tp16-reduction-order-source-audit-v1/README.md)
establishes the intended expression. It does not establish identical special
arithmetic for vector Add and HCCL atomicAdd. The explicit signed-zero,
subnormal, finite-overflow and BF16 tie inputs and their acceptance boundary
are in [floating-semantics.json](floating-semantics.json). The original order
witnesses must remain separate from those stress columns.

The root driver owns all device execution. It initializes SHMEM once through
the parent, prepares separate symmetric windows for both probes in the same
rank order, serializes their executions, destroys their graphs and windows,
then finalizes once. Binaries remain outside this archive; their paths, sizes
and hashes are in [binary-artifacts.json](binary-artifacts.json), and the exact
handoff is in [handoff.json](handoff.json).
