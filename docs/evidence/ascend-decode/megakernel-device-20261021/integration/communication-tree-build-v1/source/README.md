# Private FP32 tree reduction probe

This candidate replaces the accepted communication probe's rank-ascending FP32
sum with the deterministic AIV tree observed in the installed CANN 9.2 runtime.
Its communication transport, 2048-element tiles, epoch protocol, CUBE stage,
barriers and final BF16 cast remain unchanged. It does not modify the production
communication helper or the accepted C2 library.

The parent DSO is `1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f`;
its communication header is
`cf9e9a843c82ff77a4a4c2d6e4ff65024107cbd3c45044280c6502d65be8bde7`.
The source, selector and device instruction evidence are in
[the pinned reduction-order audit](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/tp16-reduction-order-source-audit-v1/README.md).

For `n` live branches, each installed round selects the largest power of two
strictly below `n`, adds paired back branches into front branches, and carries
unpaired front branches unchanged. `tree_plan.h` evaluates this expression in
postorder with one leaf load and zero or more stack additions per encoded byte.
It covers ranks 2 through 16, including non-powers of two. Each input is loaded
once and there are exactly `n - 1` FP32 additions. Sixteen ranks require at most
five simultaneous FP32 vectors. No zero seed or intermediate BF16 rounding is
introduced.

The AIV phase allocates `2048 * 2 + 5 * 2048 * 4 = 45056` bytes (44 KiB).
This replaces the parent's 20 KiB allocation. In C2 the communication phase
begins after the compute arena has been drained and reset, so this phase-local
size is below the accepted compute peaks. The private probe retains its own
existing phase lifecycle; this statement is not a C2 integration result.

The private C ABI is `communication_tree_probe_` followed by `initialize`,
`prepare`, `execute`, `destroy`, `finalize`, or `last_error`; its parameter layout
matches the parent. A combined test initializes SHMEM once through the parent,
prepares two distinct symmetric windows in the same rank order, serializes all
launches, resets every graph before destroying its window, then finalizes once.
Both DSOs must resolve the same recorded SHMEM dependencies.

CPU contracts compare the actual C++ plan against an independent round
recurrence, including expression identity, unique leaves, stack liveness, 3840
finite BF16-valued inputs, and 2160 signed-zero/subnormal/extreme CPU cases. The
last cases establish CPU expression equivalence only. Installed HCCL uses FP32
atomic addition; this candidate uses vector `Add`. The source and instruction
audit does not establish identical zero signs, denormal handling or overflow
behavior for those two hardware operations.

Device comparison must retain the original order witnesses and separately
report positive/negative zeros, BF16 subnormals, exact cancellations, finite
inputs whose sums overflow, and final BF16 ties. Inputs stay finite, and extreme
columns must not be scaled. Only the first reduction and its immediate copy
are comparison boundaries: the probe's later CUBE stage may legitimately
produce nonfinite values for these stress inputs. A successful ordinary-input
gate does not imply that the special-value gate passed, or that TP16 was run.

Freeze with `prepare.py ACCEPTED_PROBE_DIRECTORY NEW_BUILD_DIRECTORY`.
`build.py` verifies parent, frozen sources and SHMEM dependencies, compiles with
CANN 9.2, records compiler dependency files, and co-loads the original and
private C ABIs without mapped NPU devices. Device execution belongs to the
separate three-rank driver.
