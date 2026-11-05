# Installed AIV reduction order

The installed CANN 9.2 deterministic AIV kernel uses a tree, whereas C2 adds rank payloads in ascending order. The CPU witness below distinguishes those expressions. This archive contains static binary/source evidence and CPU checks, not a TP16 device result. The separate three-rank device experiment uses the unchanged communication probe.

## Binary and selector

The files were read from a3-194, `/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0`. Full binaries and disassembly remain under `/tmp/tilert-megakernel-validation-20261021/tp16-reduction-audit/installed`; their hashes, commands, original paths and excerpt line numbers are in [manifest.json](manifest.json) and [installed-identities.txt](installed-identities.txt).

| Artifact | SHA256 |
| --- | --- |
| `libhccl_alg.so` | `9662caf979f889655d98278805fae701d29040b0b87a2fc21702ba11633c2956` |
| `hccl_aiv_all_reduce_op.o` | `deafbdd1659ff53b8c5f9ce1debad90f2e1a919a2cc5a419d781830b4cf54589` |
| C2 `communication_device.h` | `cf9e9a843c82ff77a4a4c2d6e4ff65024107cbd3c45044280c6502d65be8bde7` |

The host selector is stripped. Absolute PCs below identify its instructions; the nearest exported symbol printed by `objdump` is not the internal function name. The mapping is supported by the selector string, imported-function relocations, argument fields and matching source branches.

| Installed host PC | Condition or action |
| --- | --- |
| `0x5d3730–0x5d3740` | Require one superpod. |
| `0x5d3980–0x5d3998` | AIV enabled and ordinary AllReduce, excluding barrier sync mode. |
| `0x5d3ea8–0x5d3ed8` | Supported AIV dtype/reduce operation; deterministic enabled, or multiple servers. |
| `0x5d3edc–0x5d3ef4` | For ranks ≤8, payload ≤524288 bytes, unless `isOnlyAiv`. |
| `0x5d4370–0x5d437c` | For ranks >8, payload ≤8388607 bytes, unless `isOnlyAiv`. |
| `0x5d3ef8–0x5d3f2c` | Ranks >1, retry disabled, homogeneous module counts; a single server satisfies the additional inter-HCCS condition. |
| `0x5d38e0–0x5d38ec` | Select string `AllReduceMeshAivFor91093Executor` at `0x8f5b00`. |

Both size branches join the same remaining predicates and executor. For FP32 `[M,6144]`, M1/M2/M12/M32 use 24/48/288/768 KiB. Three ranks meet this branch for M1/M2/M12, but M32 exceeds its 512 KiB limit. Sixteen ranks meet the size predicate for all four widths. These static conditions do not establish an unobserved TP16 communicator's topology/retry state or its runtime selection.

Local explanatory source is `cann/hcomm` commit `2b4f2a4ed628cbeb48d76eead015ef220ea5400b`, under `src/legacy/ascend910/algorithm`. Its working-tree selector contains an existing HCCS guard absent from HEAD; both HEAD and that diff are archived. The installed selector contains the corresponding guard. This is a match of the decoded branches, not a claim that the entire local checkout produced the installed binaries. Source line numbers in installed log strings also differ.

## Executor to FP32 kernel

The installed executor uses `COMM_COMBINE_ORDER` (enum 9): `0x6b27a0–0x6b27d4` fetches that communicator. Source `topo_info_extractor.cc:386` preserves the rank-ordered vector; the device gathers each sender into the slot indexed by that rank.

At `0x6b2998–0x6b29a8`, the executor writes `argsType=2` (`ARGS_TYPE_SIMPLE`). Its deterministic getter branches to `0x6b2c50`, which writes 1 at `AivAlgArgs+8`; `0x6b29fc` calls the exported `ExecuteKernelLaunch` through relocation `0xaaed20`. That function tests argument type 2 at `0x592f68`, constructs the 136-byte V3 argument block at `0x593140`, and stores the deterministic field at block offset 132 (`0x593150`).

The installed registration table pairs `aiv_all_reduce_cn_float` with dtype 4 (FP32) and argument type 2. The entry's device symbol starts at `0x12d008`; `0x12d084` reads offset 132 and `0x12d098–0x12d0c4` branches on zero. The nonzero branch contains the tree below. Kernel registration alone is not evidence of a runtime-selected call.

## Tree and CPU counterexample

For `n` live rank slots, choose the largest power of two `p` strictly below `n`. For each `i < n-p`, assign `slot[i] = FP32(slot[i] + slot[p+i])`. Keep the other front slots unchanged, then set `n=p`. Thus a power-of-two rank count halves each round; a non-power-of-two count first folds only its excess ranks. No missing rank is replaced with an added zero.

Installed device anchors are `0x1308d4–0x13091c` (round count), `0x1309b0–0x1309c8` (strict-lower power), `0x130b00` (back index), `0x130f88–0x130fa4` (back index bound), and `0x131094–0x1310b0` (front/back payload offsets). The SUM branch enables FP32 atomic addition at `0x130fb0–0x131064` before copying the back slot into the front. The archived installed AscendC header maps `SetAtomicAdd<float>` to `set_atomic_add()` and `set_atomic_f32()`.

For 16 ranks the distances are 8, 4, 2, 1. For inputs `[1, 2^-25, -1, 2^-25, 0×12]`, ascending-rank FP32 additions give `2^-25` (`0x33000000`); this tree gives `2^-24` (`0x33800000`). Every input and both outputs are exactly representable in BF16. For three ranks, `[1, 2^-25, -1]` gives serial 0 and tree `2^-25`. Two ranks have only one addition and cannot distinguish these orders.

Run `python3 reproduce.py --output cpu-report.json`. The retained [CPU report](cpu-report.json) records every witness intermediate and every rank count's expression. Its 3,840 comparisons verify that a postorder stack matches the independent round-based calculation for ranks 2–16. Each sender occurs once, there are exactly `ranks-1` additions, and at most five FP32 vectors are live. This checks the proposed schedule, not vector-versus-atomic hardware equivalence.

## Minimal private candidate

Keep the existing SHMEM puts, epochs, ready signals, ownership, barriers and 2048-element tile assignment. Evaluate each tile's expression in postorder with five FP32 stack slots. A leaf loads one sender and casts BF16 to FP32; each scheduled addition consumes the last two live slots and retains its result in the left slot. The rank-16 leaf order is `0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15`, with the additions in [cpu-report.json](cpu-report.json). A left fold over that permutation is a different expression.

Each leaf retains the original MTE2→V fence and final V→MTE2 fence. Multiple additions after a leaf require a V barrier between producers and consumers. The final BF16 cast, GM store and MTE3→MTE2 reuse fence remain. Total arithmetic stays 16 casts and 15 additions per 16-rank tile. The narrow buffer occupies 4 KiB; five FP32 slots occupy 40 KiB. C2's attention functions return through `EndPhase`, and the FFN consumer has its own `EndPhase` before communication. Those reset the TPipe before the phase-local communication buffers are allocated; 44 KiB does not add to the existing computation phase's live UB footprint. Keeping 2048 elements avoids a tile-scheduling change. A 768-element reduction tile would fit in 19 KiB with the same push buffer, but increases tiles per row from 3 to 8 and is unnecessary for the current lifetime contract.

Matching parentheses does not establish equality over every BF16 bit pattern. C2 uses vector FP32 addition; HCCL uses FP32 atomic addition. The finite normal witnesses avoid underflow and overflow. Signed zero, BF16 subnormals that remain FP32 subnormals, and overflow from large finite BF16 operands require explicit arithmetic comparisons. The decoded atomic-control instructions and SDK wrapper do not establish FTZ, zero-sign or overflow equivalence. Missing subtrees must remain bit-preserving carries in any case. No production communication source is changed by this archive.

## Rank scaling

For `E=M*6144` elements, C2 submits `2*ranks*E` payload bytes per rank per AllReduce, including self-copy. Its off-rank payload is `2*(ranks-1)*E` bytes per rank. At M12, this rises from 144 KiB at two ranks to 2160 KiB at 16 ranks; the 16-rank aggregate is 33.75 MiB per AllReduce, or 135 MiB for the four reductions in two layers.

The decoded HCCL algorithm exchanges FP32 partitions, then broadcasts the reduced partitions. For these evenly divisible shapes, its source-level remote payload count is `8*E*(ranks-1)/ranks` bytes per rank. C2's BF16 all-to-all payload is therefore `ranks/4` times that count: half at two ranks, four times at 16. This excludes flags, framing, local GM traffic and topology effects; it is not a latency prediction. The measured two-rank gain does not establish a TP16 performance gain.

## Smallest runtime discriminator

Use three ranks on one A3 server, FP32 SUM on M1/M12, with command-local `HCCL_OP_EXPANSION_MODE=AIV`, `HCCL_DETERMINISTIC=true`, no `HCCL_ALGO` override, and persistent debug logging (`ASCEND_GLOBAL_LOG_LEVEL=0`, `ASCEND_SLOG_PRINT_TO_STDOUT=0`, `ASCEND_PROCESS_LOG_PATH=<evidence>`). Keep the original BF16→FP32 and FP32→BF16 conversions. Observe the unchanged communication probe's `first_sum`; its later CUBE and second AllReduce do not enter this comparison.

Require the actual selector log plus `ExecuteKernelLaunchInner` rank size, count, FP32 SUM and `deterministic=1`. That log includes argument size, not an `argsType` field; V3 size 136 is linked to type 2 by the host binary above. Preserve kernel/DSO identities and the distinction between registration and launch. Three-rank success establishes that small-rank execution; the >8-rank selector predicate and unmeasured TP16 topology remain separately stated.
