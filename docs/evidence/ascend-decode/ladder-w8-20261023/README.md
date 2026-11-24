# The ladder on a baseline that passes: one rung is worth 0.14 ms, three never loaded

2026-10-23, a3-180, sixteen dies, `shared_fia` at width 8, 1024 output tokens.
Each rung adds one fusion to the arm that reads 14.743 ms and passes its gate.

| rung | TPOT | verify | host_tokens | in prompt | outcome |
| --- | --- | --- | --- | --- | --- |
| baseline | 14.743 ms | — | — | 0.338 | pass |
| `--hoist-metadata --rope-graph` | **14.605** | 2.138 | 50.454 | 0.338 | pass |
| vendor hooks applied, no flag | 14.739 | 3.538 | 49.763 | 0.338 | pass |
| `--routed-shared-kernel` | — | — | — | — | SIGABRT in the kernel |
| `--mla-tail` | — | — | — | — | needs physical ND latent |

## Hoisting the metadata is worth 0.14 ms, not 7.28

`handoff.md` priced it at 7.28 ms a round with a floor of 0.13. Measured on an
arm that passes, it moves TPOT from 14.743 to 14.605 -- 0.9 percent -- and the
output is identical in all three of its measures, so the work is the same and
only its issue is cheaper. That is the second lever this project priced by
estimate and found empty on measurement; `--narrow-bank` at 17.41 ms was the
first.

## Why the fusion rungs had never been measured

Three reasons, none of them the implementation:

- Their torch extensions were never built. `routed_int4_torch.so` and
  `mla_tail_torch.so` are absent from the tree, which is mounted read-only, so
  the arms died on `load_library`. Built out of tree against
  `TILERT_ROUTED_LIB_DIR`, `TILERT_ROUTED_SHARED_LIB_DIR` and
  `TILERT_MLA_TAIL_LIB_DIR`, all three libraries compile from current sources.
- `routed_shared.load()` required a comparison candidate. The active-list
  variant is reached only at 24 blocks and width 12, and its build asserts a
  kernel hash of `186b700c` against the current `26d142aa` -- a stale pin, and
  2 of its 13 hunks no longer apply, both in the kernel's expert loops. A
  library that exists to be compared is not a dependency of the arm, so
  `library_paths` now offers it only when it has been built.
- Both rungs need a vendor hook. `routed-shared-hook.patch` and
  `mla-tail-hook.patch` apply cleanly to a writable copy of the vendor tree,
  which `VLLM_ASCEND_TREE` already points the engine at. With the hooks in and
  no flag set, the arm reads 14.739 against 14.743 -- so the patches are inert
  and any later difference belongs to the fusion.

## What they say once loaded

`--mla-tail` raises "latent must use physical ND storage": it wants a cache
layout `shared_fia` does not use, so the two optimisations are mutually
exclusive as they stand.

`--routed-shared-kernel` aborts in the kernel after graph compilation, at
width 8. Its own namespace table keys the active variant to 24 blocks and
width 12, which is the shape the accepted library was validated at, so the
measurement moves to width 12 where the engine reads 22.135 ms on `sfa`.

## The routed/shared fusion faults in the AI core, at both widths

With the libraries built, the candidate made optional in both `library_paths`
and `namespace_name`, and the vendor hook applied, `--routed-shared-kernel`
reaches the device and faults there:

    dumpScene=aic_err_brief_dump, exceptionType=2(aicore),
    kernelName=_Z25routed_shar...   streamId=47, taskId=4209

Every rank reports it at the same task, at width 8 after graph compilation and
at width 12 after four minutes of decoding. So the obstacle is no longer the
environment: the kernel itself does not run against this engine's shapes.

Its own width-12 baseline, measured in the same session and on the same
patched vendor tree, reads TPOT 22.283 with verify 3.755 and host_tokens
52.974 -- which is the third independent confirmation that the round is device
time, the host holding 7 percent of it.

Where to resume: the exception dump names the kernel and a task id, and
`/root/ascend/log/debug/plog` on a3-180 holds the brief dump. The fusion is
worth 4.63 ms a round by the single-die fit, so it stays the largest measured
device-side lever that has never run.
