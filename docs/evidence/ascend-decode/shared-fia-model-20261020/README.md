# M32 shared-FIA model validation

Current result: [TP16 M32/SFA, all78 layers](layer-major/m32/full/README.md)
passes32/32 frozen labels with the FP32 head. The same-source shared-FIA
full arm completes30/32, failing queries3 and21; final norm cross-arm
NRMSE is0.15993. Teacher acceptance is unchanged. Earlier interrupted
whole-resident attempts follow.

Source: `fba76ee41eeb236265d806856e938f5134c1446b`. Intended comparison:
serial SFA, eager M32 shared FIA, Graph, and dirty reset; 8K prefix and
33 outputs include the prefill token plus 32 frozen successor labels.

The a3-187 run started at 01:23:55 on October 20, with all 16 dies idle.
Docker recorded SIGTERM at 01:25:59, SIGKILL at 01:26:19, and container
stop/die/destroy at 01:26:21, exit137. The last model log was checkpoint
prefetch; no rank report or model result was produced. The host-mounted
workspace survived at `/home/shanshan/trt-asc-187`. This is an interrupted
run, not a numerical result.

Evidence: [launcher](validate-m32.sh), [log](validate-m32.log),
[Docker events](docker-events.jsonl), [source manifest](source-manifest.json),
[teacher binding](teacher-binding.json).

The teacher prompt hash `d70449c0…` and capture hash `fe53282d…` encode the
same 8192 IDs: default JSON plus newline versus compact JSON. Compare the
actual IDs and both serializations, then require outputs `[:32]` to match
teacher inputs and `[1:33]` to match all 32 successor labels. A local
serial/speculative pass alone does not establish that comparison.

## CPU findings

The old `chain-w8` requests3/4/5 first differ at output offset4:6500 versus
teacher2524. Each has27/32 mismatched successor labels; after the first
branch, later free-running outputs no longer cover teacher-forced inputs.
The source declares FP32 head and its head hash matches the current source.
The cause is not established. [Raw comparison](old-chain-teacher-witness.json)
and [prompt identity](teacher-binding-witness.json) keep these claims separate.

`55ed217` requires capacity for the complete final verification and poisons
the corresponding tail. At P8192/B33/M32, C8254 previously admitted a final
M24 call; C8255 is the boundary. Server CPU checks passed168 cases plus32
field/range comparisons at widths8/32. No NPU was initialized.
[Runner checks](cpu/checks.json), [tail checks](cpu/poison.json).

`d62e1b2` adds an optional fixed-teacher stage to the existing runner. Both
M1 and M32 consume all32 frozen inputs from a freshly injected prefix,
including after a wrong prediction. They check prefix preservation and save
the first wrong row's logits, normalized hidden, top-two margin and expected
token score. Teacher acceptance and serial/speculative consistency are separate
results. Server CPU passed169 checks, including short-label, missing-prediction
and false-argmax cases; the real teacher file also passed input binding.
[Checks](teacher/teacher-host-checks.json), [CLI/binding](teacher/teacher-cli-check.json).

Optional `--teacher-routes` saves actual expert IDs as `[32,75,8]`, bound
to the frozen inputs, positions and collector/router source hashes. Per
forward it reports active experts and the largest expert group; it does
not alter the teacher label gate. Server CPU passed34 collector cases and
169 runner cases, including route ownership/order, shared versus dispersed
experts and preservation of the first numerical difference.
[Collector](routes/choices-check.json), [runner](routes/teacher-check.json).

Teacher-forced routes diagnose the same32 inputs. Lookup verification also
processes rejected candidate suffixes, whose experts contribute to cost;
measure those actual candidates before selecting a wide-kernel policy.
The historical78-layer M1 capture has hidden/residual/index tensors but no
expert IDs. Its residual alone is not a verified reconstruction of the
router's normalized input. No full-model M32 routing result exists yet.

The existing `capture_cann_routes.py` now supports M24/M32 and lookup.
It records actual candidate IDs, positions and `[M,75,8]` routes, using
persistent copies during Graph replay. Its summary TPOT is null. Nine
server CPU fixture cases pass: changing replay/reset, input ownership,
two simulated ranks and missing/stale writes. This validates the wrapper
logic; real NPU replay remains pending. [CPU receipt](routes/result.json).

Committed source `7747b5c` and two separate launchers are prepared under
`/workspace/experiments/teacher-routes-20261020` on a3-194:
[teacher M1/M32](routes/teacher-validate-m32.sh), then
[actual lookup M32](routes/capture-lookup-m32.sh). They use separate cache
roots/working directories/ports and the original verified GMM overlay.
Neither has been launched or queued.
Both actual launcher CLIs pass against the committed snapshot without importing
torch_npu. [CLI receipt](routes/launchers-cli.json).
At04:30 on October20, all16 dies on each of a3-gs/78/84/162/194/187 still
hold58,095–61,650MiB. No whole-host model run was launched.

## Prepared environment

a3-194 `trt-asc` uses CANN9.2.0 B114, torch/torch-npu2.12.0 and the same
vLLM source pair as a3-84. CPU imports,96 weight shards, prefix hash and
patched GMM header/ELF hashes passed. The HIXL library and its dependencies
are present; no device or live-PD validation was run on this host.
[Environment receipt](environment194/cpu-ready.json).

At01:58 on October20, another user's Ray workers occupied all16 dies,
HBM-Usage48259–48516/65536MB per die. The model command is prepared at
`/workspace/experiments/shared-fia-20261020/teacher-validate-m32.sh` and has
not been launched or queued. The host workspace is
`/home/shanshan/trt-asc-194`.

## Static packages

The historical M8 race was closed by the SwiGLU barrier and correct static
package provenance. The same cache key held an old GMM ELF with six distinct
hidden results in six identical forwards and a patched ELF with one; the
patched cache remained identical over20 repeats. All16 ranks agree on these
counts. This is historical evidence, not a new M32 run.

CANN caches static packages below cwd; its key does not include vendor
header/ELF identity. `587cabf` gives each timing arm its own cwd and calls the
same source by absolute path. Four actual-launcher CPU cases cover both
attention choices and absolute/relative input paths. Runtime activation,
precheck and torchrun are mocked in those cases; two server CPU imports use
the real benchmark. [Audit](cache/summary.json),
[launcher checks](cache/launcher-cpu-receipt.json),
[server imports](cache/cwd-cpu-receipt.json).
