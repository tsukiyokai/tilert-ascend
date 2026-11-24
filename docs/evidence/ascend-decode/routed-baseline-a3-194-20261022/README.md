# A verifiable routed baseline on a3-194, and the PYTHONPATH that blocked it

The routed body's own check chain now runs on a3-194, so a change to that kernel
can be compared byte for byte against the unmodified one. That was the missing
piece for porting the measured skew into the expert loop.

## The baseline

Four changing-route replays, two weight banks, four calls, twelve rows, 24
blocks, one die:

    every replay   bitwise_equal, max_abs 0.0, 0 differing bytes
    active experts 8, 8, 87, 81
    median a call  native 167.08 us   vendor 268.26 us

`torch 2.12.0+cpu`, `torch_npu 2.12.0`, CANN 9.2.0 throughout, on
`Ascend910_9392` with 24 cube and 48 vector cores and 192 MB of L2.

## The recipe

Build and run in `trt-asc:a3-84-unified-20261018`, which carries the full CANN
at `/opt/ascend/cann-9.2.0` and the `/opt/trt-asc` venv:

    source /opt/ascend/cann-9.2.0/set_env.sh
    export PYTHONPATH=<tree>/tests/glm52/routed_int4:<tree>:$PYTHONPATH
    export XDG_CACHE_HOME=/tmp/cache
    bash tests/glm52/routed_int4/build_nz_reference.sh
    ROUTED_ROW_BALANCED=1 bash tests/glm52/routed_int4/build_balanced.sh <out>
    python3 tests/glm52/routed_int4/bench_weight_banks.py \
        --library <out>/routed_int4_torch.so --native-schedule row \
        --device npu:0 --rows 12 --blocks 24 --banks 2 --calls 4 \
        --iterations 10 --timing-pairs 2 --output <report>

## What blocked it, and two corrections it forces

Every attempt failed with `[Set][Options]OpCompileProcessor init failed!` until
the Python traceback was read rather than the Ascend log's tail. The real line
is `ModuleNotFoundError: No module named 'tbe'`: GE's initialization imports
CANN's op-compiler frontend, which `set_env.sh` puts on `PYTHONPATH`, and
`export PYTHONPATH=...` **overwrote** that instead of prepending to it.

So two entries in this project's record are wrong. `OpCompileProcessor init
failed` was listed as one of four machine-side failure modes, an environment
fault of its own; it is the downstream symptom of a clobbered `PYTHONPATH`. And
the `/opt/trt-asc` venv was recorded as absent from the image; the traceback
shows it at `/opt/trt-asc/venv/lib/python3.11/site-packages`.

The stripped fixture image `sha256:1910b1ac` cannot run this chain at all, with
or without the OPP paths, because it has no `tbe`. Its own fixtures avoid GE.
Along the way `allow_internal_format`, `npu_grouped_matmul` and
`npu_format_cast` were each confirmed working there, which is how the search
narrowed to the import.

[`routed-baseline.json`](routed-baseline.json) is the report.
