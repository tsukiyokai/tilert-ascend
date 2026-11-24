# Complete FFN phase profile

This isolated variant instruments the accepted complete FFN build-v2. Thirteen
per-worker SYS_CNT reads delimit residual/norm, gate plus shared quantization,
TopK, body setup, the seven routed/shared phases, and output arithmetic. The
original six body barriers and all enclosing publication boundaries remain.
Trace stores occur after the final output publication. A separate INT64 trace
tensor is disjoint from every input, weight and workspace.

```sh
python3 build.py FROZEN_COMPOSED_BUILD_V2 NEW_OUTPUT_DIRECTORY
```

The build reuses the frozen parent's compiler arguments and retains source
hashes, commands and logs. Its CPU preflight loads both libraries without
initializing NPU. Device acceptance requires the same 30 numerical boundaries
as the ordinary FFN and valid traces from every resident worker.

The [M12/24 device gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-profile212-m12-v1-a3-194/summary.json)
passes 480 raw-byte comparisons against two references across eight changing
inputs. All 72 workers produce valid identities and monotonic 13-mark traces.
Each input has one recorded replay, so these windows do not measure timing
repeatability.

The decoder uses the A3 SYS_CNT rate of 50 ticks/us. Local windows include
work, synchronization waits and instrumentation; they exclude kernel launch
and final trace stores. Maxima from different workers must not be summed.
An ordinary paired event measurement is required for performance claims.

The [16-bank event comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-profile-timing-v3-a3-194/summary.json)
at M12/24 with 65 active experts measures 12.961/13.306 us of added latency
in the two allocation-owner arenas, with all 60 pairs slower than the ordinary
FFN. All 19,200 numerical boundary checks, 360 event-tail checks and 348
full-worker trace validations pass. This measures total instrumentation cost,
including trace stores outside the recorded local window; it does not assign
that cost to individual phases.

`build.py --sparse` samples only the body-setup boundaries, after TopK's
`EndPhase` and before the routed/shared body. It retains the 32-word record:
timestamp fields 0..3 repeat the first sample and fields 4..12 repeat the
second. `prepare_check.py --sparse` validates that layout and reports only
this interval. The logical trace-store footprint is unchanged; instruction
count and register allocation must be checked in the compiled device code.
This variant uses the same private profile namespace and must run in a
separate process from the 13-clock profile library.
