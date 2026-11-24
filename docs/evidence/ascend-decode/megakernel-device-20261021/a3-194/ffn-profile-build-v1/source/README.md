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

The decoder uses the A3 SYS_CNT rate of 50 ticks/us. Local windows include
work, synchronization waits and instrumentation; they exclude kernel launch
and final trace stores. Maxima from different workers must not be summed.
An ordinary paired event measurement is required for performance claims.
