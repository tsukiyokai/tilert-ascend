# Sparse attention is bandwidth bound after all, on four passes and a third of the die

Read from the kernel source, then checked against two measurements that were
already on the board. It overturns the exclusion of "bytes" I recorded earlier
the same day.

## The operator that runs is a fork, and the template is V not C

`vllm_ascend/utils.py:454-458` prepends the custom vendor to
`ASCEND_CUSTOM_OPP_PATH` and `csrc/aclnn_torch_adapter/op_api_common.h:278-295`
resolves `aclnnSparseFlashAttention` from that tier first, so the vllm-ascend
fork runs rather than CANN's built-in. Corroborated by input arity: CANN's
entry takes a `sinks` tensor the fork has no parameter for, and the capture
shows nine inputs.

Tiling key 1158, `TEMPLATE_MODE = V_TEMPLATE`, decided at
`sparse_flash_attention_tiling.cpp:282-286` because `sparse_block_size` is 1
and the branch takes `≤ 4`. All 32 keys compile into each object, so only the
host tiling identifies the one that runs. The call site is
`device_op.py:565-582`, not `sfa_v1.py:224` — the latter passes
`query_rope=None`, which contradicts the capture's `8,4,64`.

Geometry: `s2BaseSize = 512`, `mBaseSize = gSize = 4`, so M is 4 and there are
2048/512 = 4 stages. `tiling.cpp:344` sets `usedCoreNum_ = aicNum_ = 24`
unconditionally, which is why the profiler reports 24/48 blocks.

## Eight rows run on eight core groups, and sixteen groups return

`kernel_mla.h:536-549`: `totalBaseNum` is the sum of the query lengths, 8;
since 8 ≤ 24 it sets `usedCoreNum = 8` and cores 8–23 return at `:549` and
`:765`. KV is never split across cores — `IS_SPLIT_G` needs `gSize > 64` and
`FLASH_DECODE` is hardwired to 0 in all eight `ASCENDC_TPL_ARGS_SEL` blocks.

**So 85.31 µs at one row and 89.17 at eight is not rows being cheap. It is one
row per group, eight groups in parallel, and a critical path of one row
either way.** The row sweep could not have revealed the cost, and my reading
of it — that the operator is row-independent and therefore not
data-proportional — was the wrong inference from a right measurement.

## Four passes over the selected KV, not one

| pass | unit | line | bytes a row |
| --- | --- | --- | --- |
| paged KV → UB | AIV MTE2 | `service_vector_mla.h:954/960` | 2.36 MB |
| UB → `kvMergeGm_` | AIV MTE3 | `:983/987` | 2.36 |
| `kvMergeGm_` → L1 for mm1 | AIC MTE2 | `service_cube_mla.h:646/652/668/674` | 2.36 |
| `kvMergeGm_` → L1 for mm2 | AIC MTE2 | `:935` | 2.10 |
| | | | **9.18** |

3.89× the 2.36 MB minimum, and 73.4 MB a call over eight rows.

## Which closes the arithmetic to within 3%

| | bytes | time less the 12.88 µs floor | rate |
| --- | --- | --- | --- |
| one row, eager | 9.2 MB | 72.43 µs | 127 GB/s |
| eight rows, eager | 73.4 | 76.29 | 962 |
| eight rows, in graph | 73.4 | 74.13 | **990** |

And my own gather benchmark, written to test a different hypothesis, measured
**956.7 GB/s** for a scattered 1152-byte read. Two independent routes to the
same number. The per-group rate is flat at 120–127 GB/s across an eightfold
change in load, which is a per-core-group MTE ceiling rather than a shared
HBM one — a shared ceiling would have degraded.
[the gather bench](../sfa-unexplained-20261019/README.md)

So the operator was bandwidth bound from the start. I excluded bytes because I
priced one pass across sixteen blocks; the kernel makes 3.89 passes on eight
of twenty-four core groups, which is 31× the bytes I counted.

## The missing UB→L1 path costs 2.86 ms a verify here

Passes two and three exist only because AIV gathers into UB and AIC needs the
data in L1, and this generation has no `UB → L1` route — the chip difference
table gives that row as 128 for David-121 and David V120/V100 and a slash for
1971, the same column that lacks `L0C → UB`.
[the paths](../cube-vector-path-20261019/README.md)

That round trip is 4.72 MB a row, **51% of the traffic**, and over eight rows
and 64 layers it is 2.42 GB a verify — 1.57 ms at the die's marginal rate,
**2.86 ms at the 844–990 GB/s this kernel achieves**.

Earlier I measured the cube-to-vector direction at 1.364 µs a 32 KiB tile and
concluded the missing on-chip path was worth 0.87 ms a round, 1.7%, and
therefore not the lever. That held for the direction and granularity I
measured. In the other direction, at this operator's volume, the same absence
costs 2.86 ms in one operator.

## Everything else is small, and accounted

Cube MAD is 160 `Mmad` at m=16 — 4 real rows padded to 16 — for 8704
MAC-cycles, about 4.8 µs of which 3.6 is padding. Vector arithmetic about 3.3
µs. Cross-core sync is 33 handoffs and 66 `CrossCore*` instructions a call
(`kernel_mla.h:742/757/758/874/881/892`, `service_vector_mla.h:1109-1173`,
`service_cube_mla.h:1099`), bounded at 1.6–6.6 µs and designed to hide behind
the next stage's MTE2 by a three-deep preload. Zero `SyncAll` executes — the
only one is gated on `LAYOUT_T != TND`. Zero `PipeBarrier<PIPE_ALL>` anywhere.
A `PipeBarrier<PIPE_M>` fires after every `Mmad` because `(16/16)*(128/16) = 8
< 10` at `service_cube_mla.h:818-819`, about 1.3 µs — and it would vanish at
`mBaseSize = 32`.

Sum ≈ 14 µs, leaving ~60 µs of MTE traffic out of 73.43.

## Two levers, both structural

**`sparse_block_size` above 4 flips the template.** At 1 the V path pays the
merge workspace round trip; C gathers straight into L1 with one `DataCopyPA`
per token (`service_cube_mla.h:681/940`), which at 1 would be 16384 single-row
strided descriptors a row — which is why `tiling.cpp:282` routes `≤ 4` to V.
Above 4 each descriptor carries that many contiguous slots.

**Three quarters of the die is idle, and split-KV is dead code.**
`sparse_flash_attention_tiling.cpp:349` declares `bool splitKVFlag_ = false;`
and no line in the file ever sets it true — it is read at `:316`, `:385` and
`:416` and written nowhere. So the mechanism that would spread 2048 KV
positions across cores is not an unmet condition but an unfinished path in
this fork, and `:316` additionally excludes TND even if the flag were set.
Opening it means implementing it, not configuring it.

Eight of 24 cube cores and 16 of 48
vector cores do the whole call, and no compiled tiling key changes that for a
shape with eight query tokens: splitting KV across cores needs `FLASH_DECODE`,
which no key enables.
