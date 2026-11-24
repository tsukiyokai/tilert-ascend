# The library wrapper was a third of the push

Same binary, one variable, two interleaved repeats at TP16 on a 96 KiB float32
payload:

| push mechanism | per reduction | spread | payload |
| --- | --- | --- | --- |
| `aclshmemx_mte_put_nbi` + `aclshmem_quiet` | 22.556, 22.405 µs | 0.7% | `wrong 0`, `got 120.0` |
| `aclshmem_ptr` + plain `DataCopy` | **14.623, 14.481 µs** | 1.0% | `wrong 0`, `got 120.0` |

**1.54×**, and against the vendor's 63.5 µs in-graph all_reduce, 4.38×.

## Where it came from

Reading `examples/shmem_mega_moe`, the SHMEM tree's own 4896-line megakernel,
which fuses token routing, cross-rank dispatch, two expert GEMMs, SwiGLU and
the combine into one device pipeline. Its entire use of the one-sided API is
three calls — `aclshmem_ptr`, `aclshmem_my_pe`, `aclshmem_n_pes`. No
`put_nbi`, no `signal_op`, no `quiet`.

It communicates like this
(`kernel/arch35/shmem_mega_moe_combine_pipeline.h:158`):

```cpp
remoteOutput.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(remoteAddress));
AscendC::DataCopyPad(remoteOutput[destinationOffset], quantizedToken, rowCopyConfig);
```

with the address from `shmem_mega_moe_types.h:224`:

```cpp
return reinterpret_cast<GM_ADDR>(aclshmem_ptr(localBase, rankId)) + offset;
```

Symmetric memory is simply addressable. Once `aclshmem_ptr` maps our pointer
to the peer's window at the same offset, a remote write is an ordinary MTE
store — no library call, no staging protocol, no fence. I had been assembling
a protocol out of the SDK's wrapper functions, and that wrapper cost 8 µs of
the 22.3.

## What is still in the 14.5 µs

The direct path keeps `PipeBarrier<PIPE_ALL>` between its two `DataCopy`
calls, and `tp_push_once` still issues `aclshmem_quiet` and
`aclshmemx_sync_vec_all` afterwards. The official example uses neither: it
pipelines with paired `SetFlag`/`WaitFlag` on specific hardware events
(`MTE3_V`, `MTE3_MTE2`, `MTE2_V`, `V_MTE3`) across a `PIPELINE_BUFFER_COUNT`
double buffer, with explicit prologue and steady phases. A full-pipeline
barrier where the official code places one event is the next thing to remove.

## The structural lesson, which is larger than the 8 µs

The official megakernel overlaps communication with compute by **partitioning
cores**, not by interleaving on one core
(`kernel/arch35/shmem_mega_moe_kernel.h`, `Process()`):

```cpp
if (vectorSubBlockIndex_ == 1) { BuildRoutingMetadataAndDispatch(...); }  // communication
if (vectorSubBlockIndex_ == 0) { RunFirstProjection<...>(...); }          // compute
```

Half the vector cores are a communication engine for the kernel's whole life,
the other half drive the cube, and flags couple them. That is what "continuous
overlap" means in practice, and it is why a standalone collective cannot have
it: a separate kernel has no cores left running the compute it should hide
behind.
