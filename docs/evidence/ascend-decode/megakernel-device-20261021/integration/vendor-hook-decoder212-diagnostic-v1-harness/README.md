# Decoder GMM1 synchronization diagnostic

This harness keeps the failed forward-v1 decoder and official operator bodies unchanged. The device driver changes only its helper import. It records module submission boundaries, synchronizes immediately before and after the original GMM1 call, and observes the original `dispose_tensor` call. Only weak references retain input Tensor identities; no extra NPU view or clone keeps the old storage alive.

The failing baseline is [forward-v1](../vendor-hook-decoder212-forward-int8-m1-v1-a3-194/). Its driver identifies stream 47, task 144, `GroupedMatmulSwigluQuantV2`, AIC 20, corrected instruction offset `0x3bd4`. The reported input is 49,152 bytes, corresponding to eight INT8 rows of width 6144. The weight-list dump size of zero does not establish a missing allocation.

The diagnostic saves all 256 group-list values, their declared count/prefix semantics, cumulative row bounds, input codes and small scale/assist tensors. Tensor descriptors include format, physical storage element count, addresses, strides, offsets, parameter owners and serialization metadata. The serialization metadata can describe an ND reconstruction; it is not substituted for the physical storage count. The original large weight tensors are not copied by Python.

A failure at the pre-call synchronization belongs to an earlier queued operation. A failure at the post-call synchronization occurs before the original input disposal and therefore excludes that later disposal as a necessary cause of this run. A successful synchronized run alone does not distinguish producer ordering from storage lifetime. The official host wrapper still owns its original workspace policy.

`device_diagnostic_run.sh` mounts a unique host evidence directory at `/tmp/extra-info` so exception tensors and kernel binaries survive container removal. The existing Torch/TorchNPU 2.12 image, CANN 9.2 mount, original extension and official Python package remain fixed. Root runs the single-device job; this artifact contains no device result.

The nine CPU checks cover interception of a real registered Torch CPU operator, both synchronization failure positions, restoration after failure, count bounds, and the original TP-cut checks. These checks do not validate NPU computation.
