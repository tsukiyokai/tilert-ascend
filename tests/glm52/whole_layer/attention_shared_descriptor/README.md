# Shared-attention argument descriptor

This candidate prepares the device argument descriptor on the host before the
single shared-indexer attention launch. Prolog, cache writes, SFA, value-up,
dynamic A8 quantization, output projection and their synchronization retain the
frozen parent's arithmetic and geometry. Input normalization and TP reduction
remain outside this entry.

`prepare.py` verifies all 26 parent source hashes and the Torch NPU 2.12 parent
library `64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c`.
The public Torch schema and host launch signature remain unchanged. The private
kernel receives a 232-byte descriptor, including two precomputed arena
addresses, instead of building it from 27 pointers on device. This adds 16 bytes
to the kernel argument pack and allocates no additional GM or UB storage.

The CANN 9.2, dav-2201 candidate library is
`ff36c603205f43420b2181ae7193a3b6b4b8a3a2445db84a3cf74840664c4d64`.
The [frozen build](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-descriptor-build-v1/build/manifest.json)
records source and artifact hashes. CPU checks cover the original host parameter
order, the unchanged arithmetic statements and all 768 supported row/group
descriptor layouts. Both actual libraries load together under Torch NPU 2.12.

The [device comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-descriptor212-wide-v1-a3-194/summary.json)
passes M=1,2,12,32 with 16 and 24 mixed core groups at an 8192-token context.
Eight changing inputs per geometry produce 16 captures, 128 graph replays and
256 exact-byte comparisons of output, latent attention and both complete
caches. Output and latent buffers are poisoned before replay. This comparison
uses synthetic weights and FP32 projection scales against the frozen native
parent.

`bench.py` captures both libraries on the same 32 Torch arguments, repeats the
comparison on each implementation's prepared arena, and rotates 16 distinct
weight banks. Each event replays a 16-call graph 30 times. Thirty alternating
event pairs per arena retain per-bank checks before and after timing, plus
output, latent and complete-cache checks after every event. Input copies and
poisoning execute before the start event. The logical weight working set
exceeds the recorded 192 MiB L2 capacity; this is not a measurement of HBM
traffic. `cpu_preflight.py` loads both actual libraries and checks their schemas
and paired caller without initializing an NPU.

The [M12/24 paired result](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-descriptor212-timing-v1-a3-194/summary.json)
regresses by 2.201 and 3.022 us in the two owner arenas, with zero wins in
60 pairs. Candidate/control medians are 152.690/150.412 us and
152.197/149.151 us. All 1,152 per-bank and 480 event-tail raw-byte checks
pass. The frozen original shared-attention implementation remains selected.
