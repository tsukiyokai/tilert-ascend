# Private TP tree layer: CPU build

The CANN 9.2 build and Torch 2.12 parent/candidate co-load passed without mapped
NPU devices or SHMEM initialization. The private library is
`7ec3e8fdd6155d99d98a9b69d2d4b02dc43c9417695f7632ad6559f42ecb96b3`.
This record establishes source, compilation, ABI and host ownership contracts;
the whole-layer device gate is separate.

- [Manifest](manifest.json): 200 frozen files, SHA
  `1f55dfc7ef1da565e2fac66cabe79dc040b557c998930c5864d8df754c5e9f38`.
- [Actual build report](results/build.json): SHA
  `ccbb7fd6d153f77f8c1104b39173501a6426d58181f4db3e3981e84b41d3e452`,
  5580 compiler dependencies, 68 SHMEM headers and 80 mapped dependency libraries.
- [Source audit](source-audit.json): all 165 local compute files match C2;
  inverse symbol replacement exactly restores the four layer files.
- [Patch](tree-integration.patch) and [compiler commands](results/commands.jsonl):
  the accepted tree files are copied byte for byte. Parent compile options and
  all five routing/Matmul definitions are retained; only dependency-output flags
  are appended. Linking changes the output filename.
- [CPU command](cpu-command.json): image
  `7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449`,
  one CPU, 8 GiB limit, read-only root, no network or mapped NPU devices.
  Torch is `2.12.0+cu130`, TorchNPU is `2.12.0`, C++11 ABI enabled.

Seven source checks, the parent's eight geometries, sixty rank/row geometries,
480 negative controls and the accepted reducer's 3840 finite plus 2160 special
CPU cases passed on ARM. The same contracts also passed on x86. The loaded
parent and candidate have distinct addresses for all ten exported C symbols,
independent host error registries and record sizes 664/928/48 bytes. Both
reject finalization before initialization.

The reducer keeps 2048 BF16 elements and five 2048-element FP32 stack slots:
4096 + 40960 = 45056 bytes. Both reductions begin after the parent's compute
phase resets the pipe. The first reduction's arena is drained/reset before FFN;
the last reduction retains the parent's final `PIPE_ALL` and DDR publication.
The symmetric window layout, current-stream ownership, epochs, three internal
joins, two reduction locations and external TopK copy contract are unchanged.

The parent is C2 `c2b72e46`; the reducer is the independently
[device-accepted private tree](../communication-tree-three-rank-v1-a3-194/AUDIT.md)
from probe `a0bd2462`. Full-layer arithmetic and performance still require their
own device evidence. Parent and candidate layer DSOs link one SHMEM runtime but
retain separate host context registries; a device process initializes only its
selected layer library, then destroys all graphs/windows before finalization.

[v1](../layer-tp-tree-build-v1/preflight.json) retained the CPU source-whitelist
failure before any Ascend compile. [v2](../layer-tp-tree-build-v2/results/kernel.log)
retained the compile rejection caused by replacing a namespace token inside
two include paths. v3 limits replacement to qualified identifiers and checks
include paths independently. Neither failed version produced a device result.

Objects and libraries remain outside the repository; their exact paths, sizes
and hashes are in [binary-artifacts.json](binary-artifacts.json).
