# Frozen serial/tree comparison fixture

The unchanged root driver, accepted serial probe and private tree probe were
frozen together and verified in image
`1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6`
on a3-194. The actual CPU gate passed without mapped or initialized NPU devices.
This archive contains that fixture and CPU gate; root owns the separate device
execution and its numerical acceptance.

| Artifact | SHA256 |
| --- | --- |
| Fixture manifest | `18d57bafa45fb629965916d0d380e014686dd322c575c469ee6327caa4cc98e8` |
| Original driver | `9973741bbf0ecb5ab330607733ad8895d68bd8d8a38904e58dea572496c0d5b3` |
| Accepted serial DSO | `1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f` |
| Private tree DSO | `a0bd2462e2f999280720fa5060466159a2eab13ce5541639ddc7afef35a45ebf` |
| Successful CPU report | `30898f1ca654512e7ae62247388a143929c272ee049ff20f2fe63d30ccc52348` |

The [successful report](cpu-v2/report.json) verifies 43 frozen inputs, the 22
build sources, all six private exports, the original driver CPU contract and
80 mapped runtime dependencies. Parent and candidate resolve
`aclshmemx_init_status` to the same function address. No SHMEM initialization
was called by the CPU gate. Device execution must initialize/finalize once and
keep the probes' two windows and graphs independently owned.

CPU v1 stopped at the dependency identity check because its launcher omitted
the host driver library mounts, causing an SDK devlib HAL stub to load. CPU v2
adds the same read-only driver and dcmi mounts used by the build and device
launcher; the fixture, driver and DSOs are byte-identical. Both commands and
logs are retained, with the exact comparison in [cpu-summary.json](cpu-summary.json).

The immutable package is at
`/home/shanshan/tilert-megakernel-20261021/communication-tree-device-v1` on
a3-194. Binaries are excluded from this text archive; their paths and hashes
are recorded in the summary. [handoff.json](handoff.json) gives the driver
arguments and required HCCL environment. The [frozen manifest](fixture/manifest.json)
describes the complete package, including those external binary files.
