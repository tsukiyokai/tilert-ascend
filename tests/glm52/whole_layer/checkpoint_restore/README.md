# GLM-5.2 checkpoint recovery

On 2026-10-22, the NAS checkpoint remained available at `7.242.104.142:/model/GLM-5.2-w4a8c8`. Its 108 existing files have the sizes in the original manifest. The missing `optional/quarot.safetensors` has been restored separately on a3-194. Loss of the old container did not imply loss of this NAS source.

The checkpoint is [Eco-Tech/GLM-5.2-w4a8c8 at revision 67e2b715](https://modelscope.cn/models/Eco-Tech/GLM-5.2-w4a8c8/files?Revision=67e2b7157ac7d8ce1d15d7251853ce146819ceb9). A fresh fixed-revision API response matches all 109 original `(path, size, SHA-256)` records. The [recovery evidence](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/checkpoint-restore-v1-a3-194/summary.json) records the metadata, commands, failed attempts, and current verification.

| Object | Bytes | Current SHA-256 check |
| --- | ---: | --- |
| NAS `config.json` | 3,699 | `817f5fb39ca5d4c4b5648de89ca00deaea7537d8c2f130172a459252a05c1073` |
| Owned `optional/quarot.safetensors` | 150,995,040 | `960e0ca37d0e58327cad7f9cbc9b1cb6822479a07af0c6153ea541c6de0fc7c0` |
| 95 NAS weight shards | 404,718,618,504 | Sizes checked; contents not read or rehashed in this recovery |

The composite view contains all 109 files, totaling 405,008,283,962 bytes. All sizes and 109 symlink owners passed inside the consumer container. This check read 150,998,739 bytes for the two listed hashes. The earlier 2026-10-15 full-copy hash verification is historical evidence; it is not a current full checkpoint integrity check. No model or NPU was started.

## Owned file and container paths

The recovered file is retained at:

```text
a3-194:/home/shanshan/tilert-megakernel-20261021/checkpoint-owned/optional/quarot.safetensors
```

The local verified copy is `/home/shanshan/.cache/tilert-checkpoint-restore-20261022/owned/optional/quarot.safetensors`. The download used aria2 with 16 connections, checksum verification, and a command-local proxy; proxy configuration and persistent routes were unchanged.

`container_check.py` creates a task-specific Docker local NFS volume, mounted read-only by the Docker daemon. It binds the supplement read-only and creates the model view inside that same container namespace:

| Container path | Owner |
| --- | --- |
| `/checkpoint/nas/model/GLM-5.2-w4a8c8` | Read-only NAS export |
| `/checkpoint/owned` | a3-194 `checkpoint-owned` directory, read-only |
| `/checkpoint/view/model` | Ephemeral symlink view of the 108 NAS files and one owned file |
| `/out` | Task evidence directory, writable |

The validation container had one CPU, no mapped accelerator devices, and no network namespace access. Its image was `1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6`. All three steps—volume creation, verification, and volume removal—completed successfully. The view ceases to exist when the container exits. A later loader must retain both mounts and construct its own view in its own container; a private host mount or a symlink to a path absent in that container is insufficient.

To repeat the bounded validation on a3-194, use a fresh task name:

```bash
python3 /home/shanshan/tilert-megakernel-20261021/checkpoint-owned/tools/container_check.py \
  --owned /home/shanshan/tilert-megakernel-20261021/checkpoint-owned \
  --name tilert-checkpoint-verify-v2
```

`verify.py` checks sizes by default. Each `--hash-file` explicitly names one manifest entry to read and hash. It rejects unknown names, size changes, unstable file metadata, an incorrectly resolved view, and a writable NAS when `--require-readonly-nas` is set. It does not automatically hash the 95 shards.

## Repeating the missing-file download

`restore.py` accepts the pinned original manifest and the saved official fixed-revision response. It checks all records before downloading only the missing optional file. It resumes a partial file, verifies the completed content independently, and atomically renames it. An existing complete file must pass size and SHA-256 verification before reuse. Use a disk-backed destination; the first attempt on the local `/tmp` tmpfs exhausted its space and was preserved under the local recovery cache.

```bash
python3 tests/glm52/whole_layer/checkpoint_restore/restore.py \
  --manifest docs/evidence/ascend-decode/megakernel-device-20261021/integration/checkpoint-restore-v1-a3-194/modelscope-files.json \
  --official-files docs/evidence/ascend-decode/megakernel-device-20261021/integration/checkpoint-restore-v1-a3-194/official/official-files.json \
  --output /home/shanshan/.cache/tilert-checkpoint-restore-20261022/owned \
  --evidence /home/shanshan/.cache/tilert-checkpoint-restore-20261022/recheck
```

The CPU contracts reject same-size corruption when hashing is requested, reject incorrect sizes even when a supplied hash matches, and verify that the view retains its two distinct owners. Current recovery evidence is separate from full-model loading, dependency compatibility, and steady-state inference validation.
