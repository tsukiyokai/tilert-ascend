"""Run the frozen private TP tree build on the CPU-only compiler host."""

import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


NAME = "layer-tp-tree-build-v2"
ROOT = Path("/tmp/tilert-megakernel-validation-20261021")
LOCAL = (ROOT / NAME).resolve()
TASK = "/home/shanshan/tilert-megakernel-20261021"
REMOTE = TASK + "/" + NAME
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes", "a3-78"]
subprocess.run(SSH + [shlex.join(["mkdir", REMOTE])], check=True)
manifest = json.loads((LOCAL / "manifest.json").read_text())
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
    for name in sorted(manifest["sources"]):
        archive.add(LOCAL / name, arcname=name)
    for name in ("manifest.json", "frozen-inputs.sha256", "preflight-x86.json"):
        archive.add(LOCAL / name, arcname=name)
subprocess.run(SSH + [shlex.join(["tar", "-xzf", "-", "-C", REMOTE])],
               input=buffer.getvalue(), check=True)
command = [
    "docker", "run", "--rm", "--pull=never", "--name", "shanshan-tilert-" + NAME,
    "--read-only", "--network", "none", "--workdir", "/tmp", "--cpus", "1",
    "--memory", "8g", "--pids-limit", "256", "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=1g",
    "-v", REMOTE + ":/candidate:rw",
    "-v", TASK + "/layer-tp-chain-build-v1/layer_tp_chain.so:/parent.so:ro",
    "-v", TASK + "/attention_vendor:/vendor:ro",
    "-v", "/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro",
    "-v", TASK + "/communication-deps/shmem:/shmem:ro",
    "-v", TASK + "/communication-deps/shmem-build/lib:/shmem-lib:ro",
    "-v", TASK + "/tests/glm52/whole_layer/communication_build-reuse/shmem_headers.sha256:/shmem-headers.sha256:ro",
    "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
    "-v", "/usr/local/dcmi:/usr/local/dcmi:ro",
    "--entrypoint", "/bin/bash",
    "sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449",
    "-c", "set -e\nsource /opt/ascend/cann-9.2.0/set_env.sh\n"
    "LD_LIBRARY_PATH=/shmem-lib:${LD_LIBRARY_PATH} PYTHONDONTWRITEBYTECODE=1 "
    "ASCEND_PROCESS_LOG_PATH=/tmp/ascend-log OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
    "python3 /candidate/source/build.py /candidate --cann /opt/ascend/cann-9.2.0 "
    "--vendor /vendor --shmem /shmem --libraries /shmem-lib "
    "--headers /shmem-headers.sha256 --control /parent.so\n",
]
(LOCAL / "cpu-command.json").write_text(json.dumps(command, indent=2) + "\n")
with (LOCAL / "cpu-launcher.log").open("w") as log:
    result = subprocess.run(SSH + [shlex.join(command)], stdout=log,
                            stderr=subprocess.STDOUT)
collect = (
    "import sys,tarfile;from pathlib import Path;p=Path(" + repr(REMOTE) + ");"
    "t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz');"
    "[t.add(f,arcname=f.name) for f in p.iterdir() if f.is_file() and "
    "(f.suffix in ('.o','.so','.log','.jsonl','.d') or f.name=='build.json')];t.close()"
)
raw = subprocess.check_output(SSH + [shlex.join(["python3", "-c", collect])])
results = LOCAL / "results"
results.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
    archive.extractall(results, filter="data")
print("Private TP tree CPU build exit", result.returncode, flush=True)
raise SystemExit(result.returncode)
