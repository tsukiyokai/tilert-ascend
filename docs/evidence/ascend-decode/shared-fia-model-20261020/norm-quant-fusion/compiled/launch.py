"""Run the bounded die-9 graph contrast and retain its terminal receipt."""

import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path("/home/shanshan/trt-asc-194/experiments/"
            "m32-full-teacher-20261020/norm-fusion/compiled-check")
EXPECTED = {
    "check.py": "9d76533e15bc710edd9814d938dcaa07fff68c7ea33b1da15efd1f0792de54a0",
    "run-pair.sh": "9913ea220eb197923bd6179371b9685ab2b862dc11621c9fbc2224060b6f7e7a",
}
for name, expected in EXPECTED.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
samples = []
for index in range(2):
    raw = subprocess.check_output(["npu-smi", "info"], text=True)
    (ROOT / f"device-01-preflight-{index}.txt").write_text(raw)
    rows = re.findall(
        r"\|\s+(\d+)\s+(\d+)\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)"
        r"\s+0\s*/\s*0\s+(\d+)\s*/\s*(\d+)", raw,
    )
    selected = [row for row in rows if int(row[1]) == 9]
    assert len(selected) == 1
    samples.append(selected[0])
    if index == 0:
        time.sleep(2)
passed = all(int(row[4]) - int(row[3]) >= 3072 for row in samples)
(ROOT / "device-01-preflight.json").write_text(json.dumps({
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "die": 9, "samples": samples, "passed": passed,
    "torch_quota_bytes": 512 * 2 ** 20,
    "scope": "Single-die compiled numerical contrast, no timing",
    "source_sha256": EXPECTED,
}, indent=2) + "\n")
if not passed:
    raise SystemExit("Insufficient device memory")
command = ["docker", "exec", "trt-asc", "bash",
           "/workspace/experiments/m32-full-teacher-20261020/"
           "norm-fusion/compiled-check/run-pair.sh", "device-01", "npu:9"]
with (ROOT / "device-01.log").open("x") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({"pid": process.pid, "command": command}), flush=True)
    status = process.wait()
(ROOT / "device-01-exit.json").write_text(
    json.dumps({"returncode": status}) + "\n",
)
print(json.dumps({"returncode": status}), flush=True)
raise SystemExit(status)
