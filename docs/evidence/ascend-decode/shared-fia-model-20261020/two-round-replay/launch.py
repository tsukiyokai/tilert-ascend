"""Run one frozen replay and retain its authoritative process exit."""
import json
from pathlib import Path
import subprocess
import sys

root = Path("/home/shanshan/trt-asc-194/experiments/m32-full-teacher-20261020/two-rounds")
layers, case = sys.argv[1:]
command = ["docker", "exec", "trt-asc", "bash", "/workspace/experiments/m32-full-teacher-20261020/two-rounds/run.sh", layers, case]
with (root / (case + ".log")).open("x") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
(root / (case + "-launch-exit.json")).write_text(json.dumps({"returncode": result.returncode}) + "\n")
print(json.dumps({"returncode": result.returncode}), flush=True)
raise SystemExit(result.returncode)
