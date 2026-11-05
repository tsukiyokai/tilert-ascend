"""Verify a task-owned supplement and read-only NAS through a daemon-owned volume."""

import argparse
import json
from pathlib import Path
import subprocess


IMAGE = "sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6"
NAS = "7.242.104.142"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owned", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.name.startswith("tilert-checkpoint-") or not all(
            char.isalnum() or char in "-_" for char in args.name):
        raise ValueError("Use a task-specific tilert-checkpoint-* name")
    owned = args.owned.resolve()
    evidence = owned / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    volume = args.name + "-nas"
    commands = [
        ["docker", "volume", "create", "--driver", "local",
         "--label", "tilert.checkpoint.restore=20261022", "--opt", "type=nfs",
         "--opt", f"o=addr={NAS},ro,nfsvers=3,soft,timeo=50,retrans=1,nolock,nosuid,nodev,noexec",
         "--opt", "device=:/", volume],
        ["docker", "run", "--rm", "--pull=never", "--name", args.name,
         "--read-only", "--network", "none", "--cpus", "1", "--memory", "1g",
         "--pids-limit", "64", "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m",
         "--tmpfs", "/checkpoint/view:rw,nosuid,nodev,noexec,size=2m",
         "--mount", f"type=volume,src={volume},dst=/checkpoint/nas,readonly",
         "--mount", f"type=bind,src={owned},dst=/checkpoint/owned,readonly",
         "--mount", f"type=bind,src={evidence},dst=/out",
         "--env", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python3", IMAGE,
         "/checkpoint/owned/tools/verify.py",
         "--manifest", "/checkpoint/owned/modelscope-files.json",
         "--nas-model", "/checkpoint/nas/model/GLM-5.2-w4a8c8",
         "--owned-model", "/checkpoint/owned",
         "--hash-file", "config.json", "--hash-file", "optional/quarot.safetensors",
         "--require-readonly-nas", "--require-no-devices",
         "--view", "/checkpoint/view/model", "--report", "/out/container-check.json"],
        ["docker", "volume", "rm", volume],
    ]
    exists = subprocess.run(["docker", "volume", "inspect", volume],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if exists.returncode == 0:
        raise ValueError("The task volume already exists; use a fresh name")
    result = {"image": IMAGE, "commands": commands,
              "scope": "CPU metadata and two explicit hashes; no weight-shard reads",
              "device_execution": False, "steps": []}
    report = evidence / "container-command.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    created = False
    try:
        for index, command in enumerate(commands[:2]):
            run = subprocess.run(command, capture_output=True, text=True)
            result["steps"].append({"step": index, "returncode": run.returncode,
                                    "stdout": run.stdout, "stderr": run.stderr})
            report.write_text(json.dumps(result, indent=2) + "\n")
            if run.returncode:
                raise RuntimeError(f"Checkpoint container step {index} failed")
            if index == 0:
                created = True
    finally:
        if created:
            cleanup = subprocess.run(commands[2], capture_output=True, text=True)
            result["steps"].append({"step": 2, "returncode": cleanup.returncode,
                                    "stdout": cleanup.stdout, "stderr": cleanup.stderr})
            result["volume_removed"] = cleanup.returncode == 0
            report.write_text(json.dumps(result, indent=2) + "\n")
            if cleanup.returncode:
                raise RuntimeError("The task NFS volume was not removed")
    print(json.dumps({"report": str(report), "volume_removed": True}))


if __name__ == "__main__":
    main()
