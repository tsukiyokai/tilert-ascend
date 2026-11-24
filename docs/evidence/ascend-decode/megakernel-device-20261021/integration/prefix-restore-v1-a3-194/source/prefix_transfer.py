"""Resume a receipt-bound prefix transfer using existing controller SSH access.

The controller relays bytes through pipes; it does not store tensor payloads.
The default is a plan. --execute enables the bounded transfer, never a model.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time


MANIFEST_SHA = "b01cceb2d2e30564807ce9efc441ad535999b56b73d0c333e93bb18694ccda74"
SOURCE = "/home/shanshan/trt-asc/offline/prefix8k-allocation-20261017/capture/case-00"
DESTINATION = "/home/shanshan/tilert-megakernel-20261021/checkpoint-owned/prefix8k-v3/case-00"
REMOTE = r'''
import fcntl
import hashlib
import json
import os
import pathlib
import sys
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
mode, root, name, size, expected, offset = sys.argv[1:]
size, offset = int(size), int(offset)
root = pathlib.Path(root)
path = root / name
owner = {'manifest_sha256': 'b01cceb2d2e30564807ce9efc441ad535999b56b73d0c333e93bb18694ccda74',
         'source': '/home/shanshan/trt-asc/offline/prefix8k-allocation-20261017/capture/case-00'}
marker = root / '.tilert-prefix-transfer.json'
receipt = {'path': name, 'bytes': size, 'sha256': expected, **owner}
partial = path.with_name(path.name + '.partial')
partial_receipt = path.with_name(path.name + '.partial.receipt.json')
def digest(value):
    h = hashlib.sha256()
    with value.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
if mode == 'init':
    if root.exists():
        assert marker.is_file() and json.loads(marker.read_text()) == owner
    else:
        root.mkdir(parents=True, exist_ok=False)
        with marker.open('x') as stream:
            json.dump(owner, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({'owner': owner, 'destination': str(root), 'pass': True}))
elif mode == 'state':
    assert json.loads(marker.read_text()) == owner
    if path.exists():
        observed = digest(path)
        assert path.stat().st_size == size and observed == expected
        print(json.dumps({'complete': True, 'offset': size,
                          'sha256': observed}))
    else:
        if partial.exists():
            assert json.loads(partial_receipt.read_text()) == receipt
        length = partial.stat().st_size if partial.exists() else 0
        assert 0 <= length <= size
        print(json.dumps({'complete': False, 'offset': length}))
elif mode == 'read':
    before = path.stat()
    assert before.st_size == size
    with path.open('rb') as stream:
        stream.seek(offset)
        remaining = size - offset
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise RuntimeError('Source became shorter during transfer')
            sys.stdout.buffer.write(block)
            remaining -= len(block)
    sys.stdout.buffer.flush()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns)
elif mode == 'write':
    assert json.loads(marker.read_text()) == owner
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + '.transfer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists()
    if partial_receipt.exists():
        assert json.loads(partial_receipt.read_text()) == receipt
    else:
        assert not partial.exists()
        with partial_receipt.open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    actual = partial.stat().st_size if partial.exists() else 0
    assert actual == offset
    with partial.open('ab') as stream:
        while block := sys.stdin.buffer.read(1024 * 1024):
            if stream.tell() + len(block) > size:
                raise RuntimeError('Transfer exceeds the receipt size')
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    actual = partial.stat().st_size
    observed = digest(partial)
    assert actual == size and observed == expected, (actual, observed)
    assert not path.exists()
    partial.rename(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(directory)
    os.close(directory)
    print(json.dumps({'path': str(path), 'bytes': actual, 'sha256': observed,
                      'resumed_from': offset, 'pass': True}))
else:
    raise ValueError(mode)
'''


def ssh(host, mode, root, item, offset=0):
    command = ["python3", "-c", REMOTE, mode, root, item["path"],
               str(item["bytes"]), item["sha256"], str(offset)]
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=10", host, shlex.join(command)]


def receipt_files(manifest):
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("Use the inspected complete TP16 v3 capture manifest")
    value = json.loads(raw)
    tokens = value['prompt_token_ids']
    prompt_sha = hashlib.sha256(json.dumps(
        tokens, separators=(',', ':')).encode('ascii')).hexdigest()
    if (len(tokens) != 8192 or prompt_sha != value['prompt_sha256']
            or sorted(row['rank'] for row in value['ranks']) != list(range(16))):
        raise ValueError('The TP16 prompt or rank identity changed')
    files = []
    for row in sorted(value["ranks"], key=lambda row: row["rank"]):
        for kind in ("main", "mtp"):
            record = row["files"][kind]
            relative = f"{kind}-prefix/rank-{row['rank']}.pt"
            if not PurePosixPath(record["path"]).as_posix().endswith("/" + relative):
                raise ValueError("The payload path differs from its receipt")
            files.append({"path": relative, "bytes": record["bytes"],
                          "sha256": record["sha256"]})
    if len(files) != 32 or sum(item["bytes"] for item in files) != 12404013472:
        raise ValueError("The exact 32-file prefix contract changed")
    files.append({"path": "manifest.json", "bytes": len(raw), "sha256": MANIFEST_SHA})
    return files


def transfer(item, args):
    key = item["path"].replace("/", "_")
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    state_command = ssh(args.destination_host, "state", DESTINATION, item)
    state_run = subprocess.run(state_command, capture_output=True, text=True)
    (args.output / f"{key}.state.stderr").write_text(state_run.stderr)
    if state_run.returncode:
        raise RuntimeError(f"Destination state rejected {item['path']}")
    state = json.loads(state_run.stdout)
    if state["complete"]:
        result = {**item, "reused_verified_file": True,
                  "target_sha256": state['sha256'], "transferred_bytes": 0,
                  "pass": True}
    else:
        read = ssh(args.source_host, "read", SOURCE, item, state["offset"])
        write = ssh(args.destination_host, "write", DESTINATION, item, state["offset"])
        with (args.output / f"{key}.source.stderr").open("wb") as source_log, (
                args.output / f"{key}.destination.stderr").open("wb") as target_log:
            source = subprocess.Popen(read, stdout=subprocess.PIPE, stderr=source_log)
            try:
                target = subprocess.Popen(write, stdin=source.stdout,
                                          stdout=subprocess.PIPE, stderr=target_log)
                source.stdout.close()
                target_output, _ = target.communicate()
            finally:
                if source.stdout and not source.stdout.closed:
                    source.stdout.close()
                source_status = source.wait()
        if source_status or target.returncode:
            (args.output / f"{key}.failure.json").write_text(json.dumps({
                **item, "start_utc": started,
                "end_utc": datetime.now(timezone.utc).isoformat(),
                "resume_offset": state["offset"],
                "source_status": source_status, "target_status": target.returncode,
                "pass": False}, indent=2) + "\n")
            raise RuntimeError(f"Transfer failed for {item['path']}; partial and logs retained")
        destination = json.loads(target_output)
        result = {**item, "destination": destination,
                  "target_sha256": destination['sha256'],
                  "transferred_bytes": item["bytes"] - state["offset"],
                  "source_status": source_status, "target_status": target.returncode,
                  "pass": True}
    result.update({"start_utc": started,
                   "end_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_seconds": time.monotonic() - start,
                   "source_receipt_sha256": item["sha256"]})
    (args.output / f"{key}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-host", default="a3-84")
    parser.add_argument("--destination-host", default="a3-194")
    parser.add_argument("--connections", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    args.output.mkdir(parents=True, exist_ok=False)
    files = receipt_files(args.manifest)
    plan = {"source_host": args.source_host, "source": SOURCE,
            "destination_host": args.destination_host, "destination": DESTINATION,
            "files": files, "connections": args.connections,
            "transport": "SSH controller relay; no local payload files",
            "authentication_changes": False, "model_or_npu_execution": False,
            "execute": args.execute}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"plan": str(args.output / "plan.json"), "executed": False}))
        return
    space = subprocess.run(
        ["ssh", args.destination_host,
         "python3 -c " + shlex.quote("import shutil; print(shutil.disk_usage("
                                    "'/home/shanshan/tilert-megakernel-20261021').free)")],
        capture_output=True, text=True, check=True)
    if int(space.stdout) < sum(item["bytes"] for item in files) + 1024 ** 3:
        raise RuntimeError("Reserve the complete prefix plus 1 GiB before transferring")
    (args.output / "space.json").write_text(json.dumps({
        "free_bytes": int(space.stdout),
        "payload_bytes": sum(item["bytes"] for item in files)}, indent=2) + "\n")
    initialized = subprocess.run(
        ssh(args.destination_host, "init", DESTINATION, files[-1]),
        capture_output=True, text=True)
    (args.output / "destination-init.stdout").write_text(initialized.stdout)
    (args.output / "destination-init.stderr").write_text(initialized.stderr)
    initialized.check_returncode()
    try:
        with ThreadPoolExecutor(max_workers=args.connections) as pool:
            results = list(pool.map(lambda item: transfer(item, args), files))
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({
            "error": str(error), "partials_preserved": True, "pass": False}, indent=2) + "\n")
        raise
    (args.output / "result.json").write_text(json.dumps({
        "files": results, "pass": all(row["pass"] for row in results)}, indent=2) + "\n")
    print(json.dumps({"verified_files": len(results), "pass": True}))


if __name__ == "__main__":
    main()
