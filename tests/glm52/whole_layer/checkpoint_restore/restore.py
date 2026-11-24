"""Recover only the missing rotation object from the pinned GLM checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.parse


REPOSITORY = "Eco-Tech/GLM-5.2-w4a8c8"
REVISION = "67e2b7157ac7d8ce1d15d7251853ce146819ceb9"
MANIFEST_SHA = "100f774ead2a54d309b86faae1ffe0d9b5e72fcca243f269cac5a4cbeee58fac"
MISSING = "optional/quarot.safetensors"
MISSING_BYTES = 150995040
MISSING_SHA = "960e0ca37d0e58327cad7f9cbc9b1cb6822479a07af0c6153ea541c6de0fc7c0"
RESOLVE = f"https://modelscope.cn/models/{REPOSITORY}/resolve/{REVISION}/"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def one_cpu():
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    return cpu


def records(document):
    if document.get("Code") != 200 or document.get("Success") is not True:
        raise ValueError("The ModelScope API response was not successful")
    result = {}
    for value in document["Data"]["Files"]:
        if value["Type"] != "blob":
            continue
        name = value["Path"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or name in result:
            raise ValueError(f"Invalid or repeated manifest path: {name}")
        result[name] = {"bytes": value["Size"], "sha256": value["Sha256"]}
    if len(result) != 109 or sum(v["bytes"] for v in result.values()) != 405008283962:
        raise ValueError("The complete 109-file manifest changed")
    return result


def load_manifest(path):
    if digest(path) != MANIFEST_SHA:
        raise ValueError("Use the original pinned ModelScope manifest")
    result = records(json.loads(path.read_text()))
    if result[MISSING] != {"bytes": MISSING_BYTES, "sha256": MISSING_SHA}:
        raise ValueError("The missing-file contract changed")
    return result


def verify_file(path, expected):
    actual = {"bytes": path.stat().st_size, "sha256": digest(path)}
    if actual != expected:
        raise ValueError(f"File size or SHA256 mismatch: {path}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-files", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--proxy-env", default="https_proxy")
    args = parser.parse_args()
    cpu = one_cpu()
    expected = load_manifest(args.manifest)
    official = json.loads(args.official_files.read_text())
    if records(official) != expected:
        raise ValueError("Fresh fixed-revision metadata differs from the pinned manifest")
    if official["Data"]["LatestCommitter"]["ShortId"] != REVISION[:8]:
        raise ValueError("Fresh metadata reports a different revision")
    args.evidence.mkdir(parents=True, exist_ok=True)
    target = args.output / MISSING
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        actual = verify_file(target, expected[MISSING])
        fetched = False
    else:
        aria2 = shutil.which("aria2c")
        if not aria2:
            raise RuntimeError("aria2c is required; no system installation is attempted")
        proxy = os.environ.get(args.proxy_env, "")
        temporary = target.with_name(target.name + ".partial")
        command = [aria2, "-c", "-x", "16", "-s", "16", "-k", "1M",
                   "--file-allocation=none", "--check-integrity=true",
                   "--auto-file-renaming=false", "--allow-overwrite=false",
                   "--max-tries=5", "--retry-wait=2", "--connect-timeout=15",
                   "--timeout=30", "--summary-interval=10",
                   "--checksum=sha-256=" + MISSING_SHA,
                   "--dir=" + str(target.parent), "--out=" + temporary.name]
        if proxy:
            command.append("--all-proxy=" + proxy)
        command.append(RESOLVE + MISSING)
        safe_command = [value.replace(proxy, "${" + args.proxy_env + "}")
                        if proxy else value for value in command]
        (args.evidence / "download-command.json").write_text(json.dumps({
            "argv_template": safe_command, "proxy_environment": args.proxy_env,
            "proxy_host": urllib.parse.urlsplit(proxy).hostname if proxy else None,
            "cpu_affinity": [cpu], "aria2_sha256": digest(Path(aria2)),
        }, indent=2) + "\n")
        process = subprocess.run(command, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, check=False)
        log = process.stdout
        if proxy:
            secrets = [proxy, urllib.parse.urlsplit(proxy).username,
                       urllib.parse.urlsplit(proxy).password]
            for secret in secrets:
                if secret:
                    log = log.replace(secret, "[proxy credential]")
        (args.evidence / "download.log").write_text(log)
        if process.returncode:
            raise RuntimeError(f"aria2 failed with exit code {process.returncode}")
        actual = verify_file(temporary, expected[MISSING])
        temporary.rename(target)
        fetched = True
    result = {
        "repository": REPOSITORY, "revision": REVISION,
        "manifest_sha256": MANIFEST_SHA,
        "official_files_sha256": digest(args.official_files),
        "official_file_records_equal": True, "official_blob_count": 109,
        "target": str(target), "file": MISSING, **actual,
        "downloaded": fetched, "download_url": RESOLVE + MISSING,
        "cpu_affinity": [cpu], "device_execution": False,
        "nas_files_modified": 0, "weight_shards_read": 0, "pass": True,
    }
    (args.evidence / "recovery.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
