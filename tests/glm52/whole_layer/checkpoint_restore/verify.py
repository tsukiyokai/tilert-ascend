"""Check manifest sizes and explicitly requested hashes in the consumer namespace."""

import argparse
import json
import os
from pathlib import Path

from restore import MANIFEST_SHA, REPOSITORY, REVISION, digest, load_manifest, one_cpu


def mount_for(path):
    path = path.resolve()
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        target = Path(fields[4].replace("\\040", " "))
        if path == target or target in path.parents:
            separator = fields.index("-")
            matches.append((len(target.parts), {
                "target": str(target), "options": fields[5].split(","),
                "filesystem": fields[separator + 1],
                "source": fields[separator + 2],
                "super_options": fields[separator + 3].split(","),
            }))
    return max(matches, key=lambda value: value[0])[1]


def inspect_file(path, expected, check_hash):
    before = path.stat()
    result = {"path": str(path), "bytes": before.st_size,
              "expected_bytes": expected["bytes"],
              "size_matches": before.st_size == expected["bytes"]}
    if check_hash:
        result.update({"sha256": digest(path),
                       "expected_sha256": expected["sha256"]})
        result["hash_matches"] = result["sha256"] == expected["sha256"]
    after = path.stat()
    result["stable_stat"] = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    result["pass"] = (result["size_matches"] and result["stable_stat"]
                      and result.get("hash_matches", True))
    return result


def create_view(output, sources):
    output.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
        if not os.path.samefile(target, source):
            raise RuntimeError(f"The model view resolved to the wrong owner: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nas-model", type=Path, required=True)
    parser.add_argument("--owned-model", type=Path, required=True)
    parser.add_argument("--hash-file", action="append", default=[])
    parser.add_argument("--require-readonly-nas", action="store_true")
    parser.add_argument("--require-no-devices", action="store_true")
    parser.add_argument("--view", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cpu = one_cpu()
    expected = load_manifest(args.manifest)
    requested = set(args.hash_file)
    if requested - expected.keys():
        raise ValueError("Every requested hash must name an exact manifest file")
    devices = sorted(str(path) for path in Path("/dev").glob("davinci*"))
    if args.require_no_devices and devices:
        raise RuntimeError("The CPU verification container must not map davinci devices")
    nas_mount = mount_for(args.nas_model)
    if args.require_readonly_nas and "ro" not in nas_mount["options"]:
        raise RuntimeError("The NAS must be mounted read-only in this namespace")
    values, sources = [], {}
    for name, entry in sorted(expected.items()):
        owned = args.owned_model / name
        source = owned if owned.exists() else args.nas_model / name
        row = {"file": name, "owner": "owned" if source == owned else "nas"}
        try:
            row.update(inspect_file(source, entry, name in requested))
        except OSError as error:
            row.update({"path": str(source), "pass": False, "error": str(error)})
        values.append(row)
        sources[name] = source
    passed = all(row["pass"] for row in values)
    if args.view and passed:
        if args.nas_model.resolve() in args.view.resolve().parents:
            raise ValueError("The model view must not be created on the NAS")
        create_view(args.view, sources)
    report = {
        "repository": REPOSITORY, "revision": REVISION,
        "manifest_sha256": MANIFEST_SHA,
        "mount_namespace": os.readlink("/proc/self/ns/mnt"),
        "nas_mount": nas_mount, "nas_model": str(args.nas_model),
        "owned_model": str(args.owned_model), "files": values,
        "checked_files": len(values), "matched_sizes": sum(
            row.get("size_matches", False) for row in values),
        "hashed_files": sorted(requested),
        "hashed_bytes": sum(expected[name]["bytes"] for name in requested),
        "all_manifest_hashes_checked": len(requested) == len(expected),
        "view": str(args.view) if args.view and passed else None,
        "view_links_verified": len(sources) if args.view and passed else 0,
        "cpu_affinity": [cpu], "mapped_davinci_devices": devices,
        "device_execution": False, "nas_files_modified": 0,
        "pass": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "checked_files", "matched_sizes", "hashed_files", "hashed_bytes",
        "all_manifest_hashes_checked", "view_links_verified", "pass")}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
