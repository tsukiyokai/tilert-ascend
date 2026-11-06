import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / ".artifacts"
GENERATED = {".artifacts", "reports", "evidence", "__pycache__",
             ".pytest_cache", ".ruff_cache", "build", "_build"}
SUFFIXES = {".o", ".a", ".elf", ".pt", ".pyc", ".log", ".npy", ".npz",
            ".safetensors", ".tar", ".gz", ".zip", ".whl"}


def git(*args, env=None, data=None):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], input=data, env=env,
        check=True, stdout=subprocess.PIPE).stdout


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def generated(name):
    path = Path(name)
    return (any(part in GENERATED or "_build" in part
                or part.startswith("build-") for part in path.parts[:-1])
            or ".so" in path.suffixes or path.suffix in SUFFIXES
            or path.name in {"provenance.json", "fusion_result.json"})


def paths(*args):
    return set(os.fsdecode(name) for name in git("ls-files", "-z", *args)
               .split(b"\0") if name)


def inventory():
    tracked = paths("--cached")
    visible = paths("--cached", "--others", "--exclude-standard")
    ignored = paths("--others", "--ignored", "--exclude-standard")
    files = []
    for name in sorted(visible | ignored):
        if name.startswith(".artifacts/"):
            continue
        path = ROOT / name
        if path.exists() or path.is_symlink():
            files.append({"path": name, "bytes": path.lstat().st_size,
                          "tracked": name in tracked,
                          "kind": "artifact" if generated(name) else "source"})
    report = {"head": git("rev-parse", "HEAD").decode().strip(),
              "branch": git("branch", "--show-current").decode().strip(),
              "files": files,
              "tracked_artifacts": [row["path"] for row in files
                                    if row["tracked"]
                                    and row["kind"] == "artifact"]}
    write(STORE / "inventory.json", report)
    return {"files": len(files), "artifact_bytes": sum(
        row["bytes"] for row in files if row["kind"] == "artifact"),
        "tracked_artifacts": len(report["tracked_artifacts"]),
        "manifest": str(STORE / "inventory.json")}


def snapshot(name):
    ref = f"refs/snapshots/{name}"
    git("check-ref-format", ref)
    source = sorted(path for path in paths(
        "--cached", "--others", "--exclude-standard")
        if not generated(path)
        and ((ROOT / path).is_file() or (ROOT / path).is_symlink()))
    store = STORE / "snapshots" / name
    store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        git("read-tree", "--empty", env=env)
        git("--literal-pathspecs", "add", "--pathspec-from-file=-",
            "--pathspec-file-nul", env=env,
            data=b"\0".join(os.fsencode(path) for path in source) + b"\0")
        tree = git("write-tree", env=env).decode().strip()
    if (store / "manifest.json").exists():
        report = json.loads((store / "manifest.json").read_text())
        assert report["tree"] == tree, f"Snapshot name already used: {name}"
        return report
    head = git("rev-parse", "HEAD").decode().strip()
    commit = git("commit-tree", tree, "-p", head,
                 "-m", f"wip: snapshot {name}").decode().strip()
    git("update-ref", ref, commit, "0" * 40)
    (store / "worktree.patch").write_bytes(git("diff", "--binary", "HEAD"))
    archive = store / "source.tar.gz"
    git("archive", "--format=tar.gz", f"--output={archive}", commit)
    report = {"name": name, "ref": ref, "commit": commit, "tree": tree,
              "parent": head, "status": "wip", "source_files": len(source),
              "created_at": datetime.now(timezone.utc).isoformat(),
              "files": [{"path": path.name, "sha256": digest(path)}
                        for path in (archive, store / "worktree.patch")]}
    write(store / "manifest.json", report)
    return report


def register(args):
    store = STORE / "runs" / args.name
    store.parent.mkdir(parents=True, exist_ok=True)
    evidence = args.evidence.resolve()
    summary_path = evidence / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    audit = json.loads(args.audit.read_text()) if args.audit else None
    assert args.source_ref or args.legacy, "New runs require --source-ref"
    commit = (git("rev-parse", f"{args.source_ref}^{{commit}}")
              .decode().strip() if args.source_ref else None)
    with tempfile.TemporaryDirectory(dir=STORE) as temporary:
        staged = Path(temporary)
        shutil.copytree(evidence, staged / "evidence")
        if args.audit:
            shutil.copy2(args.audit, staged / "audit.json")
        for binary in args.binary:
            target = staged / "binaries" / binary.parent.name / binary.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(binary, target)
        report = {
            "name": args.name, "source_commit": commit,
            "source_identity": "snapshot" if commit else "legacy_partial",
            "verdict": ("failed" if not summary else
                        "trajectory_verified" if audit and audit["pass"] else
                        "rejected" if audit else
                        "recorded_baseline" if args.legacy else "unverified"),
            "scope": audit["scope"] if audit else "historical receipt",
            "summary": summary, "origin": str(evidence.relative_to(ROOT)),
            "files": [{"path": str(path.relative_to(staged)),
                       "bytes": path.stat().st_size, "sha256": digest(path)}
                      for path in sorted(staged.rglob("*")) if path.is_file()],
        }
        if store.exists():
            assert json.loads((store / "manifest.json").read_text()) == report, (
                f"Run name already used: {args.name}")
            verify(store / "manifest.json")
        else:
            write(staged / "manifest.json", report)
            staged.rename(store)
    return {"name": args.name, "verdict": report["verdict"],
            "files": len(report["files"]), "manifest": str(store / "manifest.json")}


def verify(path):
    report = json.loads(path.read_text())
    changed = [row["path"] for row in report["files"]
               if digest(path.parent / row["path"]) != row["sha256"]]
    assert not changed, f"Artifact hash mismatch: {changed}"
    if "ref" in report:
        assert git("rev-parse", report["ref"]).decode().strip() == report["commit"]
    return {"pass": True, "files": len(report["files"])}


def check():
    names = git("diff", "--cached", "--name-only", "--diff-filter=ACMRT",
                "-z").split(b"\0")
    bad = [os.fsdecode(name) for name in names
           if name and generated(os.fsdecode(name))]
    assert not bad, ("Generated files belong in .artifacts; unstage: "
                     + ", ".join(bad))
    return {"pass": True, "staged_files": sum(bool(name) for name in names)}


def initialize():
    git("config", "--local", "core.hooksPath", ".githooks")
    return {"hooks": git("config", "--get", "core.hooksPath").decode().strip()}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory")
    commands.add_parser("check")
    commands.add_parser("init")
    item = commands.add_parser("snapshot")
    item.add_argument("name")
    item = commands.add_parser("register")
    item.add_argument("name")
    item.add_argument("--evidence", type=Path, required=True)
    item.add_argument("--audit", type=Path)
    item.add_argument("--binary", "--artifact", type=Path, action="append", default=[])
    item.add_argument("--source-ref")
    item.add_argument("--legacy", action="store_true")
    item = commands.add_parser("verify")
    item.add_argument("manifest", type=Path)
    args = parser.parse_args()
    match args.command:
        case "inventory":
            result = inventory()
        case "snapshot":
            result = snapshot(args.name)
        case "register":
            result = register(args)
        case "verify":
            result = verify(args.manifest)
        case "init":
            result = initialize()
        case "check":
            result = check()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
