"""Install the exact audited lookup artifact without replacing live files."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

from build import DIRECTORY, validate
from check import digest, load_build


def install(args, report):
    checks = json.loads(args.checks.read_text())
    if (checks["status"] != "pass"
            or checks["build_receipt_sha256"] != digest(args.build)):
        raise ValueError("Passing checks must bind this exact build receipt")
    sys.path.insert(0, str(args.root))
    _, build, identity = load_build(args.build, args.root)
    binary = Path(build["binary"])
    destination = args.root / DIRECTORY / binary.name
    report.update(build_receipt_sha256=digest(args.build),
                  checks_sha256=digest(args.checks), native=identity,
                  destination=str(destination))
    if destination.exists():
        if digest(destination) != build["binary_sha256"]:
            raise FileExistsError("Existing native extension is immutable")
    else:
        descriptor, name = tempfile.mkstemp(prefix=".lookup-", suffix=".so",
                                            dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                with binary.open("rb") as source:
                    shutil.copyfileobj(source, stream)
            shutil.copymode(binary, temporary)
            os.link(temporary, destination)
        finally:
            temporary.unlink()
    report["installed_load"] = validate(destination, build["native_identity"])
    report["status"] = "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.output.exists():
        raise FileExistsError("Choose a new immutable install receipt")
    report = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "installer_sha256": digest(Path(__file__)),
    }
    try:
        install(args, report)
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "output": str(args.output)}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
