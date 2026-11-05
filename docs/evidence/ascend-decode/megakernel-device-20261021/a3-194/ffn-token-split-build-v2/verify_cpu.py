"""Verify frozen build hashes and stage its exact portable CPU tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("parent", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
build = Path(__file__).resolve().parent
manifest = json.loads((build / "manifest.json").read_text())
for collection in ("sources", "artifacts"):
    for name, identity in manifest[collection].items():
        actual = hashlib.sha256((build / name).read_bytes()).hexdigest()
        if actual != identity:
            raise RuntimeError(f"Frozen {collection} changed: {name}")
with tempfile.TemporaryDirectory() as temporary:
    staged = Path(temporary) / "tests/glm52/whole_layer/ffn_token_split"
    staged.mkdir(parents=True)
    for name in ("prepare.py", "build.py", "test_contract.py", "ownership_probe.cpp", "README.md"):
        shutil.copy2(build / "candidate_tools" / name, staged / name)
    for name in ("split_contract.h", "split_native.h"):
        shutil.copy2(build / "source" / name, staged / name)
    environment = dict(os.environ, FFN_TOKEN_SPLIT_PARENT=str(args.parent.resolve()),
                       PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "unittest", "test_contract", "-v"],
                            cwd=staged, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(result.stdout)
print(json.dumps({"source_count": len(manifest["sources"]),
                  "artifact_count": len(manifest["artifacts"]),
                  "frozen_hashes_verified": True, "returncode": result.returncode,
                  "log": str(args.output)}))
if result.returncode:
    raise SystemExit(result.returncode)
