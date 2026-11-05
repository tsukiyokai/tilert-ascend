"""Build bound host lookup extensions from unchanged Python sources."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import traceback


MODULE = "tilert.models.glm_5_2_ascend.lookup_draft"
DIRECTORY = Path("tilert/models/glm_5_2_ascend")
DIRECTIVES = {
    "language_level": 3, "infer_types": False, "binding": True,
    "boundscheck": True, "wraparound": True,
    "nonecheck": True, "overflowcheck": True,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_guard(identity):
    return f'''

from hashlib import sha256 as _native_digest
from pathlib import Path as _NativePath

_NATIVE_BUILD = {identity!r}
for _native_name, _native_expected in _NATIVE_BUILD["source_sha256"].items():
    _native_path = _NativePath(__file__).with_name(_native_name)
    _native_actual = _native_digest(_native_path.read_bytes()).hexdigest()
    if _native_actual != _native_expected:
        raise ImportError("Native lookup sources changed; rebuild extension")
_NATIVE_BUILD["loaded_binary_sha256"] = _native_digest(
    _NativePath(__file__).read_bytes()
).hexdigest()
del _native_digest, _NativePath, _native_name, _native_expected, _native_path
del _native_actual
'''


def prepare(root, output):
    import Cython
    import setuptools

    paths = [root / DIRECTORY / ("lookup_draft" + suffix)
             for suffix in (".py", ".pxd")]
    identity = {
        "kind": "cython", "module": MODULE,
        "source_sha256": {path.name: digest(path) for path in paths},
        "cython": Cython.__version__, "setuptools": setuptools.__version__,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "python": platform.python_version(), "machine": platform.machine(),
        "directives": DIRECTIVES, "acyclic_nodes": True,
    }
    source = output / "source"
    package = source / DIRECTORY
    package.mkdir(parents=True)
    original = paths[0].read_text()
    if original.count("class _Node:") != 1:
        raise ValueError("Expected one lookup node class")
    # Node links own descendants only; the drafter retains cyclic GC support.
    lowered = original.replace(
        "class _Node:",
        "import cython\n\n\n@cython.no_gc\n@cython.cclass\nclass _Node:", 1,
    )
    (package / "lookup_draft.py").write_text(lowered + native_guard(identity))
    shutil.copyfile(paths[1], package / "lookup_draft.pxd")
    setup = f'''from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    name="tilert-lookup-native",
    ext_modules=cythonize(
        [Extension({MODULE!r}, [{str(DIRECTORY / 'lookup_draft.py')!r}])],
        compiler_directives={DIRECTIVES!r},
        include_path=["."],
        annotate=True,
    ),
)
'''
    (source / "setup_native.py").write_text(setup)
    return identity, source, paths


def validate(binary, identity):
    script = '''import gc, importlib.util, json, pathlib, sys
name, filename = sys.argv[1:]
spec = importlib.util.spec_from_file_location(name, filename)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
loaded = pathlib.Path(module.__file__).resolve()
assert loaded == pathlib.Path(filename).resolve()
assert "__pyx_vtable__" in vars(module.LookupDrafter)
assert "context" in vars(module.LookupDrafter)
assert gc.is_tracked(module._Node(0, 0, -1)) is (
    not module._NATIVE_BUILD.get("acyclic_nodes", False))
assert gc.is_tracked(module.LookupDrafter([1,2,1,2]))
assert module.LookupDrafter([1,2,7,9,2,8,2], 3, 1).propose(1) == [8]
assert "torch" not in sys.modules and "Cython" not in sys.modules
print(json.dumps(module._NATIVE_BUILD, sort_keys=True))
'''
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, MODULE, str(binary)],
        capture_output=True, text=True, check=True,
    )
    loaded = json.loads(process.stdout)
    assert loaded == {**identity, "loaded_binary_sha256": digest(binary)}
    return loaded


def build(args, report):
    identity, source, originals = prepare(args.root, args.output)
    report["native_identity"] = identity
    compiler = os.environ.get("CC") or sysconfig.get_config_var("CC")
    report["compiler_version"] = subprocess.check_output(
        [*shlex.split(compiler), "--version"], text=True,
    )
    report["compiler_config"] = {
        name: sysconfig.get_config_var(name)
        for name in ("CC", "CFLAGS", "CCSHARED", "LDSHARED")
    }
    report["compiler_environment"] = {
        name: os.environ.get(name)
        for name in ("CC", "CFLAGS", "CPPFLAGS", "LDFLAGS")
    }
    command = [sys.executable, "setup_native.py", "build_ext", "--build-lib",
               str(args.output / "lib"), "--build-temp",
               str(args.output / "objects")]
    report["build_command"] = command
    process = subprocess.run(command, cwd=source, capture_output=True,
                             text=True)
    (args.output / "build.stdout").write_text(process.stdout)
    (args.output / "build.stderr").write_text(process.stderr)
    if process.returncode:
        raise RuntimeError("Native compilation failed; inspect build.stderr")
    binary = args.output / "lib" / DIRECTORY / (
        "lookup_draft" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    for original in originals:
        assert digest(original) == identity["source_sha256"][original.name]
        shutil.copyfile(original, binary.with_name(original.name))
    report["binary"] = str(binary)
    report["binary_sha256"] = digest(binary)
    generated = source / DIRECTORY
    report["generated_c"] = str(generated / "lookup_draft.c")
    report["generated_c_sha256"] = digest(generated / "lookup_draft.c")
    report["generated_input_sha256"] = digest(generated / "lookup_draft.py")
    report["isolated_load"] = validate(binary, identity)
    report["status"] = "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "hostname": platform.node(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.root, text=True,
        ).strip(), "builder_sha256": digest(Path(__file__)),
        "scope": "Host lookup extension; no Torch or NPU runtime required",
    }
    try:
        build(args, report)
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    (args.output / "build.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({"status": report["status"],
                      "report": str(args.output / "build.json")}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
