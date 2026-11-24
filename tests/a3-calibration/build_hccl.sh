#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=${CANN_ROOT:-/opt/ascend/cann-9.2.0}
build_dir=${1:-"$script_dir/build"}
compiler=${CXX:-g++}
python=${PYTHON:-python3}

if [[ ! -f "$cann_root/include/hccl/hccl.h" ]]; then
    printf 'CANN headers not found under %s\n' "$cann_root" >&2
    exit 1
fi
mkdir -p -- "$build_dir"
build_dir=$(cd -- "$build_dir" && pwd)
binary="$build_dir/hccl_latency"
source_file="$script_dir/hccl_latency.cpp"
compile=("$compiler" -std=c++17 -O2 -g -Wall -Wextra -Werror
    -fstack-protector-strong -fPIE -pie
    -I"$cann_root/include" "$source_file" -o "$binary"
    -L"$cann_root/lib64" -Wl,-rpath,"$cann_root/lib64"
    -Wl,-z,relro,-z,now,-z,noexecstack
    -Wl,--no-as-needed -lhccl -lhcomm -lacl_rt -Wl,--as-needed -ldl -pthread)
"${compile[@]}"

"$python" - "$binary" "$source_file" "$cann_root" "${compile[@]}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

binary, source, cann = map(pathlib.Path, sys.argv[1:4])
command = sys.argv[4:]


def digest(path):
    resolved = path.resolve(strict=True)
    hasher = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"path": str(path), "resolved": str(resolved),
            "sha256": hasher.hexdigest(), "bytes": resolved.stat().st_size}


artifacts = [binary, source, source.parent / "build_hccl.sh"]
artifacts += [cann / "include" / name for name in (
    "acl/acl_rt.h", "hccl/hccl.h", "hccl/hccl_comm.h", "hccl/hccl_types.h")]
artifacts += [cann / "lib64" / name for name in (
    "libhccl.so", "libhcomm.so", "libacl_rt.so")]
artifacts += [cann / "toolkit/tools/hccl_test" / name for name in (
    "common/src/hccl_test_common.cc", "opbase_test/hccl_allreduce_rootinfo_test.cc")]
install_info = cann / "aarch64-linux/ascend_toolkit_install.info"
manifest = {
    "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "machine": platform.machine(),
    "host": platform.node(),
    "compile_command": command,
    "compiler_version": subprocess.check_output([command[0], "--version"], text=True),
    "cann_install_info": install_info.read_text() if install_info.exists() else None,
    "artifacts": [digest(path) for path in artifacts],
    "verification_scope": "CPU compilation only; executable and NPU workload not run",
}
binary.with_name(binary.name + ".build.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

printf 'Built %s\nBuild manifest: %s.build.json\n' "$binary" "$binary"
printf 'Run only after reserving the 16 devices:\n  %q --output NEW_DIRECTORY --expansion aiv\n' "$binary"
