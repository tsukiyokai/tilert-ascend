import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.glm52.e2e_tp8_ep8.contracts import (
    ALLOWED_A3_CHIPS,
    ALLOWED_A3_GENERATION,
    ALLOWED_KBENV_STATES,
    DEFAULT_KBENV_BIN,
    IsolationError,
    MIN_KBENV_RAM_PCT,
    WORLD_SIZE,
)
from tests.glm52.e2e_tp8_ep8.hygiene_gate import (
    allocate_free_port_base,
    allocate_isolation_env,
    assert_host_allowed,
    parse_kbenv_output,
    parse_strict_ipv4,
    select_a3_host,
    select_free_die_slice,
    validate_run_id,
    verify_isolation_env,
    verify_shm_isolation,
)

DEFAULT_IMAGE = "0d182d9cae98"
DEFAULT_PRESTAGE_ROOT = "/dev/shm/tilert_prestage"
SHARED_RO_PRESTAGE_ROOT = "/dev/shm/glm52_shared_ro"
DEFAULT_IFNAME = "enp23s0f3"
SFS_MODEL_DIR = "/mnt/sfs_turbo_new/shanshan/GLM-5.2-w4a8c8"
SFS_CKPT_PATH = (
    "/mnt/sfs_turbo_new/shanshan/dspine-glm52/checkpoints/cal_l78_v3_hybrid_best.pt"
)
SFS_AGY_TP8_DIR = "/mnt/sfs_turbo_new/shanshan/dspine-glm52/runtime_bundle_v5/agy_tp8"
SFS_PROBE_TARGETS: tuple[str, ...] = (
    SFS_MODEL_DIR,
    SFS_CKPT_PATH,
    "/mnt/sfs_turbo_new/trt-sync-tmp",
)
CONTAINER_SFS_CKPT_PROBE = "/sfs/shanshan/dspine-glm52/checkpoints/cal_l78_v3_hybrid_best.pt"
PRESTAGE_CKPT_PATH = "/prestage/cal_l78_v3_hybrid_best.pt"
CONTAINER_SFS_PROBES: tuple[str, ...] = (
    "/work/model-raw",
    CONTAINER_SFS_CKPT_PROBE,
)
DEFAULT_DSPINE_STAMP = "tests/.artifacts/checks/dspine_unit_stamp.json"
PRESTAGE_SPECS: dict[str, tuple[str, str]] = {
    "ckpt": (SFS_CKPT_PATH, "cal_l78_v3_hybrid_best.pt"),
    "case01": (f"{SFS_AGY_TP8_DIR}/cases/case-01-coding", "case-01-coding"),
    "mtp-prefix": (f"{SFS_AGY_TP8_DIR}/mtp-prefix-rank-0.pt", "mtp-prefix-rank-0.pt"),
    "model-raw": (SFS_MODEL_DIR, "GLM-5.2-w4a8c8"),
}
DEFAULT_STATIC_RO_MOUNTS: tuple[tuple[str, str], ...] = (
    ("/mnt/sfs_turbo_new/shanshan", "/sfs/shanshan"),
    (
        "/mnt/sfs_turbo_new/trt-sync-tmp/vendor-hook-runtime212-int8",
        "/assets/vendor-hook-runtime212-int8",
    ),
    ("/mnt/sfs_turbo_new/trt-sync-tmp/kernel-overlay", "/opt/kernel-overlay"),
    ("/mnt/sfs_turbo_new/trt-sync-tmp/vllm-unified", "/assets/vllm-unified"),
    ("/mnt/sfs_turbo_new/trt-sync-tmp/vllm-ascend-unified", "/assets/vllm-ascend-unified"),
    ("/mnt/sfs_turbo_new/trt-sync-tmp/shmem-lib", "/assets/shmem-lib"),
    (
        "/mnt/sfs_turbo_new/trt-sync-tmp/runtime-sfa-token-mlatail",
        "/work/runtime-sfa-token-mlatail",
    ),
    ("/mnt/sfs_turbo_new/trt-sync-tmp/runs/v3x58", "/assets/runs/v3x58"),
    (
        "/mnt/sfs_turbo_new/shanshan/dspine-glm52/runtime_bundle_v5/cann-9.2.0",
        "/opt/ascend/cann-9.2.0",
    ),
    (
        "/mnt/sfs_turbo_new/shanshan/dspine-glm52/runtime_bundle_v5/nnal",
        "/usr/local/Ascend/nnal",
    ),
    (
        "/mnt/sfs_turbo_new/shanshan/dspine-glm52/runtime_bundle_v5/shmem-capacity",
        "/assets/shmem-capacity",
    ),
    ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver"),
    ("/usr/local/sbin/npu-smi", "/usr/local/sbin/npu-smi"),
)
DEFAULT_WHOLE_LAYER_BASE: dict[str, str] = {
    "indexed": "/assets/runs/v3x58/build/indexed/whole_layer_indexed_attention_tp8.so",
    "shared": "/assets/runs/v3x58/build/full/whole_layer_full_attention_tp8.so",
    "ffn": "/sfs/shanshan/dspine-glm52/runtime_bundle_v5/agy_tp8/build/complete_ffn_tp8_ep8.so",
    "metadata": (
        "/sfs/shanshan/dspine-glm52/runtime_bundle_v5/agy_tp8/build/layer_local_chain_tp8_ep8.so"
    ),
}
RUN_SUBDIRS: tuple[str, ...] = (
    "source",
    "builds",
    "control",
    "bench",
    "report",
    "vllm_cache",
    "inductor_cache",
    "triton_cache",
    "xdg_cache",
    "ascend_log",
    "tmp",
)
SHARED_DEVICES: tuple[str, ...] = (
    "/dev/davinci_manager",
    "/dev/devmm_svm",
    "/dev/hisi_hdc",
)
MEMINFO_KB_RE = re.compile(r"^(MemTotal|MemAvailable):\s+(\d+)\s+kB$", re.MULTILINE)
SUPPORTED_WORLD_SIZES: tuple[int, ...] = (8, 16)
ALARM_DIE_MAX_HBM_PCT: float = 8.0
ALARM_DIE_TOKENS: frozenset[str] = frozenset({"·", "×"})


def _eligible_die_indices(host: Any, allow_alarm_dies: bool) -> tuple[int, ...]:
    if not allow_alarm_dies:
        return tuple(host.free_die_indices)
    die_map = getattr(host, "die_map", "")
    if die_map:
        return tuple(idx for idx, tok in enumerate(die_map) if tok in ALARM_DIE_TOKENS)
    return tuple(host.free_die_indices)


def _validate_world_size(world_size: int) -> int:
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size not in SUPPORTED_WORLD_SIZES
    ):
        raise IsolationError(
            f"Unsupported world_size {world_size!r}; expected one of {SUPPORTED_WORLD_SIZES}"
        )
    return world_size


def _infer_env_world_size(env: Mapping[str, str], world_size: int | None) -> int:
    if world_size is not None:
        return _validate_world_size(world_size)
    vis = env.get("ASCEND_RT_VISIBLE_DEVICES", "")
    parts = [p.strip() for p in vis.split(",") if p.strip()]
    if len(parts) in SUPPORTED_WORLD_SIZES:
        return len(parts)
    return WORLD_SIZE


def generate_run_id(prefix: str = "m21") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    suffix = secrets.token_hex(3)
    return validate_run_id(f"{prefix}_{stamp}_{suffix}")


def parse_die_slice(spec: str, world_size: int = WORLD_SIZE) -> tuple[int, ...] | None:
    ws = _validate_world_size(world_size)
    clean = spec.strip()
    if not clean or clean == "auto":
        return None
    if clean in ("0..15", "0-15", "all", "full"):
        if ws != 16:
            raise IsolationError(f"die_slice {spec!r} requires world_size=16, got {ws}")
        return tuple(range(16))
    if clean in ("0..7", "0-7", "low"):
        if ws != 8:
            raise IsolationError(f"die_slice {spec!r} requires world_size=8, got {ws}")
        return tuple(range(8))
    if clean in ("8..15", "8-15", "high"):
        if ws != 8:
            raise IsolationError(f"die_slice {spec!r} requires world_size=8, got {ws}")
        return tuple(range(8, 16))
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    if len(parts) != ws:
        raise IsolationError(f"die_slice must specify {ws} dies, got {spec!r}")
    dies: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise IsolationError(f"Invalid die index {part!r} in {spec!r}")
        idx = int(part)
        if not (0 <= idx < 16) or idx in dies:
            raise IsolationError(f"Invalid or duplicate die index {idx} in {spec!r}")
        dies.append(idx)
    return tuple(dies)


def select_host_and_dies(
    hosts: Sequence[Any],
    target_host: str | None = None,
    die_slice: str = "auto",
    min_free_dies: int | None = None,
    allow_idle_dies: bool = False,
    world_size: int = WORLD_SIZE,
    allow_alarm_dies: bool = False,
) -> tuple[Any, tuple[int, ...]]:
    ws = _validate_world_size(world_size)
    if allow_alarm_dies and not target_host:
        raise IsolationError("--allow-alarm-dies requires explicit target_host")
    req_dies = ws if min_free_dies is None else min_free_dies
    if (
        isinstance(req_dies, bool)
        or not isinstance(req_dies, int)
        or req_dies < ws
        or req_dies > 16
    ):
        raise IsolationError(f"min_free_dies must be in {ws}..16, got {min_free_dies!r}")
    explicit_dies = parse_die_slice(die_slice, world_size=ws)
    if target_host is None:
        selected = select_a3_host(hosts, min_free_dies=req_dies)
    else:
        canonical = assert_host_allowed(target_host)
        matches = [h for h in hosts if h.ip == canonical]
        if not matches:
            raise IsolationError(f"Target host {canonical} not found in kbenv output")
        selected = matches[0]
        if selected.generation != ALLOWED_A3_GENERATION:
            raise IsolationError(f"Host {canonical} generation {selected.generation!r} != A3")
        if selected.chip not in ALLOWED_A3_CHIPS:
            raise IsolationError(f"Host {canonical} chip {selected.chip!r} not eligible")
        if not (allow_idle_dies or allow_alarm_dies) and selected.state not in ALLOWED_KBENV_STATES:
            raise IsolationError(f"Host {canonical} state {selected.state!r} not idle/partial")
        if selected.ram_pct is None or selected.ram_pct < MIN_KBENV_RAM_PCT:
            raise IsolationError(
                f"Host {canonical} RAM {selected.ram_pct}% < {MIN_KBENV_RAM_PCT}%"
            )
    eligible = _eligible_die_indices(selected, allow_alarm_dies=allow_alarm_dies)
    if explicit_dies is not None:
        elig_set = set(eligible)
        missing = [d for d in explicit_dies if d not in elig_set]
        if missing:
            raise IsolationError(
                f"Requested dies {missing} busy on {selected.ip} "
                f"(free={list(selected.free_die_indices)})"
            )
        return selected, explicit_dies
    return selected, select_free_die_slice(eligible, count=ws)


def select_fleet_slots(
    hosts: Sequence[Any],
    max_slots: int | None = None,
    dual_slot: bool = True,
    exclude_hosts: Sequence[str] = (),
    world_size: int = WORLD_SIZE,
) -> list[tuple[str, tuple[int, ...]]]:
    ws = _validate_world_size(world_size)
    excluded = {
        parse_strict_ipv4(ip)
        for ip in exclude_hosts
        if ip is not None and str(ip).strip()
    }
    eligible: list[Any] = []
    for host in hosts:
        try:
            assert_host_allowed(host.ip)
        except IsolationError:
            continue
        if host.ip in excluded:
            continue
        if host.generation != ALLOWED_A3_GENERATION or host.chip not in ALLOWED_A3_CHIPS:
            continue
        if host.state not in ALLOWED_KBENV_STATES:
            continue
        if host.ram_pct is None or host.ram_pct < MIN_KBENV_RAM_PCT:
            continue
        if host.total_dies != 16 or host.free_dies < ws:
            continue
        eligible.append(host)
    eligible.sort(
        key=lambda h: (
            0 if h.state == "空闲" else 1,
            -h.free_dies,
            -(h.ram_pct or 0),
            h.ip,
        )
    )
    slots: list[tuple[str, tuple[int, ...]]] = []
    for host in eligible:
        free_set = set(host.free_die_indices)
        if ws == 16:
            full = tuple(range(16))
            if all(d in free_set for d in full):
                slots.append((host.ip, full))
            continue
        low = tuple(range(8))
        high = tuple(range(8, 16))
        added_half = False
        if all(d in free_set for d in low):
            slots.append((host.ip, low))
            added_half = True
        if (dual_slot or not added_half) and all(d in free_set for d in high):
            slots.append((host.ip, high))
            added_half = True
        if not added_half and len(host.free_die_indices) >= ws:
            slots.append((host.ip, select_free_die_slice(host.free_die_indices, ws)))
    if max_slots is not None:
        return slots[: max(0, int(max_slots))]
    return slots


def parse_meminfo_bytes(meminfo_text: str) -> tuple[int, int]:
    found = {k: int(v) * 1024 for k, v in MEMINFO_KB_RE.findall(meminfo_text)}
    if "MemTotal" not in found or "MemAvailable" not in found:
        raise IsolationError("Failed to parse MemTotal/MemAvailable from /proc/meminfo")
    return found["MemTotal"], found["MemAvailable"]


def parse_df_avail_bytes(df_text: str) -> int:
    lines = [ln.strip() for ln in df_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise IsolationError(f"Unexpected df output: {df_text!r}")
    parts = lines[-1].split()
    if len(parts) < 4 or not parts[3].isdigit():
        raise IsolationError(f"Cannot parse df available bytes from {lines[-1]!r}")
    return int(parts[3])


def check_prestage_capacity(
    shm_free_bytes: int,
    mem_total_bytes: int,
    mem_avail_bytes: int,
    payload_bytes: int,
    min_ram_pct: int = MIN_KBENV_RAM_PCT,
    shm_headroom_bytes: int = 32 * 1024**3,
) -> tuple[bool, str]:
    if payload_bytes < 0 or mem_total_bytes <= 0:
        return False, "Invalid capacity inputs"
    rem_shm = shm_free_bytes - payload_bytes
    if rem_shm < shm_headroom_bytes:
        return (
            False,
            f"Insufficient /dev/shm headroom after staging: "
            f"{rem_shm // (1024**3)} GiB < {shm_headroom_bytes // (1024**3)} GiB",
        )
    rem_ram_pct = ((mem_avail_bytes - payload_bytes) * 100.0) / float(mem_total_bytes)
    if rem_ram_pct < float(min_ram_pct):
        return (
            False,
            f"Insufficient host RAM headroom after staging: "
            f"{rem_ram_pct:.1f}% < {min_ram_pct}%",
        )
    return True, f"ok (rem_shm={rem_shm // (1024**3)}GiB, rem_ram={rem_ram_pct:.1f}%)"


def build_sfs_probe_script(
    paths: Sequence[str] = SFS_PROBE_TARGETS,
    timeout_s: int = 5,
) -> str:
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
    checks = [f"timeout {int(timeout_s)} test -e {shlex.quote(p)}" for p in paths]
    return " && ".join(checks)


def build_container_sfs_probe_script(
    probes: Sequence[str] = CONTAINER_SFS_PROBES,
    timeout_s: int = 5,
    prestage_ckpt: bool | None = None,
) -> str:
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
    checks: list[str] = []
    for path in probes:
        if path == CONTAINER_SFS_CKPT_PROBE:
            if prestage_ckpt is True:
                checks.append(
                    f"timeout {int(timeout_s)} test -e {shlex.quote(PRESTAGE_CKPT_PATH)}"
                )
            elif prestage_ckpt is False:
                checks.append(f"timeout {int(timeout_s)} test -e {shlex.quote(path)}")
            else:
                checks.append(
                    f"(test -e {shlex.quote(PRESTAGE_CKPT_PATH)} || "
                    f"timeout {int(timeout_s)} test -e {shlex.quote(path)})"
                )
        else:
            checks.append(f"timeout {int(timeout_s)} test -e {shlex.quote(path)}")
    return " && ".join(checks)


def du_timeout_for_item(item: str) -> int:
    return 60 if item == "model-raw" else 15


def build_whole_layer_base_json(run_id: str, world_size: int = WORLD_SIZE) -> str:
    _validate_world_size(world_size)
    clean_id = validate_run_id(run_id)
    payload = {
        **DEFAULT_WHOLE_LAYER_BASE,
        "tree": f"/dev/shm/{clean_id}/builds/chain.so",
    }
    return json.dumps(payload, indent=2) + "\n"


def build_container_env_script(
    iso_env: Mapping[str, str],
    host_ifname: str = DEFAULT_IFNAME,
    world_size: int | None = None,
) -> str:
    ws = _infer_env_world_size(iso_env, world_size)
    verify_isolation_env(iso_env, world_size=ws)
    run_id = iso_env["TILERT_E2E_RUN_ID"]
    run_dir = iso_env["TILERT_RUN_DIR"]
    verify_shm_isolation(run_dir, run_id)
    py_paths = ":".join(
        [
            "/work/source",
            "/assets/vllm-unified",
            "/work/runtime-sfa-token-mlatail",
            "/assets/vendor-hook-runtime212-int8/vendor-hook-pinned/vendor-metadata",
            "/assets/vendor-hook-runtime212-int8/vendor-hook-deps/transformers519",
            "/assets/vendor-hook-runtime212-int8/vendor-hook-deps/python",
            "/opt/ascend/cann-9.2.0/python/site-packages",
            "/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe",
        ]
    )
    ld_paths = ":".join(
        [
            "/opt/ascend/cann-9.2.0/aarch64-linux/lib64",
            "/assets/shmem-capacity",
            "/assets/shmem-lib/lib",
            "/opt/openEuler/gcc-toolset-12/root/usr/lib64",
            "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib",
            "/opt/ascend/cann-9.2.0/lib64",
            "/opt/ascend/cann-9.2.0/lib64/plugin/opskernel",
            "/opt/ascend/cann-9.2.0/lib64/plugin/nnengine",
            (
                "/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/"
                "op_tiling/lib/linux/aarch64"
            ),
            "/usr/local/Ascend/driver/lib64",
            "/usr/local/Ascend/driver/lib64/common",
            "/usr/local/Ascend/driver/lib64/driver",
        ]
    )
    bin_paths = ":".join(
        [
            "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin",
            "/opt/ascend/cann-9.2.0/bin",
            "/opt/ascend/cann-9.2.0/tools/ccec_compiler/bin",
            "/opt/ascend/cann-9.2.0/tools/profiler/bin",
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ]
    )
    ascend_custom = ":".join(
        [
            "/opt/kernel-overlay/vendors/custom_transformer",
            (
                "/work/runtime-sfa-token-mlatail/vllm_ascend/"
                "_cann_ops_custom/vendors/custom_transformer"
            ),
        ]
    )
    cmake_pfx = (
        "/opt/ascend/cann-9.2.0/toolkit/tools/tikicpulib/lib/cmake:"
        "/opt/ascend/cann-9.2.0/lib64/cmake"
    )
    host_vis = iso_env["ASCEND_RT_VISIBLE_DEVICES"]
    logical_vis = ",".join(str(idx) for idx in range(len(host_vis.split(","))))
    exports: list[tuple[str, str]] = [
        ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_HUB_OFFLINE", "1"),
        ("TILERT_E2E_RUN_ID", iso_env["TILERT_E2E_RUN_ID"]),
        ("TILERT_RUN_DIR", run_dir),
        ("TILERT_JOB_DIR", iso_env["TILERT_JOB_DIR"]),
        ("TILERT_BENCH_ROOT", iso_env["TILERT_BENCH_ROOT"]),
        ("TILERT_BUILD_DIR", f"{run_dir}/builds"),
        ("HCCL_SOCKET_IFNAME", host_ifname),
        ("GLOO_SOCKET_IFNAME", host_ifname),
        ("HCCL_DETERMINISTIC", iso_env["HCCL_DETERMINISTIC"]),
        ("HCCL_OP_EXPANSION_MODE", iso_env["HCCL_OP_EXPANSION_MODE"]),
        ("HCCL_BUFFSIZE", "64"),
        ("HCCL_EXEC_TIMEOUT", "120"),
        ("MASTER_PORT", iso_env["MASTER_PORT"]),
        ("TILERT_WHOLE_LAYER_ENDPOINT", iso_env["TILERT_WHOLE_LAYER_ENDPOINT"]),
        ("TILERT_ONESIDED_ENDPOINT", iso_env["TILERT_ONESIDED_ENDPOINT"]),
        ("HCCL_NPU_SOCKET_PORT_RANGE", iso_env["HCCL_NPU_SOCKET_PORT_RANGE"]),
        ("ASCEND_RT_VISIBLE_DEVICES", logical_vis),
        ("TILERT_HOST_VISIBLE_DEVICES", host_vis),
        ("WORLD_SIZE", str(ws)),
        ("ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT", "1"),
        ("ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT", "5"),
        ("ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE", "0"),
        ("ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE", "0"),
        ("ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE", "0"),
        ("ATB_COMPARE_TILING_EVERY_KERNEL", "0"),
        ("ATB_MATMUL_SHUFFLE_K_ENABLE", "1"),
        ("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "1"),
        ("ATB_SHARE_MEMORY_NAME_SUFFIX", iso_env["ATB_SHARE_MEMORY_NAME_SUFFIX"]),
        ("TILERT_SHMEM_LIB_DIR", "/assets/shmem-capacity"),
        ("VLLM_CACHE_ROOT", iso_env["VLLM_CACHE_ROOT"]),
        ("TORCHINDUCTOR_CACHE_DIR", f"{run_dir}/inductor_cache"),
        ("TRITON_CACHE_DIR", f"{run_dir}/triton_cache"),
        ("XDG_CACHE_HOME", f"{run_dir}/xdg_cache"),
        ("ASCEND_PROCESS_LOG_PATH", f"{run_dir}/ascend_log"),
        ("ASCEND_SLOG_PRINT_TO_STDOUT", "0"),
        ("ASCEND_GLOBAL_LOG_LEVEL", "3"),
        ("TMPDIR", f"{run_dir}/tmp"),
        ("ASCEND_HOME_PATH", "/opt/ascend/cann-9.2.0"),
        ("ASCEND_TOOLKIT_HOME", "/opt/ascend/cann-9.2.0"),
        ("ASCEND_OPP_PATH", "/opt/ascend/cann-9.2.0/opp"),
        ("ASCEND_AICPU_PATH", "/opt/ascend/cann-9.2.0"),
        ("TOOLCHAIN_HOME", "/opt/ascend/cann-9.2.0/toolkit"),
        ("CMAKE_PREFIX_PATH", cmake_pfx),
        ("ASCEND_CUSTOM_OPP_PATH", ascend_custom),
        ("ATB_HOME_PATH", "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1"),
        ("VLLM_PLUGINS", "ascend,ascend_model"),
        ("PYTHONDONTWRITEBYTECODE", "1"),
        ("LCCL_DETERMINISTIC", "0"),
        ("LCCL_PARALLEL", "0"),
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("PYTHONPATH", py_paths),
        ("LD_LIBRARY_PATH", ld_paths),
        ("PATH", bin_paths),
    ]
    if "TILERT_TARGET_HOST" in iso_env:
        exports.append(("TILERT_TARGET_HOST", iso_env["TILERT_TARGET_HOST"]))
    lines = ["unset PROFILER_SAMPLECONFIG PROFILING_MODE ASCEND_LAUNCH_BLOCKING"]
    lines.extend(f"export {k}={shlex.quote(v)}" for k, v in exports)
    lines.extend(
        [
            'if [ -f "${TILERT_BUILD_DIR}/shmem/tilert_shmem_bf16.so" ] '
            '&& [ -f "${TILERT_BUILD_DIR}/shmem/libtp_push_kernel.so" ]; then',
            '  export TILERT_SHMEM_LIB_DIR="${TILERT_BUILD_DIR}/shmem"',
            '  export LD_LIBRARY_PATH="${TILERT_SHMEM_LIB_DIR}:${LD_LIBRARY_PATH}"',
            "fi",
        ]
    )
    return "\n".join(lines) + "\n"


def build_docker_run_cmd(
    run_id: str,
    visible_devices: Sequence[int],
    image: str = DEFAULT_IMAGE,
    model_host_path: str | None = None,
    extra_ro_mounts: Sequence[tuple[str, str]] = (),
    privileged: bool = False,
    world_size: int = WORLD_SIZE,
) -> list[str]:
    ws = _validate_world_size(world_size)
    clean_id = validate_run_id(run_id)
    shm_dir = f"/dev/shm/{clean_id}"
    verify_shm_isolation(shm_dir, clean_id)
    dies = tuple(visible_devices)
    if (
        len(dies) != ws
        or len(set(dies)) != ws
        or any(isinstance(d, bool) or not isinstance(d, int) or not (0 <= d < 16) for d in dies)
    ):
        raise IsolationError(f"visible_devices must be {ws} unique ints in 0..15")
    cmd = [
        "docker",
        "run",
        "-d",
        "--init",
        "--name",
        clean_id,
        "--entrypoint",
        "",
        "--network=host",
        "--ipc=host",
        "--cgroupns=host",
        "--runtime=ascend",
        "--security-opt",
        "label=disable",
        "--shm-size=64g",
    ]
    if privileged:
        cmd.append("--privileged")
    for die in dies:
        cmd.append(f"--device=/dev/davinci{die}")
    for dev in SHARED_DEVICES:
        cmd.append(f"--device={dev}")
    mounts: list[tuple[str, str, str]] = [
        (f"{shm_dir}/source", "/work/source", "ro"),
        (shm_dir, shm_dir, "rw"),
        (model_host_path or SFS_MODEL_DIR, "/work/model-raw", "ro"),
    ]
    for h_path, c_path in DEFAULT_STATIC_RO_MOUNTS:
        mounts.append((h_path, c_path, "ro"))
    for h_path, c_path in extra_ro_mounts:
        mounts.append((h_path, c_path, "ro"))
    for h_path, c_path, mode in mounts:
        suffix = "" if mode == "rw" else ":ro"
        cmd.extend(["-v", f"{h_path}:{c_path}{suffix}"])
    cmd.extend([image, "sleep", "infinity"])
    return cmd


def build_docker_exec_cmd(
    run_id: str,
    inner_cmd: str,
    sfs_timeout_s: int = 5,
    probes: Sequence[str] = CONTAINER_SFS_PROBES,
    prestage_ckpt: bool | None = None,
) -> list[str]:
    clean_id = validate_run_id(run_id)
    shm_dir = f"/dev/shm/{clean_id}"
    verify_shm_isolation(shm_dir, clean_id)
    probe_cmd = build_container_sfs_probe_script(
        probes,
        timeout_s=sfs_timeout_s,
        prestage_ckpt=prestage_ckpt,
    )
    script = "\n".join(
        [
            "set -euo pipefail",
            f"source {shlex.quote(shm_dir)}/env.sh",
            "unset PROFILER_SAMPLECONFIG PROFILING_MODE ASCEND_LAUNCH_BLOCKING",
            f"mkdir -p {shlex.quote(shm_dir)}/tmp && cd {shlex.quote(shm_dir)}/tmp",
            probe_cmd,
            inner_cmd,
        ]
    )
    return ["docker", "exec", clean_id, "bash", "-c", script]


def build_teardown_cmd(run_id: str) -> str:
    clean_id = validate_run_id(run_id)
    shm_dir = f"/dev/shm/{clean_id}"
    verify_shm_isolation(shm_dir, clean_id)
    return (
        f"docker rm -f {shlex.quote(clean_id)} >/dev/null 2>&1 || true; "
        f"rm -rf {shlex.quote(shm_dir)}"
    )


def build_remote_port_check_cmd(
    env: Mapping[str, str],
    world_size: int | None = None,
) -> str:
    ws = _infer_env_world_size(env, world_size)
    verify_isolation_env(env, world_size=ws)
    p_master = int(env["MASTER_PORT"])
    p_wl = int(env["TILERT_WHOLE_LAYER_ENDPOINT"].rsplit(":", 1)[1])
    p_os = int(env["TILERT_ONESIDED_ENDPOINT"].rsplit(":", 1)[1])
    lo_s, hi_s = env["HCCL_NPU_SOCKET_PORT_RANGE"].split("-", 1)
    lo, hi = int(lo_s), int(hi_s)
    py_check = (
        "import subprocess, sys; "
        "out = subprocess.check_output(['ss', '-Htan'], text=True); "
        f"fixed = {{{p_master}, {p_wl}, {p_os}}}; "
        "used = set(); "
        "[used.add(int(tok.rsplit(':', 1)[1])) for ln in out.splitlines() "
        "for tok in ln.split()[3:5] if ':' in tok and tok.rsplit(':', 1)[1].isdigit()]; "
        f"bad = sorted((used & fixed) | {{p for p in used if {lo} <= p <= {hi}}}); "
        "sys.exit(0 if not bad else 1)"
    )
    return f"python3 -c {shlex.quote(py_check)}"


def verify_npu_smi_dies(
    npu_smi_output: str,
    visible_devices: Sequence[int],
    *,
    max_hbm_pct: float | None = None,
) -> dict[int, float]:
    dies = {int(d) for d in visible_devices}
    busy: set[int] = set()
    hbm_exact: dict[int, float] = {}
    hbm_pcts: dict[int, float] = {}
    procs = False
    saw_proc_header = False
    has_mem_col = False
    cur_npu: int | None = None
    for ln in npu_smi_output.splitlines():
        if "Process id" in ln:
            procs = True
            saw_proc_header = True
            continue
        if "|" not in ln:
            continue
        parts = ln.split("|")
        if procs:
            if len(parts) >= 4 and parts[2].strip().isdigit():
                toks = parts[1].split()
                if len(toks) >= 2 and toks[0].isdigit() and toks[1].isdigit():
                    busy.add(int(toks[0]) * 2 + int(toks[1]))
            continue
        if "Memory-Usage" in ln and "HBM-Usage" in ln:
            has_mem_col = True
            continue
        if len(parts) < 4:
            continue
        toks = parts[1].split()
        if len(toks) >= 2 and toks[0].isdigit() and "Ascend" in toks[1]:
            cur_npu = int(toks[0])
            continue
        if cur_npu is not None and len(toks) >= 2 and toks[0].isdigit() and toks[1].isdigit():
            die = cur_npu * 2 + int(toks[0])
            usage_toks = re.sub(r"\s*/\s*", "/", parts[3]).split()
            ratio_cnt = sum(1 for tok in usage_toks if "/" in tok)
            m = re.fullmatch(r"(\d+)/(\d+)", usage_toks[-1]) if usage_toks else None
            if m:
                used, total = int(m.group(1)), int(m.group(2))
                if total > 0 and (ratio_cnt >= 2 or (not has_mem_col and total >= 8192)):
                    exact = (used * 100.0) / total
                    hbm_exact[die] = exact
                    hbm_pcts[die] = round(exact, 4)
    if not saw_proc_header:
        raise IsolationError("Missing npu-smi Process id table")
    ov = sorted(dies & busy)
    if ov:
        raise IsolationError(f"Requested dies have running NPU processes: {ov}")
    if max_hbm_pct is not None:
        missing = sorted(d for d in dies if d not in hbm_exact)
        if missing:
            raise IsolationError(f"Missing npu-smi HBM telemetry for dies: {missing}")
        high = {d: hbm_exact[d] for d in sorted(dies) if hbm_exact[d] > max_hbm_pct}
        if high:
            raise IsolationError(
                f"Alarm die check rejected HBM usage > {max_hbm_pct}%: {high}"
            )
    return {d: hbm_pcts[d] for d in sorted(dies) if d in hbm_pcts}


def build_remote_die_check_cmd(visible_devices: Sequence[int]) -> str:
    dies_csv = ",".join(str(int(d)) for d in visible_devices)
    py_check = (
        "import re, subprocess, sys; "
        f"dies = {{{dies_csv}}}; "
        "out = subprocess.check_output(['npu-smi', 'info'], text=True); "
        "procs = False; busy = set(); "
        "\nfor ln in out.splitlines():\n"
        "  if 'Process id' in ln: procs = True; continue\n"
        "  if procs and '|' in ln:\n"
        "    parts = ln.split('|')\n"
        "    if len(parts) >= 4 and parts[2].strip().isdigit():\n"
        "      toks = parts[1].split()\n"
        "      if len(toks) >= 2 and toks[0].isdigit() and toks[1].isdigit():\n"
        "        busy.add(int(toks[0]) * 2 + int(toks[1]))\n"
        "ov = sorted(dies & busy); "
        "sys.exit(0 if not ov else 1)"
    )
    return f"python3 -c {shlex.quote(py_check)}"


def _load_ssh_passwords() -> dict[str, str]:
    cred_file = Path(
        os.environ.get("TRT_SSH_CREDENTIALS", Path.home() / ".config/kbenv/passwords")
    )
    creds: dict[str, str] = {}
    if cred_file.is_file():
        for line in cred_file.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].strip():
                creds[parts[0].strip()] = parts[1].strip()
    return creds


_SSH_BATCH_CACHE: dict[tuple[str, bool, str], list[str] | None] = {}


def _ssh_batch_opts(host: str) -> list[str] | None:
    force_pw = os.environ.get("TILERT_SSH_FORCE_PASSWORD", "").lower() in ("1", "true")
    codex_key = Path.home() / ".ssh/id_ed25519_a3_a5_codex"
    cache_key = (host, force_pw, str(codex_key))
    if cache_key in _SSH_BATCH_CACHE:
        cached = _SSH_BATCH_CACHE[cache_key]
        return list(cached) if cached is not None else None
    base_opts = ["-o", "BatchMode=yes"]
    probe = subprocess.run(
        ["ssh", *base_opts, "-o", "ConnectTimeout=5", f"root@{host}", "true"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        _SSH_BATCH_CACHE[cache_key] = base_opts
        return list(base_opts)
    if not force_pw and codex_key.is_file():
        key_opts = [
            "-i",
            str(codex_key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        probe_key = subprocess.run(
            ["ssh", *key_opts, "-o", "ConnectTimeout=5", f"root@{host}", "true"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe_key.returncode == 0:
            _SSH_BATCH_CACHE[cache_key] = key_opts
            return list(key_opts)
    _SSH_BATCH_CACHE[cache_key] = None
    return None


def _ssh_base_cmd(host: str) -> tuple[list[str], dict[str, str] | None]:
    batch_opts = _ssh_batch_opts(host)
    if batch_opts is not None:
        return ["ssh", *batch_opts, "-o", "ConnectTimeout=10", f"root@{host}"], None
    creds = _load_ssh_passwords()
    if host not in creds:
        raise IsolationError(f"SSH BatchMode failed and no password found for {host}")
    env = dict(os.environ)
    env["SSHPASS"] = creds[host]
    cmd = [
        "sshpass",
        "-e",
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "ConnectTimeout=10",
        f"root@{host}",
    ]
    return cmd, env


def _scp_cmd(host: str, src: str, dst: str) -> tuple[list[str], dict[str, str] | None]:
    batch_opts = _ssh_batch_opts(host)
    if batch_opts is not None:
        return ["scp", "-q", "-r", *batch_opts, f"root@{host}:{src}", dst], None
    creds = _load_ssh_passwords()
    if host not in creds:
        raise IsolationError(f"SCP BatchMode failed and no password found for {host}")
    env = dict(os.environ)
    env["SSHPASS"] = creds[host]
    cmd = [
        "sshpass",
        "-e",
        "scp",
        "-q",
        "-r",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        f"root@{host}:{src}",
        dst,
    ]
    return cmd, env


def remote_ssh(
    host: str,
    remote_cmd: str,
    *,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    assert_host_allowed(host)
    base, env = _ssh_base_cmd(host)
    return subprocess.run(
        [*base, remote_cmd],
        input=input_bytes,
        env=env,
        capture_output=True,
        check=check,
    )


def _kbenv_hosts(kbenv_bin: str) -> list[Any]:
    proc = subprocess.run(
        [kbenv_bin, "--once"],
        capture_output=True,
        text=True,
        check=True,
        timeout=90.0,
    )
    return parse_kbenv_output(proc.stdout)


def probe_host(
    host: str | None,
    die_slice: str = "auto",
    image: str = DEFAULT_IMAGE,
    kbenv_bin: str = DEFAULT_KBENV_BIN,
    allow_idle_dies: bool = False,
    world_size: int = WORLD_SIZE,
    allow_alarm_dies: bool = False,
) -> dict[str, Any]:
    ws = _validate_world_size(world_size)
    hosts = _kbenv_hosts(kbenv_bin)
    selected, dies = select_host_and_dies(
        hosts,
        target_host=host,
        die_slice=die_slice,
        allow_idle_dies=allow_idle_dies,
        world_size=ws,
        allow_alarm_dies=allow_alarm_dies,
    )
    sfs_cmd = build_sfs_probe_script(SFS_PROBE_TARGETS, timeout_s=5)
    check_script = "\n".join(
        [
            "set -euo pipefail",
            sfs_cmd,
            f"docker image inspect {shlex.quote(image)} >/dev/null",
            "echo '---DF---'",
            "df -B1 /dev/shm",
            "echo '---MEM---'",
            "cat /proc/meminfo",
            "echo '---PS---'",
            "docker ps --format '{{.Names}}'",
        ]
    )
    res = remote_ssh(selected.ip, check_script)
    out = res.stdout.decode("utf-8", errors="replace")
    df_part = out.split("---DF---\n", 1)[1].split("---MEM---\n", 1)[0]
    mem_part = out.split("---MEM---\n", 1)[1].split("---PS---\n", 1)[0]
    ps_part = out.split("---PS---\n", 1)[1]
    shm_avail = parse_df_avail_bytes(df_part)
    mem_total, mem_avail = parse_meminfo_bytes(mem_part)
    if allow_alarm_dies:
        smi_res = remote_ssh(selected.ip, "npu-smi info")
        verify_npu_smi_dies(
            smi_res.stdout.decode("utf-8", errors="replace"),
            dies,
            max_hbm_pct=ALARM_DIE_MAX_HBM_PCT,
        )
    else:
        remote_ssh(selected.ip, build_remote_die_check_cmd(dies))
    return {
        "host": selected.ip,
        "state": selected.state,
        "world_size": ws,
        "free_dies": selected.free_dies,
        "selected_dies": list(dies),
        "ram_pct": selected.ram_pct,
        "shm_avail_gib": round(shm_avail / (1024**3), 2),
        "mem_avail_gib": round(mem_avail / (1024**3), 2),
        "mem_total_gib": round(mem_total / (1024**3), 2),
        "running_containers": [ln for ln in ps_part.splitlines() if ln.strip()],
        "sfs_ok": True,
        "image_present": True,
    }


def prestage_host(
    host: str,
    items: Sequence[str],
    prestage_root: str = DEFAULT_PRESTAGE_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    assert_host_allowed(host)
    for item in items:
        if item not in PRESTAGE_SPECS:
            raise ValueError(f"Unknown prestage item: {item!r}")
    df_res = remote_ssh(host, "df -B1 /dev/shm")
    mem_res = remote_ssh(host, "cat /proc/meminfo")
    shm_free = parse_df_avail_bytes(df_res.stdout.decode("utf-8"))
    mem_total, mem_avail = parse_meminfo_bytes(mem_res.stdout.decode("utf-8"))
    results: list[dict[str, Any]] = []
    cur_shm = shm_free
    cur_mem = mem_avail
    for item in items:
        sfs_src, rel_name = PRESTAGE_SPECS[item]
        dst = f"{prestage_root.rstrip('/')}/{rel_name}"
        marker = f"{dst}.prestage_complete.json"
        has_marker = remote_ssh(host, f"test -f {shlex.quote(marker)}", check=False).returncode == 0
        if has_marker:
            results.append({"item": item, "dst": dst, "status": "already_staged"})
            continue
        du_timeout = du_timeout_for_item(item)
        du_out = remote_ssh(
            host,
            f"set -euo pipefail; timeout {du_timeout} du -sb {shlex.quote(sfs_src)} | "
            "awk '{print $1}'",
        ).stdout.decode("utf-8").strip()
        size_bytes = int(du_out)
        ok, reason = check_prestage_capacity(cur_shm, mem_total, cur_mem, size_bytes)
        if not ok:
            raise IsolationError(f"Prestage rejected for {item} on {host}: {reason}")
        if dry_run:
            results.append(
                {
                    "item": item,
                    "src": sfs_src,
                    "dst": dst,
                    "size_gib": round(size_bytes / (1024**3), 3),
                    "status": f"dry_run:{reason}",
                }
            )
            cur_shm -= size_bytes
            cur_mem -= size_bytes
            continue
        tmp_dst = f"{dst}.tmp.{os.getpid()}"
        stage_script = "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(prestage_root)}",
                f"rm -rf {shlex.quote(tmp_dst)}",
                f"cp -a {shlex.quote(sfs_src)} {shlex.quote(tmp_dst)}",
                f"rm -rf {shlex.quote(dst)}",
                f"mv {shlex.quote(tmp_dst)} {shlex.quote(dst)}",
                f"printf '{{\"item\":\"%s\",\"bytes\":%s}}\\n' {shlex.quote(item)} {size_bytes} "
                f"> {shlex.quote(marker)}",
            ]
        )
        remote_ssh(host, stage_script)
        cur_shm -= size_bytes
        cur_mem -= size_bytes
        results.append(
            {
                "item": item,
                "src": sfs_src,
                "dst": dst,
                "size_gib": round(size_bytes / (1024**3), 3),
                "status": "staged",
            }
        )
    return {"host": host, "prestage_root": prestage_root, "items": results}


def _detect_remote_prestage_mounts(
    host: str,
    prestage_root: str = DEFAULT_PRESTAGE_ROOT,
    shared_ro_root: str = SHARED_RO_PRESTAGE_ROOT,
) -> tuple[str | None, list[tuple[str, str]]]:
    p_root = prestage_root.rstrip("/")
    s_root = shared_ro_root.rstrip("/")
    py_probe = (
        "import json, pathlib; "
        "ok_m = lambda p: ("
        "pathlib.Path(p).is_file() "
        "and isinstance((d := json.loads(pathlib.Path(p).read_text())), dict) "
        "and d.get('status') not in ('failed', 'error')"
        "); "
        f"pr, sr = pathlib.Path({p_root!r}), pathlib.Path({s_root!r}); "
        "out = {'model': None, 'mounts': []}; "
        "m_p, m_s = pr / 'GLM-5.2-w4a8c8', sr / 'GLM-5.2-w4a8c8'; "
        "out['model'] = str(m_p) if (m_p.is_dir() and ok_m(str(m_p) + '.prestage_complete.json')) "
        "else (str(m_s) if (m_s.is_dir() and ok_m(str(m_s) + '.prestage_complete.json')) "
        "else None); "
        "cands = ["
        "(pr / 'cal_l78_v3_hybrid_best.pt', "
        "str(pr / 'cal_l78_v3_hybrid_best.pt') + '.prestage_complete.json', "
        "sr / 'dspine-glm52/checkpoints/cal_l78_v3_hybrid_best.pt', "
        "str(sr / 'dspine-glm52.prestage_complete.json'), "
        "'/prestage/cal_l78_v3_hybrid_best.pt'), "
        "(pr / 'case-01-coding', "
        "str(pr / 'case-01-coding') + '.prestage_complete.json', "
        "sr / 'cases-8k1k/case-01-coding', "
        "str(sr / 'cases-8k1k.prestage_complete.json'), "
        "'/prestage/case-01-coding'), "
        "(pr / 'mtp-prefix-rank-0.pt', "
        "str(pr / 'mtp-prefix-rank-0.pt') + '.prestage_complete.json', "
        "sr / 'dspine-glm52/runtime_bundle_v5/agy_tp8/mtp-prefix-rank-0.pt', "
        "str(sr / 'dspine-glm52.prestage_complete.json'), "
        "'/prestage/mtp-prefix-rank-0.pt'), "
        "(sr / 'cases-8k1k', str(sr / 'cases-8k1k.prestage_complete.json'), "
        "sr / 'cases-8k1k', str(sr / 'cases-8k1k.prestage_complete.json'), "
        "'/prestage/cases-8k1k')"
        "]; "
        "out['mounts'] = ["
        "[str(a), dst] if (a.exists() and ok_m(am)) else [str(b), dst] "
        "for a, am, b, bm, dst in cands "
        "if (a.exists() and ok_m(am)) or (b.exists() and ok_m(bm))"
        "]; "
        "print(json.dumps(out))"
    )
    res = remote_ssh(host, f"python3 -c {shlex.quote(py_probe)}", check=False)
    if res.returncode != 0:
        return None, []
    try:
        parsed = json.loads(res.stdout.decode("utf-8", errors="replace").strip())
    except json.JSONDecodeError:
        return None, []
    model_override = parsed.get("model")
    extra_mounts = [(str(src), str(dst)) for src, dst in parsed.get("mounts", ())]
    return model_override, extra_mounts


def build_up_source_tar_cmd(repo_root: Path = REPO_ROOT) -> list[str]:
    return [
        "tar",
        "--exclude=.git",
        "--exclude=tests/.artifacts",
        "--exclude=docs/evidence",
        "--exclude=docs/assets",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        "--exclude=*.pyc",
        "-czf",
        "-",
        "-C",
        str(repo_root),
        ".",
    ]


def up_container(
    host: str | None = None,
    run_id: str | None = None,
    die_slice: str = "auto",
    port_base: int | None = None,
    image: str = DEFAULT_IMAGE,
    kbenv_bin: str = DEFAULT_KBENV_BIN,
    use_prestage: bool = True,
    allow_idle_dies: bool = False,
    privileged: bool = False,
    world_size: int = WORLD_SIZE,
    allow_alarm_dies: bool = False,
) -> dict[str, Any]:
    ws = _validate_world_size(world_size)
    hosts = _kbenv_hosts(kbenv_bin)
    selected, dies = select_host_and_dies(
        hosts,
        target_host=host,
        die_slice=die_slice,
        allow_idle_dies=allow_idle_dies,
        world_size=ws,
        allow_alarm_dies=allow_alarm_dies,
    )
    clean_id = validate_run_id(run_id or generate_run_id())
    remote_ssh(selected.ip, build_sfs_probe_script(SFS_PROBE_TARGETS, timeout_s=5))
    remote_ssh(selected.ip, f"docker image inspect {shlex.quote(image)} >/dev/null")
    if allow_alarm_dies:
        smi_res = remote_ssh(selected.ip, "npu-smi info")
        verify_npu_smi_dies(
            smi_res.stdout.decode("utf-8", errors="replace"),
            dies,
            max_hbm_pct=ALARM_DIE_MAX_HBM_PCT,
        )
    else:
        remote_ssh(selected.ip, build_remote_die_check_cmd(dies))

    reserved: list[int] = []
    iso_env: dict[str, str] | None = None
    for _ in range(8):
        cand_port = allocate_free_port_base(reserved) if port_base is None else port_base
        cand_env = allocate_isolation_env(
            clean_id,
            port_base=cand_port,
            target_host=selected.ip,
            visible_devices=dies,
            world_size=ws,
        )
        chk = remote_ssh(
            selected.ip, build_remote_port_check_cmd(cand_env, world_size=ws), check=False
        )
        if chk.returncode == 0:
            iso_env = cand_env
            break
        if port_base is not None:
            raise IsolationError(f"Requested port_base {port_base} busy on {selected.ip}")
        reserved.append(cand_port)
    if iso_env is None:
        raise IsolationError(f"Unable to reserve free port block on {selected.ip}")

    ifname_res = remote_ssh(
        selected.ip,
        "ip -o link show | awk -F': ' '$2 != \"lo\" {print $2; exit}'",
        check=False,
    )
    ifname = ifname_res.stdout.decode("utf-8").strip() or DEFAULT_IFNAME
    shm_dir = iso_env["TILERT_RUN_DIR"]
    dirs_cmd = "mkdir -p " + " ".join(shlex.quote(f"{shm_dir}/{sub}") for sub in RUN_SUBDIRS)
    tar_proc = subprocess.run(
        build_up_source_tar_cmd(REPO_ROOT),
        capture_output=True,
        check=True,
    )
    remote_ssh(
        selected.ip,
        f"{dirs_cmd} && tar -xzf - -C {shlex.quote(shm_dir)}/source",
        input_bytes=tar_proc.stdout,
    )

    env_txt = build_container_env_script(iso_env, host_ifname=ifname, world_size=ws)
    wl_txt = build_whole_layer_base_json(clean_id, world_size=ws)
    cfg_cmd = (
        f"cat <<'__TILERT_ENV__' > {shlex.quote(shm_dir)}/env.sh\n"
        f"{env_txt}\n__TILERT_ENV__\n"
        f"cat <<'__TILERT_WL__' > {shlex.quote(shm_dir)}/whole_layer_base.json\n"
        f"{wl_txt}\n__TILERT_WL__"
    )
    remote_ssh(selected.ip, cfg_cmd)

    model_override = None
    extra_ro: list[tuple[str, str]] = []
    if use_prestage:
        model_override, extra_ro = _detect_remote_prestage_mounts(selected.ip)
    run_cmd = build_docker_run_cmd(
        clean_id,
        dies,
        image=image,
        model_host_path=model_override,
        extra_ro_mounts=extra_ro,
        privileged=privileged,
        world_size=ws,
    )
    remote_ssh(selected.ip, " ".join(shlex.quote(tok) for tok in run_cmd))
    return {
        "run_id": clean_id,
        "host": selected.ip,
        "world_size": ws,
        "visible_devices": list(dies),
        "port_base": int(iso_env["MASTER_PORT"]),
        "shm_dir": shm_dir,
        "prestage_model": model_override,
        "prestage_ckpt": any(dst == PRESTAGE_CKPT_PATH for _, dst in extra_ro),
        "prestage_mounts": [dst for _, dst in extra_ro],
    }


def build_shmem_libs_script(
    out_dir: str | None = None,
    src_dir: str = "/work/source/tests/glm52/shmem",
) -> str:
    target = '"${TILERT_BUILD_DIR}/shmem"' if out_dir is None else shlex.quote(out_dir)
    q_src = shlex.quote(src_dir)
    lines = [
        "set -euo pipefail",
        f"OUT={target}",
        f"SRC={q_src}",
        'CANN="${ASCEND_HOME_PATH:-/opt/ascend/cann-9.2.0}"',
        'mkdir -p "$OUT"',
        (
            'DIGEST=$(python3 -c "import hashlib, pathlib; '
            "s = pathlib.Path('$SRC'); h = hashlib.sha256(); "
            "[h.update((s / n).read_bytes()) for n in "
            "('tp_push_kernel.cpp', 'tp_push_kernel.h', 'tilert_shmem_bf16.cpp') "
            'if (s / n).is_file()]; print(h.hexdigest())")'
        ),
        'MAN="$OUT/shmem.manifest.json"',
        'for base_so in libshmem.so libshmem_utils.so aclshmem_torch.so '
        'libkv_shuffle_kernel.so liballgather_kernel.so; do',
        '  if [ ! -e "$OUT/$base_so" ]; then',
        '    if [ -e "/assets/shmem-capacity/$base_so" ]; then',
        '      ln -sfn "/assets/shmem-capacity/$base_so" "$OUT/$base_so"',
        '    elif [ -e "/assets/shmem-lib/lib/$base_so" ]; then',
        '      ln -sfn "/assets/shmem-lib/lib/$base_so" "$OUT/$base_so"',
        "    fi",
        "  fi",
        "done",
        'CUR=$(python3 -c "import json, pathlib; p = pathlib.Path(\'$MAN\'); '
        'print(json.loads(p.read_text()).get(\'source_sha256\', \'\') if p.is_file() else \'\')" '
        "|| true)",
        'if [ "$CUR" = "$DIGEST" ] && [ -f "$OUT/libtp_push_kernel.so" ] '
        '&& [ -f "$OUT/tilert_shmem_bf16.so" ]; then',
        '  echo "SHMEM_BUILD_CACHED $OUT $DIGEST"',
        "else",
        (
            '  SITE=$(python3 -c "import importlib.util, pathlib; '
            'print(pathlib.Path(importlib.util.find_spec(\'torch\').origin).resolve().parents[1])")'
        ),
        (
            '  bisheng -xcce --npu-arch=dav-2201 -std=c++17 -O2 -w -fPIC '
            '-DL2_CACHE_HINT=1 -D__NPU_DEVICE__=1 -DASCENDC_DUMP=0 '
            '-Daclshmemx_sync_vec_all=aclshmemx_barrier_all_vec '
            '-I"$SRC" -I/assets/shmem-lib/include -I/assets/shmem-lib/src/device '
            '-I/assets/shmem-lib/src/host_device -I"$CANN/include" '
            '-I"$CANN/aarch64-linux/asc/include" -I"$CANN/aarch64-linux/tikcpp/tikcfw" '
            '-I"$CANN/aarch64-linux/tikcpp/tikcfw/impl" '
            '-I"$CANN/aarch64-linux/tikcpp/tikcfw/interface" '
            '-I"$CANN/aarch64-linux/tikcpp/tikcfw/lib" '
            '-c "$SRC/tp_push_kernel.cpp" -o "$OUT/tp_push_kernel.o" &'
        ),
        (
            '  c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 '
            '-I"$SITE/torch/include" -I"$SITE/torch/include/torch/csrc/api/include" '
            '-I"$SITE/torch_npu/include" -I"$CANN/include" '
            '-c "$SRC/tilert_shmem_bf16.cpp" -o "$OUT/tilert_shmem_bf16.o" &'
        ),
        "  wait",
        (
            '  bisheng --cce-fatobj-link -shared "$OUT/tp_push_kernel.o" '
            '-o "$OUT/libtp_push_kernel.so.tmp" '
            '-L/assets/shmem-lib/lib -lshmem -lshmem_utils '
            '-L"$CANN/aarch64-linux/lib64" -L"$CANN/lib64" '
            '-lascendcl -lruntime -lunified_dlog -lc_sec -lregister -lprofapi -lplatform'
        ),
        '  mv -f "$OUT/libtp_push_kernel.so.tmp" "$OUT/libtp_push_kernel.so"',
        (
            '  c++ -shared "$OUT/tilert_shmem_bf16.o" -o "$OUT/tilert_shmem_bf16.so.tmp" '
            '-L"$OUT" -ltp_push_kernel -L/assets/shmem-lib/lib -lshmem '
            '-L"$CANN/aarch64-linux/lib64" -lruntime '
            '-L"$SITE/torch/lib" -L"$SITE/torch_npu/lib" -ltorch -ltorch_cpu -lc10 -ltorch_npu'
        ),
        '  mv -f "$OUT/tilert_shmem_bf16.so.tmp" "$OUT/tilert_shmem_bf16.so"',
        '  printf \'{"source_sha256":"%s"}\\n\' "$DIGEST" > "$MAN"',
        '  echo "SHMEM_BUILD_OK $OUT $DIGEST"',
        "fi",
    ]
    return "\n".join(lines)


def build_shmem_in_container(
    host: str,
    run_id: str,
    out_dir: str | None = None,
) -> int:
    return exec_in_container(host, run_id, build_shmem_libs_script(out_dir=out_dir), stream=True)


def exec_in_container(
    host: str,
    run_id: str,
    inner_cmd: str,
    sfs_timeout_s: int = 5,
    stream: bool = True,
) -> int:
    assert_host_allowed(host)
    docker_cmd = build_docker_exec_cmd(run_id, inner_cmd, sfs_timeout_s=sfs_timeout_s)
    remote_str = " ".join(shlex.quote(tok) for tok in docker_cmd)
    base, env = _ssh_base_cmd(host)
    if stream:
        proc = subprocess.run([*base, remote_str], env=env, check=False)
        return proc.returncode
    proc = subprocess.run([*base, remote_str], env=env, capture_output=True, check=False)
    return proc.returncode


def down_container(host: str, run_id: str) -> dict[str, Any]:
    assert_host_allowed(host)
    teardown = build_teardown_cmd(run_id)
    remote_ssh(host, teardown)
    return {"host": host, "run_id": run_id, "torn_down": True}


def sync_source(
    host: str,
    run_id: str,
    paths: Sequence[str] = ("tilert", "tests/glm52"),
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    assert_host_allowed(host)
    clean_id = validate_run_id(run_id)
    shm_dir = f"/dev/shm/{clean_id}"
    verify_shm_isolation(shm_dir, clean_id)
    rel_paths = [p.strip("/") for p in paths if p.strip("/")]
    if not rel_paths:
        raise ValueError("sync_source requires at least one relative path")
    tar_proc = subprocess.run(
        [
            "tar",
            "--exclude=.git",
            "--exclude=tests/.artifacts",
            "--exclude=__pycache__",
            "--exclude=.pytest_cache",
            "--exclude=*.pyc",
            "-czf",
            "-",
            "-C",
            str(repo_root),
            *rel_paths,
        ],
        capture_output=True,
        check=True,
    )
    remote_ssh(
        host,
        f"test -d {shlex.quote(shm_dir)}/source && "
        f"tar -xzf - -C {shlex.quote(shm_dir)}/source",
        input_bytes=tar_proc.stdout,
    )
    return {
        "host": host,
        "run_id": clean_id,
        "shm_dir": shm_dir,
        "paths": rel_paths,
        "tar_bytes": len(tar_proc.stdout),
    }


def _resolve_dspine_stamp_path(
    stamp_path: Path | str | None = None,
    repo_root: Path = REPO_ROOT,
) -> Path:
    return Path(stamp_path) if stamp_path is not None else Path(repo_root) / DEFAULT_DSPINE_STAMP


def dspine_unit_digest(repo_root: Path = REPO_ROOT) -> str:
    root = Path(repo_root)
    files: list[Path] = []
    dspine_dir = root / "tilert/models/glm_5_2_ascend/dspine"
    if dspine_dir.is_dir():
        for cand in sorted(dspine_dir.rglob("*")):
            if cand.is_file() and "__pycache__" not in cand.parts and cand.suffix != ".pyc":
                files.append(cand)
    for rel in (
        "tilert/models/glm_5_2_ascend/weights.py",
        "tilert/models/glm_5_2_ascend/abi.py",
        "tests/glm52/test_dspine_unit.py",
    ):
        cand = root / rel
        if cand.is_file():
            files.append(cand)
    h = hashlib.sha256()
    seen: set[Path] = set()
    for path in sorted(files):
        if path in seen:
            continue
        seen.add(path)
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def read_dspine_unit_stamp(
    stamp_path: Path | str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any] | None:
    path = _resolve_dspine_stamp_path(stamp_path, repo_root=repo_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_dspine_unit_stamp(
    digest: str,
    *,
    passed: bool = True,
    source: str = "local",
    stamp_path: Path | str | None = None,
    repo_root: Path = REPO_ROOT,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    path = _resolve_dspine_stamp_path(stamp_path, repo_root=repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "digest": digest,
        "passed": bool(passed),
        "source": str(source),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def run_dspine_check(
    *,
    host: str | None = None,
    run_id: str | None = None,
    force: bool = False,
    stamp_path: Path | str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if bool(host) != bool(run_id):
        raise ValueError("Both --host and --run-id are required together")
    resolved_stamp = _resolve_dspine_stamp_path(stamp_path, repo_root=repo_root)
    digest = dspine_unit_digest(repo_root)
    stamp = read_dspine_unit_stamp(resolved_stamp, repo_root=repo_root)
    if (
        not force
        and stamp is not None
        and stamp.get("passed") is True
        and stamp.get("digest") == digest
    ):
        return {
            "status": "skipped",
            "returncode": 0,
            "digest": digest,
            "stamp_path": str(resolved_stamp),
            "stamp": stamp,
        }
    if host and run_id:
        rc = exec_in_container(
            host,
            run_id,
            "pytest -q -p no:cacheprovider /work/source/tests/glm52/test_dspine_unit.py",
            stream=True,
        )
        source = f"container:{host}:{run_id}"
    else:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                str(Path(repo_root) / "tests/glm52/test_dspine_unit.py"),
            ],
            check=False,
        )
        rc = proc.returncode
        source = "local"
    if rc != 0:
        return {
            "status": "failed",
            "returncode": rc,
            "digest": digest,
            "source": source,
            "stamp_path": str(resolved_stamp),
        }
    write_dspine_unit_stamp(digest, passed=True, source=source, stamp_path=resolved_stamp)
    return {
        "status": "passed",
        "returncode": 0,
        "digest": digest,
        "source": source,
        "stamp_path": str(resolved_stamp),
    }


def pull_from_container_shm(host: str, remote_path: str, local_path: Path) -> None:
    assert_host_allowed(host)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd, env = _scp_cmd(host, remote_path, str(local_path))
    subprocess.run(cmd, env=env, check=True)


def run_ephemeral(
    host: str | None,
    inner_cmd: str,
    run_id: str | None = None,
    die_slice: str = "auto",
    port_base: int | None = None,
    image: str = DEFAULT_IMAGE,
    kbenv_bin: str = DEFAULT_KBENV_BIN,
    keep_container: bool = False,
    pull_from: str | None = None,
    pull_to: str | None = None,
    use_prestage: bool = True,
    allow_idle_dies: bool = False,
    privileged: bool = False,
    world_size: int = WORLD_SIZE,
    allow_alarm_dies: bool = False,
) -> dict[str, Any]:
    info = up_container(
        host=host,
        run_id=run_id,
        die_slice=die_slice,
        port_base=port_base,
        image=image,
        kbenv_bin=kbenv_bin,
        use_prestage=use_prestage,
        allow_idle_dies=allow_idle_dies,
        privileged=privileged,
        world_size=world_size,
        allow_alarm_dies=allow_alarm_dies,
    )
    rc = 1
    try:
        rc = exec_in_container(info["host"], info["run_id"], inner_cmd, stream=True)
        if pull_from and pull_to:
            pull_from_container_shm(info["host"], pull_from, Path(pull_to))
    finally:
        if not keep_container:
            down_container(info["host"], info["run_id"])
    info["returncode"] = rc
    return info


def fanout_fleet(
    inner_cmd: str,
    max_slots: int = 2,
    dual_slot: bool = True,
    exclude_hosts: Sequence[str] = (),
    image: str = DEFAULT_IMAGE,
    kbenv_bin: str = DEFAULT_KBENV_BIN,
    world_size: int = WORLD_SIZE,
) -> list[dict[str, Any]]:
    ws = _validate_world_size(world_size)
    hosts = _kbenv_hosts(kbenv_bin)
    slots = select_fleet_slots(
        hosts,
        max_slots=max_slots,
        dual_slot=dual_slot,
        exclude_hosts=exclude_hosts,
        world_size=ws,
    )
    if not slots:
        raise IsolationError("No eligible A3 slots found for fanout")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futs = {}
        for idx, (ip, dies) in enumerate(slots):
            slice_str = ",".join(str(d) for d in dies)
            rid = generate_run_id(f"m21f{idx}")
            fut = pool.submit(
                run_ephemeral,
                ip,
                inner_cmd,
                rid,
                slice_str,
                None,
                image,
                kbenv_bin,
                False,
                None,
                None,
                True,
                world_size=ws,
            )
            futs[fut] = (ip, slice_str, rid)
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="subcmd", required=True)

    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--host", default=None)
    p_probe.add_argument("--die-slice", default="auto")
    p_probe.add_argument("--image", default=DEFAULT_IMAGE)
    p_probe.add_argument("--kbenv", default=DEFAULT_KBENV_BIN)
    p_probe.add_argument("--allow-idle-dies", action="store_true")
    p_probe.add_argument("--allow-alarm-dies", action="store_true")
    p_probe.add_argument(
        "--world-size", type=int, choices=SUPPORTED_WORLD_SIZES, default=WORLD_SIZE
    )

    p_fleet = sub.add_parser("probe-fleet")
    p_fleet.add_argument("--max-slots", type=int, default=None)
    p_fleet.add_argument("--no-dual-slot", action="store_true")
    p_fleet.add_argument("--exclude", default="")
    p_fleet.add_argument("--kbenv", default=DEFAULT_KBENV_BIN)
    p_fleet.add_argument(
        "--world-size", type=int, choices=SUPPORTED_WORLD_SIZES, default=WORLD_SIZE
    )

    p_stage = sub.add_parser("prestage")
    p_stage.add_argument("--host", required=True)
    p_stage.add_argument("--items", default="ckpt,case01,mtp-prefix")
    p_stage.add_argument("--prestage-root", default=DEFAULT_PRESTAGE_ROOT)
    p_stage.add_argument("--dry-run", action="store_true")

    p_up = sub.add_parser("up")
    p_up.add_argument("--host", default=None)
    p_up.add_argument("--run-id", default=None)
    p_up.add_argument("--die-slice", default="auto")
    p_up.add_argument("--port-base", type=int, default=None)
    p_up.add_argument("--image", default=DEFAULT_IMAGE)
    p_up.add_argument("--kbenv", default=DEFAULT_KBENV_BIN)
    p_up.add_argument("--no-prestage", action="store_true")
    p_up.add_argument("--allow-idle-dies", action="store_true")
    p_up.add_argument("--allow-alarm-dies", action="store_true")
    p_up.add_argument("--privileged", action="store_true")
    p_up.add_argument(
        "--world-size", type=int, choices=SUPPORTED_WORLD_SIZES, default=WORLD_SIZE
    )

    p_exec = sub.add_parser("exec")
    p_exec.add_argument("--host", required=True)
    p_exec.add_argument("--run-id", required=True)
    p_exec.add_argument("--cmd", required=True)
    p_exec.add_argument("--sfs-timeout", type=int, default=5)

    p_down = sub.add_parser("down")
    p_down.add_argument("--host", required=True)
    p_down.add_argument("--run-id", required=True)

    p_sync = sub.add_parser("sync-source")
    p_sync.add_argument("--host", required=True)
    p_sync.add_argument("--run-id", required=True)
    p_sync.add_argument("--paths", default="tilert,tests/glm52")

    p_dspine = sub.add_parser("dspine-check")
    p_dspine.add_argument("--host", default=None)
    p_dspine.add_argument("--run-id", default=None)
    p_dspine.add_argument("--force", action="store_true")
    p_dspine.add_argument("--stamp-path", default=None)

    p_shmem = sub.add_parser("build-shmem")
    p_shmem.add_argument("--host", required=True)
    p_shmem.add_argument("--run-id", required=True)
    p_shmem.add_argument("--out-dir", default=None)

    p_run = sub.add_parser("run")
    p_run.add_argument("--host", default=None)
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--die-slice", default="auto")
    p_run.add_argument("--port-base", type=int, default=None)
    p_run.add_argument("--image", default=DEFAULT_IMAGE)
    p_run.add_argument("--kbenv", default=DEFAULT_KBENV_BIN)
    p_run.add_argument("--cmd", required=True)
    p_run.add_argument("--keep-container", action="store_true")
    p_run.add_argument("--pull-from", default=None)
    p_run.add_argument("--pull-to", default=None)
    p_run.add_argument("--no-prestage", action="store_true")
    p_run.add_argument("--allow-idle-dies", action="store_true")
    p_run.add_argument("--allow-alarm-dies", action="store_true")
    p_run.add_argument("--privileged", action="store_true")
    p_run.add_argument(
        "--world-size", type=int, choices=SUPPORTED_WORLD_SIZES, default=WORLD_SIZE
    )

    p_fan = sub.add_parser("fanout")
    p_fan.add_argument("--cmd", required=True)
    p_fan.add_argument("--max-slots", type=int, default=2)
    p_fan.add_argument("--no-dual-slot", action="store_true")
    p_fan.add_argument("--exclude", default="")
    p_fan.add_argument("--image", default=DEFAULT_IMAGE)
    p_fan.add_argument("--kbenv", default=DEFAULT_KBENV_BIN)
    p_fan.add_argument(
        "--world-size", type=int, choices=SUPPORTED_WORLD_SIZES, default=WORLD_SIZE
    )

    args = parser.parse_args(argv)
    if args.subcmd == "probe":
        out = probe_host(
            args.host,
            die_slice=args.die_slice,
            image=args.image,
            kbenv_bin=args.kbenv,
            allow_idle_dies=args.allow_idle_dies,
            world_size=args.world_size,
            allow_alarm_dies=args.allow_alarm_dies,
        )
        print(json.dumps(out, indent=2))
        return 0
    if args.subcmd == "probe-fleet":
        excl = [x.strip() for x in args.exclude.split(",") if x.strip()]
        slots = select_fleet_slots(
            _kbenv_hosts(args.kbenv),
            max_slots=args.max_slots,
            dual_slot=not args.no_dual_slot,
            exclude_hosts=excl,
            world_size=args.world_size,
        )
        print(
            json.dumps(
                [{"host": ip, "dies": list(dies)} for ip, dies in slots],
                indent=2,
            )
        )
        return 0
    if args.subcmd == "prestage":
        items = [x.strip() for x in args.items.split(",") if x.strip()]
        out = prestage_host(
            args.host,
            items=items,
            prestage_root=args.prestage_root,
            dry_run=args.dry_run,
        )
        print(json.dumps(out, indent=2))
        return 0
    if args.subcmd == "up":
        out = up_container(
            host=args.host,
            run_id=args.run_id,
            die_slice=args.die_slice,
            port_base=args.port_base,
            image=args.image,
            kbenv_bin=args.kbenv,
            use_prestage=not args.no_prestage,
            allow_idle_dies=args.allow_idle_dies,
            privileged=args.privileged,
            world_size=args.world_size,
            allow_alarm_dies=args.allow_alarm_dies,
        )
        print(json.dumps(out, indent=2))
        return 0
    if args.subcmd == "exec":
        return exec_in_container(
            args.host,
            args.run_id,
            args.cmd,
            sfs_timeout_s=args.sfs_timeout,
        )
    if args.subcmd == "down":
        out = down_container(args.host, args.run_id)
        print(json.dumps(out, indent=2))
        return 0
    if args.subcmd == "sync-source":
        paths = [x.strip() for x in args.paths.split(",") if x.strip()]
        out = sync_source(args.host, args.run_id, paths=paths)
        print(json.dumps(out, indent=2))
        return 0
    if args.subcmd == "dspine-check":
        out = run_dspine_check(
            host=args.host,
            run_id=args.run_id,
            force=args.force,
            stamp_path=args.stamp_path,
        )
        print(json.dumps(out, indent=2))
        return int(out["returncode"])
    if args.subcmd == "build-shmem":
        return build_shmem_in_container(
            args.host,
            args.run_id,
            out_dir=args.out_dir,
        )
    if args.subcmd == "run":
        out = run_ephemeral(
            host=args.host,
            inner_cmd=args.cmd,
            run_id=args.run_id,
            die_slice=args.die_slice,
            port_base=args.port_base,
            image=args.image,
            kbenv_bin=args.kbenv,
            keep_container=args.keep_container,
            pull_from=args.pull_from,
            pull_to=args.pull_to,
            use_prestage=not args.no_prestage,
            allow_idle_dies=args.allow_idle_dies,
            privileged=args.privileged,
            world_size=args.world_size,
            allow_alarm_dies=args.allow_alarm_dies,
        )
        print(json.dumps(out, indent=2))
        return int(out["returncode"])
    if args.subcmd == "fanout":
        excl = [x.strip() for x in args.exclude.split(",") if x.strip()]
        out = fanout_fleet(
            args.cmd,
            max_slots=args.max_slots,
            dual_slot=not args.no_dual_slot,
            exclude_hosts=excl,
            image=args.image,
            kbenv_bin=args.kbenv,
            world_size=args.world_size,
        )
        print(json.dumps(out, indent=2))
        return 0 if all(item["returncode"] == 0 for item in out) else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
