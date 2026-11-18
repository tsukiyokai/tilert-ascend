import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
import ipaddress
import json
import math
from pathlib import Path
import posixpath
import re
import secrets
import socket
import subprocess
import tokenize
from typing import Any

from .contracts import (
    ALLOWED_A3_CHIPS,
    ALLOWED_A3_GENERATION,
    ALLOWED_KBENV_STATES,
    DEFAULT_KBENV_BIN,
    FORBIDDEN_HOSTS,
    FORBIDDEN_RUN_IDS,
    HygieneError,
    INACTIVE_PROFILER_VALUES,
    IsolationError,
    MIN_KBENV_RAM_PCT,
    PRIMARY_A3_CHIP,
    PROFILER_ENV_VARS,
    REQUIRED_ISOLATION_VARS,
    SHARED_SHM_BLOCKLIST,
    WORLD_SIZE,
)

ROOT_ENTRIES: frozenset[str] = frozenset(
    {
        ".flake8",
        ".github",
        ".gitignore",
        ".pre-commit-config.yaml",
        "Dockerfile",
        "LICENSE",
        "README.md",
        "assets",
        "docs",
        "pyproject.toml",
        "requirements-dev.txt",
        "requirements.txt",
        "scripts",
        "tests",
        "tilert",
    }
)
ALLOWED_LOCAL_ROOT_ENTRIES: frozenset[str] = ROOT_ENTRIES | frozenset({".git", ".venv"})
GENERATED_PARTS: frozenset[str] = frozenset(
    {
        ".artifacts",
        "reports",
        "evidence",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "_build",
    }
)
GENERATED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".o",
        ".a",
        ".elf",
        ".pt",
        ".pyc",
        ".log",
        ".npy",
        ".npz",
        ".safetensors",
        ".tar",
        ".gz",
        ".zip",
        ".whl",
    }
)
KBENV_ROW_RE = re.compile(
    r"^\s*(?:>\s*)?(\d+\.\d+\.\d+\.\d+)\s+(A\d+)\s+(\S+)\s+"
    r"\S+\s+(?:\d+%|--)\s+(?:(\S+)\s+)?(?:(\d+)/(\d+)|-)\s+(\d+%|--)\s+(\S+)\s+(\S+)\s*$",
    re.ASCII,
)
ALLOWED_DIE_GLYPHS: frozenset[str] = frozenset({"·", "+", "=", "#", "×"})
HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdefABCDEF")
CPP_RAW_PREFIXES: frozenset[str] = frozenset({"R", "u8R", "uR", "UR", "LR"})
ENDPOINT_RE = re.compile(r"^(?:tcp://)?(\d+\.\d+\.\d+\.\d+):(\d+)$")
RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")


@dataclass(frozen=True)
class KbenvHost:
    ip: str
    generation: str
    chip: str
    free_dies: int
    total_dies: int
    ram_pct: int | None
    state: str
    linger: str
    free_die_indices: tuple[int, ...] = ()
    note: str = ""
    die_map: str = ""


def parse_strict_ipv4(raw_host: str) -> str:
    candidate = raw_host.strip()
    if not candidate:
        raise IsolationError(f"Empty host address: {raw_host!r}")
    try:
        addr = ipaddress.IPv4Address(candidate)
    except ipaddress.AddressValueError as exc:
        raise IsolationError(f"Non-strict IPv4 host rejected: {raw_host!r}") from exc
    canonical = str(addr)
    if canonical != candidate:
        raise IsolationError(f"Non-canonical IPv4 host rejected: {raw_host!r}")
    return canonical


def assert_host_allowed(ip: str) -> str:
    canonical = parse_strict_ipv4(ip)
    if canonical in FORBIDDEN_HOSTS:
        raise IsolationError(f"Forbidden host targeted: {canonical}")
    return canonical


def validate_run_id(run_id: str) -> str:
    clean = run_id.strip()
    if (
        not clean
        or clean != run_id
        or clean in FORBIDDEN_RUN_IDS
        or ".." in clean
        or RUN_ID_RE.match(clean) is None
    ):
        raise IsolationError(f"Invalid run_id for isolation: {run_id!r}")
    return clean


def parse_kbenv_output(stdout: str) -> list[KbenvHost]:
    hosts: list[KbenvHost] = []
    for raw_line in stdout.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("    ") and hosts:
            prev = hosts[-1]
            note = f"{prev.note} {line.strip()}".strip()
            hosts[-1] = KbenvHost(
                ip=prev.ip,
                generation=prev.generation,
                chip=prev.chip,
                free_dies=prev.free_dies,
                total_dies=prev.total_dies,
                ram_pct=prev.ram_pct,
                state=prev.state,
                linger=prev.linger,
                free_die_indices=prev.free_die_indices,
                note=note,
                die_map=prev.die_map,
            )
            continue
        match = KBENV_ROW_RE.match(line)
        if match is None:
            continue
        ip_raw, gen, chip, die_map, free_s, total_s, ram_s, state, linger = match.groups()
        try:
            ip = parse_strict_ipv4(ip_raw)
        except IsolationError:
            continue
        free_dies = int(free_s) if free_s is not None else 0
        total_dies = int(total_s) if total_s is not None else 0
        ram_pct = int(ram_s[:-1]) if ram_s.endswith("%") else None
        die_map_str = die_map or ""
        if die_map_str and set(die_map_str) <= ALLOWED_DIE_GLYPHS:
            free_indices = tuple(
                idx for idx, glyph in enumerate(die_map_str) if glyph == "·"
            )
        else:
            free_indices = ()
        hosts.append(
            KbenvHost(
                ip=ip,
                generation=gen,
                chip=chip,
                free_dies=free_dies,
                total_dies=total_dies,
                ram_pct=ram_pct,
                state=state,
                linger=linger,
                free_die_indices=free_indices,
                die_map=die_map_str,
            )
        )
    return hosts


def select_free_die_slice(
    free_die_indices: Sequence[int],
    count: int = WORLD_SIZE,
) -> tuple[int, ...]:
    for idx in free_die_indices:
        if isinstance(idx, bool) or not isinstance(idx, int) or not (0 <= idx < 16):
            raise IsolationError(f"Die index out of Ascend A3 range 0..15: {idx!r}")
    unique = sorted(set(free_die_indices))
    if len(unique) < count:
        raise IsolationError(
            f"Only {len(unique)} free dies available, require {count}"
        )
    if count == 8:
        low_slice = tuple(range(8))
        if all(idx in unique for idx in low_slice):
            return low_slice
        high_slice = tuple(range(8, 16))
        if all(idx in unique for idx in high_slice):
            return high_slice
    return tuple(unique[:count])


def select_a3_host(hosts: Sequence[KbenvHost], min_free_dies: int = WORLD_SIZE) -> KbenvHost:
    if (
        isinstance(min_free_dies, bool)
        or not isinstance(min_free_dies, int)
        or not (1 <= min_free_dies <= 16)
    ):
        raise IsolationError(f"min_free_dies must be int in 1..16, got {min_free_dies!r}")
    candidates: list[KbenvHost] = []
    for host in hosts:
        try:
            assert_host_allowed(host.ip)
        except IsolationError:
            continue
        if host.generation != ALLOWED_A3_GENERATION:
            continue
        if host.chip not in ALLOWED_A3_CHIPS:
            continue
        if host.state not in ALLOWED_KBENV_STATES:
            continue
        if host.ram_pct is None or host.ram_pct < MIN_KBENV_RAM_PCT:
            continue
        if host.total_dies != 16:
            continue
        if (
            not host.die_map
            or len(host.die_map) != 16
            or not (set(host.die_map) <= ALLOWED_DIE_GLYPHS)
        ):
            continue
        if host.free_dies < min_free_dies:
            continue
        if len(host.free_die_indices) != host.free_dies:
            continue
        if any(not (0 <= d < 16) for d in host.free_die_indices):
            continue
        candidates.append(host)
    if not candidates:
        raise IsolationError(
            f"No eligible {ALLOWED_A3_GENERATION} host with >= {min_free_dies} free dies"
        )

    def rank_key(host: KbenvHost) -> tuple[int, int, int, int]:
        state_score = 0 if host.state == "空闲" else 1
        chip_score = 0 if host.chip == PRIMARY_A3_CHIP else 1
        linger_score = 0 if host.linger == "长期" else 1
        return (state_score, chip_score, -host.free_dies, linger_score)

    return min(candidates, key=rank_key)


def run_kbenv_precheck(
    kbenv_bin: str = DEFAULT_KBENV_BIN,
    extra_args: list[str] | None = None,
    min_free_dies: int = WORLD_SIZE,
    timeout: float = 30.0,
) -> tuple[list[KbenvHost], KbenvHost]:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0.0
    ):
        raise IsolationError(
            f"kbenv timeout must be finite positive numeric, got {timeout!r}"
        )
    cmd = [kbenv_bin]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append("--once")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout),
        )
    except subprocess.TimeoutExpired as exc:
        raise IsolationError(f"kbenv precheck timed out after {float(timeout):.1f}s") from exc
    if proc.returncode != 0:
        raise IsolationError(
            f"kbenv precheck exited {proc.returncode}: {proc.stderr.strip()}"
        )
    hosts = parse_kbenv_output(proc.stdout)
    selected = select_a3_host(hosts, min_free_dies=min_free_dies)
    assert_host_allowed(selected.ip)
    return hosts, selected


def _can_bind_port_block(port_base: int) -> bool:
    ports = [port_base, port_base + 1, port_base + 2, *range(port_base + 100, port_base + 251)]
    sockets: list[socket.socket] = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


def allocate_free_port_base(reserved_bases: Sequence[int] = ()) -> int:
    blocks = list(range(150))
    secrets.SystemRandom().shuffle(blocks)
    for block_idx in blocks:
        candidate = 20000 + block_idx * 300
        if any(abs(candidate - used) < 300 for used in reserved_bases):
            continue
        if _can_bind_port_block(candidate):
            return candidate
    raise IsolationError("Unable to reserve an uncontended ephemeral TCP port block")


def expected_isolation_metadata(env: Mapping[str, str]) -> dict[str, Any]:
    return {
        "shm_dir": env["TILERT_RUN_DIR"],
        "master_port": int(env["MASTER_PORT"]),
        "whole_layer_endpoint": env["TILERT_WHOLE_LAYER_ENDPOINT"],
        "onesided_endpoint": env["TILERT_ONESIDED_ENDPOINT"],
        "hccl_port_range": env["HCCL_NPU_SOCKET_PORT_RANGE"],
        "atb_share_memory_suffix": env["ATB_SHARE_MEMORY_NAME_SUFFIX"],
    }


def allocate_isolation_env(
    run_id: str,
    port_base: int | None = 29900,
    shm_root: str = "/dev/shm",
    target_host: str | None = None,
    visible_devices: Sequence[int] | None = None,
) -> dict[str, str]:
    clean_id = validate_run_id(run_id)
    resolved_host = assert_host_allowed(target_host) if target_host is not None else None
    resolved_port = allocate_free_port_base() if port_base is None else port_base
    if resolved_port < 1024 or resolved_port + 250 > 65535:
        raise IsolationError(f"Invalid port_base for isolation: {resolved_port}")
    norm_root = posixpath.normpath(shm_root.strip())
    if norm_root != "/dev/shm":
        raise IsolationError(f"shm_root must be '/dev/shm', got {shm_root!r}")
    shm_dir = f"{norm_root}/{clean_id}"
    verify_shm_isolation(shm_dir, clean_id)
    if visible_devices is None:
        dies = tuple(range(WORLD_SIZE))
    else:
        dies = tuple(visible_devices)
        if (
            len(dies) != WORLD_SIZE
            or len(set(dies)) != WORLD_SIZE
            or any(
                isinstance(d, bool) or not isinstance(d, int) or not (0 <= d < 16)
                for d in dies
            )
        ):
            raise IsolationError(
                f"visible_devices must contain {WORLD_SIZE} unique ints in 0..15"
            )
    hccl_lo = resolved_port + 100
    hccl_hi = resolved_port + 250
    env = {
        "TILERT_E2E_RUN_ID": clean_id,
        "TILERT_RUN_DIR": shm_dir,
        "TILERT_JOB_DIR": f"{shm_dir}/control",
        "TILERT_BENCH_ROOT": f"{shm_dir}/bench",
        "VLLM_CACHE_ROOT": f"{shm_dir}/vllm_cache",
        "ATB_SHARE_MEMORY_NAME_SUFFIX": f"glm52_{clean_id}",
        "MASTER_PORT": str(resolved_port),
        "TILERT_WHOLE_LAYER_ENDPOINT": f"tcp://127.0.0.1:{resolved_port + 1}",
        "TILERT_ONESIDED_ENDPOINT": f"tcp://127.0.0.1:{resolved_port + 2}",
        "HCCL_NPU_SOCKET_PORT_RANGE": f"{hccl_lo}-{hccl_hi}",
        "HCCL_DETERMINISTIC": "true",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(idx) for idx in dies),
    }
    if resolved_host is not None:
        env["TILERT_TARGET_HOST"] = resolved_host
    return env


def _parse_ascii_port(value: str, label: str) -> int:
    if value != value.strip():
        raise IsolationError(f"Unstripped whitespace in {label}: {value!r}")
    if not value or not value.isascii() or not value.isdigit():
        raise IsolationError(f"Invalid ASCII port in {label}: {value!r}")
    port = int(value)
    if not (1024 <= port <= 65535):
        raise IsolationError(f"Out-of-range port in {label}: {port}")
    return port


def _parse_endpoint_port(value: str, label: str) -> int:
    if value != value.strip():
        raise IsolationError(f"Unstripped whitespace in {label}: {value!r}")
    match = ENDPOINT_RE.match(value)
    if match is None:
        raise IsolationError(f"Invalid endpoint format in {label}: {value!r}")
    host_part, port_part = match.groups()
    canonical = parse_strict_ipv4(host_part)
    if canonical != "127.0.0.1":
        raise IsolationError(
            f"Endpoint {label} must bind loopback 127.0.0.1, got {canonical!r}"
        )
    return _parse_ascii_port(port_part, label)


def _parse_port_range(value: str) -> tuple[int, int]:
    if value != value.strip():
        raise IsolationError(f"Unstripped whitespace in HCCL_NPU_SOCKET_PORT_RANGE: {value!r}")
    match = RANGE_RE.match(value)
    if match is None:
        raise IsolationError(f"Invalid HCCL_NPU_SOCKET_PORT_RANGE: {value!r}")
    lo_s, hi_s = match.groups()
    if not (lo_s.isascii() and hi_s.isascii()):
        raise IsolationError(f"Non-ASCII digits in HCCL_NPU_SOCKET_PORT_RANGE: {value!r}")
    lo = int(lo_s)
    hi = int(hi_s)
    if lo < 1024 or hi > 65535 or lo >= hi:
        raise IsolationError(f"Invalid port range bounds: {lo}-{hi}")
    return lo, hi


def _parse_visible_devices(value: str, label: str) -> frozenset[int]:
    if value != value.strip() or not value:
        raise IsolationError(f"Invalid ASCEND_RT_VISIBLE_DEVICES in {label}: {value!r}")
    parts = value.split(",")
    if len(parts) != WORLD_SIZE:
        raise IsolationError(
            f"ASCEND_RT_VISIBLE_DEVICES in {label} must list {WORLD_SIZE} dies, got {value!r}"
        )
    dies: set[int] = set()
    for part in parts:
        if part != part.strip() or not part or not part.isascii() or not part.isdigit():
            raise IsolationError(
                f"Invalid die token {part!r} in {label} ASCEND_RT_VISIBLE_DEVICES"
            )
        die_idx = int(part)
        if str(die_idx) != part or not (0 <= die_idx < 16):
            raise IsolationError(
                f"Die index out of Ascend A3 range 0..15 in {label}: {part!r}"
            )
        dies.add(die_idx)
    if len(dies) != WORLD_SIZE:
        raise IsolationError(
            f"Duplicate die indices in {label} ASCEND_RT_VISIBLE_DEVICES: {value!r}"
        )
    return frozenset(dies)


def _used_ports_from_env(env: Mapping[str, str], label: str = "env") -> set[int]:
    required_port_keys = (
        "MASTER_PORT",
        "TILERT_WHOLE_LAYER_ENDPOINT",
        "TILERT_ONESIDED_ENDPOINT",
        "HCCL_NPU_SOCKET_PORT_RANGE",
    )
    for key in required_port_keys:
        if not env.get(key, "").strip():
            raise IsolationError(f"Missing required port variable {key} in {label}")
    ports = {
        _parse_ascii_port(env["MASTER_PORT"], "MASTER_PORT"),
        _parse_endpoint_port(env["TILERT_WHOLE_LAYER_ENDPOINT"], "TILERT_WHOLE_LAYER_ENDPOINT"),
        _parse_endpoint_port(env["TILERT_ONESIDED_ENDPOINT"], "TILERT_ONESIDED_ENDPOINT"),
    }
    if len(ports) != 3:
        raise IsolationError("Endpoint ports collide within single run isolation env")
    lo, hi = _parse_port_range(env["HCCL_NPU_SOCKET_PORT_RANGE"])
    range_set = set(range(lo, hi + 1))
    if ports & range_set:
        raise IsolationError("Single-run ports overlap HCCL_NPU_SOCKET_PORT_RANGE")
    return ports | range_set


def verify_shm_isolation(shm_dir: str, run_id: str) -> None:
    clean_id = validate_run_id(run_id)
    if shm_dir != shm_dir.strip():
        raise IsolationError(f"shm_dir {shm_dir!r} has unstripped whitespace")
    if not shm_dir or ".." in Path(shm_dir).parts or ".." in shm_dir.split("/"):
        raise IsolationError(f"shm_dir {shm_dir!r} contains disallowed traversal or is empty")
    norm = posixpath.normpath(shm_dir)
    expected_root = f"/dev/shm/{clean_id}"
    if norm in SHARED_SHM_BLOCKLIST or not (
        norm == expected_root or norm.startswith(f"{expected_root}/")
    ):
        raise IsolationError(f"shm_dir {shm_dir!r} is not isolated under {expected_root}")


def _verify_single_isolation_env(
    env: Mapping[str, str],
    target_host: str | None = None,
    label: str = "env",
) -> tuple[str, str, str, str, set[int], str | None, frozenset[int]]:
    expected_host = assert_host_allowed(target_host) if target_host is not None else None
    raw_env_host = env.get("TILERT_TARGET_HOST", "")
    if raw_env_host != raw_env_host.strip():
        raise IsolationError(f"TILERT_TARGET_HOST has unstripped whitespace in {label}")
    env_host: str | None = None
    if raw_env_host:
        env_host = assert_host_allowed(raw_env_host)
        if expected_host is not None and env_host != expected_host:
            raise IsolationError(
                f"{label} TILERT_TARGET_HOST {env_host!r} != target_host {expected_host!r}"
            )
    elif expected_host is not None:
        raise IsolationError(f"Missing TILERT_TARGET_HOST in {label}")

    for key in REQUIRED_ISOLATION_VARS:
        if not env.get(key, "").strip():
            raise IsolationError(f"Missing required isolation env var {key} in {label}")
    for key in PROFILER_ENV_VARS:
        if env.get(key, "").strip() not in INACTIVE_PROFILER_VALUES:
            raise IsolationError(f"Profiler or sync debug flag active in {label}: {key}")
    if env.get("HCCL_DETERMINISTIC") != "true":
        raise IsolationError(f"HCCL_DETERMINISTIC must be 'true' in {label}")
    if env.get("HCCL_OP_EXPANSION_MODE") != "AIV":
        raise IsolationError(f"HCCL_OP_EXPANSION_MODE must be 'AIV' in {label}")
    run_id = validate_run_id(env["TILERT_E2E_RUN_ID"])
    run_dir = env["TILERT_RUN_DIR"]
    if run_dir != run_dir.strip():
        raise IsolationError(f"TILERT_RUN_DIR has unstripped whitespace in {label}")
    verify_shm_isolation(run_dir, run_id)
    expected_prefix = f"/dev/shm/{run_id}/"
    for path_key in ("TILERT_JOB_DIR", "TILERT_BENCH_ROOT"):
        val = env[path_key].strip()
        if val != env[path_key]:
            raise IsolationError(f"{path_key} has unstripped whitespace in {label}")
        verify_shm_isolation(val, run_id)
        if not posixpath.normpath(val).startswith(expected_prefix):
            raise IsolationError(
                f"{path_key}={val!r} in {label} must reside under {expected_prefix!r}"
            )
    if "VLLM_CACHE_ROOT" in env and env["VLLM_CACHE_ROOT"] != "":
        vllm_cache = env["VLLM_CACHE_ROOT"]
        if vllm_cache != vllm_cache.strip():
            raise IsolationError(f"VLLM_CACHE_ROOT has unstripped whitespace in {label}")
        verify_shm_isolation(vllm_cache, run_id)
        if not posixpath.normpath(vllm_cache).startswith(expected_prefix):
            raise IsolationError(
                f"VLLM_CACHE_ROOT={vllm_cache!r} in {label} must reside under {expected_prefix!r}"
            )
    raw_atb = env["ATB_SHARE_MEMORY_NAME_SUFFIX"]
    atb_suffix = raw_atb.strip()
    if (
        raw_atb != atb_suffix
        or not atb_suffix
        or run_id not in atb_suffix
        or atb_suffix == "shared_global"
    ):
        raise IsolationError(
            f"ATB_SHARE_MEMORY_NAME_SUFFIX in {label} must be trimmed and include run_id"
        )
    visible_dies = _parse_visible_devices(env["ASCEND_RT_VISIBLE_DEVICES"], label)
    ports = _used_ports_from_env(env, label=label)
    return (
        run_id,
        atb_suffix,
        posixpath.normpath(run_dir),
        posixpath.normpath(env["TILERT_JOB_DIR"]),
        ports,
        env_host,
        visible_dies,
    )


def verify_isolation_env(
    env: Mapping[str, str],
    peer_envs: Sequence[Mapping[str, str]] = (),
    target_host: str | None = None,
) -> None:
    run_id, atb_suffix, run_dir, job_dir, own_ports, own_host, own_dies = (
        _verify_single_isolation_env(env, target_host=target_host, label="env")
    )
    seen_ids = {run_id}
    seen_atb = {atb_suffix}
    seen_run_dirs = {run_dir}
    seen_job_dirs = {job_dir}
    used_ports = set(own_ports)
    used_dies_by_host: dict[str, set[int]] = {}
    unspecified_host_dies: set[int] = set()
    if own_host is None:
        unspecified_host_dies |= own_dies
    else:
        used_dies_by_host[own_host] = set(own_dies)
    for idx, peer in enumerate(peer_envs):
        peer_label = f"peer_envs[{idx}]"
        p_id, p_atb, p_run_dir, p_job_dir, p_ports, p_host, p_dies = (
            _verify_single_isolation_env(peer, label=peer_label)
        )
        if p_id in seen_ids:
            raise IsolationError(f"Duplicate TILERT_E2E_RUN_ID across concurrent runs: {p_id}")
        if p_atb in seen_atb:
            raise IsolationError("Duplicate ATB_SHARE_MEMORY_NAME_SUFFIX across runs")
        if p_run_dir in seen_run_dirs:
            raise IsolationError("Duplicate TILERT_RUN_DIR across concurrent runs")
        if p_job_dir in seen_job_dirs:
            raise IsolationError("Duplicate TILERT_JOB_DIR across concurrent runs")
        overlap = used_ports & p_ports
        if overlap:
            raise IsolationError(f"Overlapping ports across concurrent runs: {sorted(overlap)}")
        if p_dies & unspecified_host_dies:
            die_overlap = sorted(p_dies & unspecified_host_dies)
            raise IsolationError(
                f"Overlapping NPU dies {die_overlap} with shared/unspecified target_host"
            )
        if p_host is None:
            for host_key, host_dies in used_dies_by_host.items():
                if p_dies & host_dies:
                    raise IsolationError(
                        f"Overlapping NPU dies {sorted(p_dies & host_dies)} "
                        f"against host {host_key} from unspecified-host {peer_label}"
                    )
            unspecified_host_dies |= p_dies
        else:
            host_dies = used_dies_by_host.setdefault(p_host, set())
            if p_dies & host_dies:
                raise IsolationError(
                    f"Overlapping NPU dies {sorted(p_dies & host_dies)} on host {p_host}"
                )
            host_dies |= p_dies
        seen_ids.add(p_id)
        seen_atb.add(p_atb)
        seen_run_dirs.add(p_run_dir)
        seen_job_dirs.add(p_job_dir)
        used_ports |= p_ports


def _is_standalone_string_expr(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (str, bytes)):
        return True
    if isinstance(expr, ast.JoinedStr):
        return True
    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        return any(_is_standalone_string_expr(elt) for elt in expr.elts)
    if isinstance(expr, ast.Dict):
        return any(
            (key is not None and _is_standalone_string_expr(key))
            or _is_standalone_string_expr(val)
            for key, val in zip(expr.keys, expr.values)
        )
    if isinstance(expr, ast.Subscript):
        return (
            _is_standalone_string_expr(expr.value)
            or _is_standalone_string_expr(expr.slice)
        )
    if isinstance(expr, ast.Slice):
        return any(
            bound is not None and _is_standalone_string_expr(bound)
            for bound in (expr.lower, expr.upper, expr.step)
        )
    if isinstance(expr, ast.IfExp):
        return (
            _is_standalone_string_expr(expr.test)
            or _is_standalone_string_expr(expr.body)
            or _is_standalone_string_expr(expr.orelse)
        )
    if isinstance(expr, ast.BoolOp):
        return any(_is_standalone_string_expr(val) for val in expr.values)
    if isinstance(expr, ast.UnaryOp):
        return _is_standalone_string_expr(expr.operand)
    if isinstance(expr, ast.BinOp):
        return _is_standalone_string_expr(expr.left) or _is_standalone_string_expr(expr.right)
    if isinstance(expr, ast.Compare):
        return _is_standalone_string_expr(expr.left) or any(
            _is_standalone_string_expr(comp) for comp in expr.comparators
        )
    if isinstance(expr, (ast.Starred, ast.NamedExpr)):
        return _is_standalone_string_expr(expr.value)
    return False


def verify_python_source_hygiene(
    paths: Sequence[Path],
    max_line_length: int = 100,
) -> None:
    for path in paths:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise HygieneError(f"UTF-8 BOM forbidden in {path}")
        if b"\r" in raw:
            raise HygieneError(f"CRLF / bare CR line ending detected in {path}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HygieneError(f"Invalid UTF-8 in {path}: {exc}") from exc
        if text and not text.endswith("\n"):
            raise HygieneError(f"Missing trailing newline in {path}")
        if text.endswith("\n\n"):
            raise HygieneError(f"Extra trailing blank line in {path}")
        for idx, line in enumerate(text.splitlines(), start=1):
            if line != line.rstrip(" \t"):
                raise HygieneError(f"Trailing whitespace in {path}:{idx}")
            if len(line) > max_line_length:
                raise HygieneError(
                    f"Line exceeds {max_line_length} chars in {path}:{idx} ({len(line)})"
                )
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT:
                    raise HygieneError(f"Python # comment forbidden in {path}:{token.start[0]}")
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, tokenize.TokenError) as exc:
            raise HygieneError(f"Invalid Python syntax in {path}: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and _is_standalone_string_expr(node.value):
                raise HygieneError(
                    f"Standalone string literal / docstring forbidden in {path}:{node.lineno}"
                )


def _skip_cpp_splices(text: str, idx: int) -> int:
    n = len(text)
    while idx < n and text[idx] == "\\":
        if idx + 1 < n and text[idx + 1] == "\n":
            idx += 2
        elif idx + 2 < n and text[idx + 1] == "\r" and text[idx + 2] == "\n":
            idx += 3
        else:
            break
    return idx


def _consume_cpp_raw_string_from_quote(text: str, quote_pos: int, path: Path) -> int:
    paren = text.find("(", quote_pos + 1)
    if paren == -1 or paren - (quote_pos + 1) > 16:
        raise HygieneError(f"Malformed C++ raw string literal in {path}")
    delim = text[quote_pos + 1 : paren]
    if any(ch in delim for ch in (" ", "\t", "\n", "\r", "\\", ")")):
        raise HygieneError(f"Invalid C++ raw string delimiter in {path}")
    end_marker = f"){delim}\""
    end_idx = text.find(end_marker, paren + 1)
    if end_idx == -1:
        raise HygieneError(f"Unterminated C++ raw string literal in {path}")
    return end_idx + len(end_marker)


def verify_cpp_no_comments(paths: Sequence[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        i = 0
        n = len(text)
        in_string = False
        in_char = False
        while i < n:
            pos = _skip_cpp_splices(text, i)
            if pos >= n:
                break
            ch = text[pos]
            if in_string:
                if ch == "\\":
                    esc_pos = _skip_cpp_splices(text, pos + 1)
                    if esc_pos >= n:
                        i = esc_pos
                        break
                    i = esc_pos + 1
                    continue
                if ch in {"\n", "\r"}:
                    raise HygieneError(f"Unterminated C++ string literal in {path}")
                if ch == '"':
                    in_string = False
                i = pos + 1
                continue
            if in_char:
                if ch == "\\":
                    esc_pos = _skip_cpp_splices(text, pos + 1)
                    if esc_pos >= n:
                        i = esc_pos
                        break
                    i = esc_pos + 1
                    continue
                if ch in {"\n", "\r"}:
                    raise HygieneError(f"Unterminated C++ char literal in {path}")
                if ch == "'":
                    in_char = False
                i = pos + 1
                continue
            if ch == '"':
                in_string = True
                i = pos + 1
                continue
            if ch == "'":
                in_char = True
                i = pos + 1
                continue
            if ch == "/":
                nxt_pos = _skip_cpp_splices(text, pos + 1)
                nxt = text[nxt_pos] if nxt_pos < n else ""
                if nxt == "/":
                    raise HygieneError(f"C++ // comment forbidden in {path}")
                if nxt == "*":
                    raise HygieneError(f"C++ /* comment forbidden in {path}")
                i = pos + 1
                continue
            nxt_pos = _skip_cpp_splices(text, pos + 1)
            nxt = text[nxt_pos] if nxt_pos < n else ""
            if ch.isdigit() or (ch == "." and nxt.isdigit()):
                prev_ch = ch
                cur_pos = nxt_pos
                while cur_pos < n:
                    cur_pos = _skip_cpp_splices(text, cur_pos)
                    if cur_pos >= n:
                        break
                    cur = text[cur_pos]
                    if cur.isalnum() or cur in {"_", "."}:
                        if cur in {"e", "E", "p", "P"}:
                            sign_pos = _skip_cpp_splices(text, cur_pos + 1)
                            if sign_pos < n and text[sign_pos] in {"+", "-"}:
                                prev_ch = text[sign_pos]
                                cur_pos = sign_pos + 1
                                continue
                        prev_ch = cur
                        cur_pos += 1
                    elif cur == "'" and prev_ch in HEX_DIGITS:
                        after_sep = _skip_cpp_splices(text, cur_pos + 1)
                        if after_sep < n and text[after_sep] in HEX_DIGITS:
                            prev_ch = text[after_sep]
                            cur_pos = after_sep + 1
                        else:
                            break
                    else:
                        break
                i = cur_pos
                continue
            if ch.isalpha() or ch == "_":
                ident_chars = [ch]
                cur_pos = pos + 1
                while cur_pos < n:
                    cur_pos = _skip_cpp_splices(text, cur_pos)
                    if cur_pos >= n:
                        break
                    cur = text[cur_pos]
                    if cur.isalnum() or cur == "_":
                        ident_chars.append(cur)
                        cur_pos += 1
                    else:
                        break
                ident = "".join(ident_chars)
                if cur_pos < n and text[cur_pos] == '"' and ident in CPP_RAW_PREFIXES:
                    i = _consume_cpp_raw_string_from_quote(text, cur_pos, path)
                else:
                    i = cur_pos
                continue
            i = pos + 1
        if in_string or in_char:
            raise HygieneError(f"Unterminated C++ literal at EOF in {path}")


def is_generated_path(rel_path: Path) -> bool:
    for part in rel_path.parts[:-1]:
        if part in GENERATED_PARTS or "_build" in part or part.startswith("build-"):
            return True
    return (
        rel_path.name.startswith("build-")
        or ".so" in rel_path.suffixes
        or rel_path.suffix in GENERATED_SUFFIXES
        or rel_path.name in {"provenance.json", "fusion_result.json"}
    )


def verify_root_hygiene(repo_root: Path) -> dict[str, Any]:
    unexpected_root = sorted(
        entry.name
        for entry in repo_root.iterdir()
        if entry.name not in ALLOWED_LOCAL_ROOT_ENTRIES
    )
    if unexpected_root:
        raise HygieneError(f"Unexpected repo root entries: {unexpected_root}")
    project_script = repo_root / "scripts" / "project.py"
    if not project_script.is_file():
        raise HygieneError(f"Missing scripts/project.py at {project_script}")
    proc = subprocess.run(
        ["python3", str(project_script), "check"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise HygieneError(
            f"scripts/project.py check failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    result = json.loads(proc.stdout)
    if not result.get("pass"):
        raise HygieneError(f"scripts/project.py check reported fail: {result}")
    for scan_rel in (Path("tests") / "glm52" / "e2e_tp8_ep8", Path("tilert")):
        scan_root = repo_root / scan_rel
        if not scan_root.is_dir():
            continue
        for item in scan_root.rglob("*"):
            rel = item.relative_to(repo_root)
            if "__pycache__" in rel.parts or rel.parts[:2] == ("tests", ".artifacts"):
                continue
            if item.is_dir() and item.name.startswith("build-"):
                raise HygieneError(f"Generated build dir found in {scan_rel}: {rel}")
            if item.is_file() and is_generated_path(rel):
                raise HygieneError(f"Generated artifact found inside {scan_rel}: {rel}")
    return result
