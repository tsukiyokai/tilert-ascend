from collections.abc import Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent / "megakernel"
REPO = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO / "tests/.artifacts/builds/megakernel_tp8_ep8"
CPP_SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx", ".cu"})
UNSET_L78_MACROS = frozenset(
    {
        "L78_PLAIN_RMSNORM",
        "L78_EXACT_INDEXER",
        "K11085_L78_CAUSAL",
        "K11089_L78_SFA_TOPK",
        "K11050_L78_SHARED_SFA",
    }
)
L78_MACRO_RE = re.compile(r"^L78_[A-Z0-9_]+$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFINED_EXPR_RE = re.compile(
    r"^(!)?\s*defined\s*(?:\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)|\s+([A-Za-z_][A-Za-z0-9_]*))$"
)

ASC_SUBDIRS = (
    "asc",
    "asc/include",
    "asc/include/adv_api",
    "asc/include/basic_api",
    "asc/include/c_api",
    "asc/include/simt_api",
    "asc/include/utils",
    "asc/include/basic_api/reg_compute",
    "asc/impl/adv_api",
    "asc/impl/basic_api",
    "asc/impl/c_api",
    "asc/impl/simt_api",
    "asc/impl/utils",
    "asc/impl/basic_api/reg_compute",
    "tikcpp/tikcfw",
    "tikcpp/tikcfw/lib",
    "tikcpp/tikcfw/impl",
    "tikcpp/tikcfw/interface",
)

BASE_FLAGS = (
    "-DSFA_MM2_PAIR_LOAD=1",
    "-DQUERY_GROUPS_NARROW=1",
    "-DBF16_QUERY12_GROUPS=1",
    "-DGU_WIDE=1",
    "-DATTENTION_HEADS=8",
    "-DFFN_OP_NS=tilert_ffn_original_shared_gu_static",
    "-DROUTED_ROW_BALANCED=1",
    "-DROUTED_SHARED_PIPELINE=1",
    "-DROUTED_INPUT_FRONT=1",
    "-DSHARED_W8_DYNAMIC_PERFORMANCE_QUANT=1",
    "-DROUTED_ACTIVE_LIST=1",
    "-DSKIP_SINGLE_KERNELS=1",
)

R4_FLAGS = (
    "-DK11040_F0_NZ=1",
    "-DK11040_TAIL_NZ=1",
    "-DK11040_GU_FAST128=1",
    "-DK11080_DIRECT_W4A8=1",
    "-DK11083_RED2_OPT=1",
    "-DK11088_GATE_CONTIG=1",
    "-DK11090_FAST_SFA_VEC=1",
    "-DK11050_AIC_PREFETCH=1",
    "-DK11100_OPP_PREFETCH=1",
    "-DK11110_SFA_PIPE=1",
    "-DK11110_DW4A8_XTASK=1",
    "-DK11070_SFA_M16_COOP=24",
    "-DK11120_SFA_GATHER_OPT=1",
    "-DK11120_SKIP_DUP_PUB=1",
    "-DK11140_DIRECT_MLA_TAIL=1",
    "-DK11150_MLA_TAIL_M32=1",
    "-DK11150_SHARED_W8_NZ=1",
    "-DK11150_SIP_RELAY_RED2=1",
    "-DK11180_QUAD4_SIP_ALL=1",
    "-DK11180_TOK_OUT_BAL=1",
    "-DK11180_NORM_OPT=1",
    "-DK112110_SIP_RED1=1",
    "-DK112140_SIP_RED1_MIN_ROWS=8",
    "-DK112120_CLEAN_DCCI=1",
    "-DK112140_PIPE_KEY_ONLY=1",
    "-DK112120_SFA_COMPACT_WS=1",
    "-DK112130_VEC_RESOLVED_TOPK=1",
    "-DK112130_SPLIT_HEADS_4_4=1",
    "-DK112140_FAST_RTK=1",
    "-DK112160_AIV_DSB_OPT=1",
    "-DK112162_FUSED_IDX_RTK=1",
    "-DK112162_MERGE_CONTIG=1",
    "-DK112168_MLA_TAIL_OVERLAP=1",
    "-DK112170_DEC_SIP_ATT=1",
    "-DK112174_FAST_MERGE_KV=1",
    "-DK112176_PHASE_DSB_OPT=1",
    "-DK112184_IDX_FRONT_OPT=1",
    "-DK112192_DIRECT_IDX_Q14=1",
    "-DK112194_PROLOG_CUBE_BAL=1",
    "-DK11240_OUTLINE_TOP=1",
    "-DK112166_SFA_QV_L1=1",
    "-DK112188_SFA_QVL1_ALL=1",
    "-DK11220_FAST_MIRROR_V4=1",
    "-DK11220_MULTI_M16=1",
    "-DK11220_XTASK_B=1",
    "-DK11220_EXACT_COST=1",
    "-DK11220_FAST_TOK_OUT=1",
    "-DK11220_FAST_ACT_P3=1",
    "-DK11220_BITMASK_OWN_P1=1",
    "-DK11220_DOWN_CONTIG_256=1",
    "-DK11220_SH_ROW_1D=1",
    "-DK11220_LOCAL_SH_RED2=1",
    "-DK11220_PF_RED2_FULL=1",
    "-DK11220_FAST_P1_V3=1",
    "-DK11220_MULTI_M32=1",
    "-DK112612_BRIDGE_FAST=1",
    "-DK112612_DOWN_PIPE2=1",
    "-DK11220_DENSE78_COLD=1",
    "-DK112510_MLA_TAIL_XCORE_V2=1",
    "-DK112512_PROLOG_48AIV_V2=1",
    "-DK112702_NO_DDR_INTRADIE=1",
    "-DK112722_NO_DDR_EXTRA=1",
    "-DK112726_BRIDGE_PRELOAD_GATE=1",
    "-DK112600_EXACT48_RED2_OPT=1",
    "-DK112514_IDX_PIPE_V2=1",
    "-DK112824_NO_R2_END_DSB=1",
    "-DK112612_TRIM_BARRIERS=1",
    "-DK112824_SPLIT_PROF=1",
    "-DK112830_FAST_POLL_RED2=1",
)


def default_build_dir() -> Path:
    env_dir = os.environ.get("TILERT_BUILD_DIR")
    return Path(env_dir) if env_dir else DEFAULT_OUT


def _cann_root() -> Path:
    for candidate in (
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/opt/ascend/cann-9.2.0",
        "/usr/local/Ascend/ascend-toolkit/latest",
    ):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return Path("/opt/ascend/cann-9.2.0")


def _site_root() -> Path:
    try:
        import torch

        return Path(torch.__file__).resolve().parents[1]
    except Exception:
        return Path("/usr/local/lib/python3.11/site-packages")


def source_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(ROOT)).encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def _extract_defined_macros(flags: Sequence[str]) -> frozenset[str]:
    macros: set[str] = set()
    for flag in flags:
        if not flag.startswith("-D"):
            continue
        body = flag[2:].split("=", 1)[0].strip()
        if IDENT_RE.match(body):
            macros.add(body)
    return frozenset(macros)


def _is_l78_unset_macro(name: str) -> bool:
    return name in UNSET_L78_MACROS or L78_MACRO_RE.match(name) is not None


def _eval_macro_defined(
    name: str,
    defined_macros: frozenset[str],
    tainted_macros: frozenset[str] = frozenset(),
) -> bool | None:
    if name in tainted_macros:
        return None
    if name in defined_macros:
        return True
    if _is_l78_unset_macro(name):
        return False
    return None


def _strip_cpp_comments(expr: str) -> str:
    out: list[str] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch == "/" and i + 1 < n and expr[i + 1] == "/":
            break
        if ch == "/" and i + 1 < n and expr[i + 1] == "*":
            end = expr.find("*/", i + 2)
            if end == -1:
                break
            out.append(" ")
            i = end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _eval_if_expr(
    expr: str,
    defined_macros: frozenset[str],
    tainted_macros: frozenset[str] = frozenset(),
) -> bool | None:
    clean = _strip_cpp_comments(expr)
    m = DEFINED_EXPR_RE.match(clean)
    if m is None:
        return None
    neg, name_paren, name_bare = m.groups()
    name = name_paren or name_bare
    val = _eval_macro_defined(name, defined_macros, tainted_macros)
    if val is None:
        return None
    return (not val) if neg else val


def _is_raw_string_start(text: str, pos: int) -> bool:
    if pos + 1 >= len(text) or text[pos] != "R" or text[pos + 1] != '"':
        return False
    for prefix_len in (0, 1, 2):
        start = pos - prefix_len
        if start < 0:
            continue
        if prefix_len > 0 and text[start:pos] not in ("u", "U", "L", "u8"):
            continue
        if start == 0 or not (text[start - 1].isalnum() or text[start - 1] == "_"):
            return True
    return False


def _consume_raw_string(text: str, quote_pos: int) -> int | None:
    paren = text.find("(", quote_pos + 1)
    if paren == -1 or paren - (quote_pos + 1) > 16:
        return None
    delim = text[quote_pos + 1 : paren]
    if any(ch in delim for ch in (" ", "\t", "\n", "\r", "\\", ")")):
        return None
    end_marker = f"){delim}\""
    end_idx = text.find(end_marker, paren + 1)
    if end_idx == -1:
        return None
    return end_idx + len(end_marker)


def _split_cpp_logical_lines(text: str) -> list[tuple[str, bool, str]]:
    norm = text.replace("\r\n", "\n")
    lines: list[tuple[str, bool, str]] = []
    n = len(norm)
    i = 0
    while i < n:
        start = i
        j = i
        while j < n and norm[j] in (" ", "\t"):
            j += 1
        while j + 1 < n and norm[j] == "/" and norm[j + 1] == "*":
            end_c = norm.find("*/", j + 2)
            if end_c == -1:
                break
            j = end_c + 2
            while j < n and norm[j] in (" ", "\t"):
                j += 1
        is_dir = j < n and norm[j] == "#"
        dir_parts: list[str] = []
        in_str = False
        in_chr = False
        in_blk = False
        in_lcom = False
        if is_dir:
            k = j + 1
            while k < n:
                ch = norm[k]
                if in_blk:
                    if ch == "*" and k + 1 < n and norm[k + 1] == "/":
                        in_blk = False
                        dir_parts.append(" ")
                        k += 2
                    else:
                        k += 1
                    continue
                if ch == "\\" and k + 1 < n and norm[k + 1] == "\n":
                    k += 2
                    continue
                if ch == "\n":
                    k += 1
                    break
                if in_str:
                    dir_parts.append(ch)
                    if ch == "\\" and k + 1 < n:
                        dir_parts.append(norm[k + 1])
                        k += 2
                        continue
                    if ch == '"':
                        in_str = False
                    k += 1
                    continue
                if in_chr:
                    dir_parts.append(ch)
                    if ch == "\\" and k + 1 < n:
                        dir_parts.append(norm[k + 1])
                        k += 2
                        continue
                    if ch == "'":
                        in_chr = False
                    k += 1
                    continue
                if ch == "/" and k + 1 < n and norm[k + 1] == "*":
                    in_blk = True
                    k += 2
                    continue
                if ch == "/" and k + 1 < n and norm[k + 1] == "/":
                    while k < n and norm[k] != "\n":
                        k += 1
                    if k < n and norm[k] == "\n":
                        k += 1
                    break
                if _is_raw_string_start(norm, k) and re.match(
                    r"^define(?:\s|$)", "".join(dir_parts).lstrip()
                ):
                    raw_end = _consume_raw_string(norm, k + 1)
                    if raw_end is not None:
                        dir_parts.append(norm[k:raw_end])
                        k = raw_end
                        continue
                if ch == '"':
                    in_str = True
                elif ch == "'":
                    in_chr = True
                dir_parts.append(ch)
                k += 1
            i = k
            lines.append((norm[start:i], True, "".join(dir_parts).strip()))
            continue
        k = i
        while k < n:
            ch = norm[k]
            if in_blk:
                if ch == "*" and k + 1 < n and norm[k + 1] == "/":
                    in_blk = False
                    k += 2
                else:
                    k += 1
                continue
            if in_lcom:
                if ch == "\\" and k + 1 < n and norm[k + 1] == "\n":
                    k += 2
                    continue
                if ch == "\n":
                    in_lcom = False
                    k += 1
                    break
                k += 1
                continue
            if in_str:
                if ch == "\\" and k + 1 < n:
                    k += 2
                    continue
                if ch == '"':
                    in_str = False
                elif ch == "\n":
                    in_str = False
                    k += 1
                    break
                k += 1
                continue
            if in_chr:
                if ch == "\\" and k + 1 < n:
                    k += 2
                    continue
                if ch == "'":
                    in_chr = False
                elif ch == "\n":
                    in_chr = False
                    k += 1
                    break
                k += 1
                continue
            if ch == "\\" and k + 1 < n and norm[k + 1] == "\n":
                k += 2
                continue
            if ch == "\n":
                k += 1
                break
            if ch == "/" and k + 1 < n and norm[k + 1] == "*":
                in_blk = True
                k += 2
                continue
            if ch == "/" and k + 1 < n and norm[k + 1] == "/":
                in_lcom = True
                k += 2
                continue
            if _is_raw_string_start(norm, k):
                raw_end = _consume_raw_string(norm, k + 1)
                if raw_end is not None:
                    k = raw_end
                    continue
            if ch == '"':
                in_str = True
            elif ch == "'":
                in_chr = True
            k += 1
        i = k
        lines.append((norm[start:i], False, ""))
    return lines


def _collect_tracked_macro_mutations(
    logical_lines: Sequence[tuple[str, bool, str]],
    defined_macros: frozenset[str],
    tainted_macros: frozenset[str],
) -> tuple[set[str], bool]:
    found: set[str] = set()
    stack: list[list[object]] = []
    for _raw_line, is_dir, dir_text in logical_lines:
        parent_active = all(bool(frame[1]) for frame in stack)
        if not is_dir:
            continue
        parts = dir_text.split(None, 1)
        kw = parts[0] if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        if kw in ("define", "undef"):
            m_id = arg.split("(", 1)[0].split(None, 1)[0] if arg else ""
            if parent_active and (m_id in defined_macros or _is_l78_unset_macro(m_id)):
                found.add(m_id)
            continue
        if kw in ("ifdef", "ifndef", "if"):
            if not parent_active:
                stack.append(["STRIPPED", False, False])
                continue
            if kw == "ifdef":
                m_name = arg.split(None, 1)[0] if arg else ""
                cond = (
                    _eval_macro_defined(m_name, defined_macros, tainted_macros)
                    if IDENT_RE.match(m_name)
                    else None
                )
            elif kw == "ifndef":
                m_name = arg.split(None, 1)[0] if arg else ""
                val = (
                    _eval_macro_defined(m_name, defined_macros, tainted_macros)
                    if IDENT_RE.match(m_name)
                    else None
                )
                cond = None if val is None else (not val)
            else:
                cond = _eval_if_expr(arg, defined_macros, tainted_macros)
            if cond is None:
                stack.append(["OPAQUE", True, True])
            else:
                stack.append(["KNOWN", cond, cond])
            continue
        if kw == "elif":
            if not stack:
                return found, False
            top = stack[-1]
            outer_active = all(bool(frame[1]) for frame in stack[:-1])
            if not outer_active or top[0] == "STRIPPED":
                top[1] = False
                continue
            if top[0] == "OPAQUE":
                top[1] = True
                continue
            if top[2]:
                top[1] = False
                continue
            cond = _eval_if_expr(arg, defined_macros, tainted_macros)
            if cond is None:
                return found, False
            top[1] = cond
            top[2] = cond
            continue
        if kw == "else":
            if not stack:
                return found, False
            top = stack[-1]
            outer_active = all(bool(frame[1]) for frame in stack[:-1])
            if not outer_active or top[0] == "STRIPPED":
                top[1] = False
                continue
            if top[0] == "OPAQUE":
                top[1] = True
                continue
            take_else = not bool(top[2])
            top[1] = take_else
            top[2] = True
            continue
        if kw == "endif":
            if not stack:
                return found, False
            stack.pop()
            continue
    if stack:
        return found, False
    return found, True


def _strip_inactive_branches_from_lines(
    logical_lines: Sequence[tuple[str, bool, str]],
    raw_bytes: bytes,
    defined_macros: frozenset[str],
    tainted_macros: frozenset[str] = frozenset(),
) -> bytes:
    out: list[bytes] = []
    stack: list[list[object]] = []
    for raw_line, is_dir, dir_text in logical_lines:
        parent_active = all(bool(frame[1]) for frame in stack)
        if not is_dir:
            if parent_active:
                out.append(raw_line.encode("utf-8"))
            continue
        parts = dir_text.split(None, 1)
        kw = parts[0] if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        if kw in ("define", "undef"):
            m_id = arg.split("(", 1)[0].split(None, 1)[0] if arg else ""
            if parent_active and (m_id in defined_macros or _is_l78_unset_macro(m_id)):
                return raw_bytes
            if parent_active:
                out.append(raw_line.encode("utf-8"))
            continue
        if kw in ("ifdef", "ifndef", "if"):
            if not parent_active:
                stack.append(["STRIPPED", False, False])
                continue
            if kw == "ifdef":
                m_name = arg.split(None, 1)[0] if arg else ""
                cond = (
                    _eval_macro_defined(m_name, defined_macros, tainted_macros)
                    if IDENT_RE.match(m_name)
                    else None
                )
            elif kw == "ifndef":
                m_name = arg.split(None, 1)[0] if arg else ""
                val = (
                    _eval_macro_defined(m_name, defined_macros, tainted_macros)
                    if IDENT_RE.match(m_name)
                    else None
                )
                cond = None if val is None else (not val)
            else:
                cond = _eval_if_expr(arg, defined_macros, tainted_macros)
            if cond is None:
                stack.append(["OPAQUE", True, True])
                out.append(raw_line.encode("utf-8"))
            else:
                stack.append(["KNOWN", cond, cond])
                out.append(b"\x00IF_KNOWN:" + (b"T" if cond else b"F") + b"\n")
            continue
        if kw == "elif":
            if not stack:
                return raw_bytes
            top = stack[-1]
            outer_active = all(bool(frame[1]) for frame in stack[:-1])
            if not outer_active or top[0] == "STRIPPED":
                top[1] = False
                continue
            if top[0] == "OPAQUE":
                top[1] = True
                out.append(raw_line.encode("utf-8"))
                continue
            if top[2]:
                top[1] = False
                out.append(b"\x00ELIF_KNOWN:F\n")
                continue
            cond = _eval_if_expr(arg, defined_macros, tainted_macros)
            if cond is None:
                return raw_bytes
            top[1] = cond
            top[2] = cond
            out.append(b"\x00ELIF_KNOWN:" + (b"T" if cond else b"F") + b"\n")
            continue
        if kw == "else":
            if not stack:
                return raw_bytes
            top = stack[-1]
            outer_active = all(bool(frame[1]) for frame in stack[:-1])
            if not outer_active or top[0] == "STRIPPED":
                top[1] = False
                continue
            if top[0] == "OPAQUE":
                top[1] = True
                out.append(raw_line.encode("utf-8"))
                continue
            take_else = not bool(top[2])
            top[1] = take_else
            top[2] = True
            out.append(b"\x00ELSE_KNOWN:" + (b"T" if take_else else b"F") + b"\n")
            continue
        if kw == "endif":
            if not stack:
                return raw_bytes
            top = stack.pop()
            outer_active = all(bool(frame[1]) for frame in stack)
            if not outer_active or top[0] == "STRIPPED":
                continue
            if top[0] == "OPAQUE":
                out.append(raw_line.encode("utf-8"))
            else:
                out.append(b"\x00ENDIF_KNOWN\n")
            continue
        if parent_active:
            out.append(raw_line.encode("utf-8"))
    if stack:
        return raw_bytes
    return b"".join(out)


def _strip_inactive_branches(
    text: str,
    defined_macros: frozenset[str],
    tainted_macros: frozenset[str] = frozenset(),
) -> bytes:
    return _strip_inactive_branches_from_lines(
        _split_cpp_logical_lines(text),
        text.encode("utf-8"),
        defined_macros,
        tainted_macros,
    )


def effective_source_digest(flags: Sequence[str] = ()) -> str:
    eff_flags = (f"-DKERNEL_VER={112830}", *BASE_FLAGS, *R4_FLAGS) if not flags else tuple(flags)
    defined_macros = _extract_defined_macros(eff_flags)
    entries: list[tuple[str, bytes, list[tuple[str, bool, str]] | None]] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(ROOT))
        raw_bytes = p.read_bytes()
        if p.suffix.lower() in CPP_SOURCE_SUFFIXES:
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                entries.append((rel, raw_bytes, None))
            else:
                entries.append((rel, raw_bytes, _split_cpp_logical_lines(text)))
        else:
            entries.append((rel, raw_bytes, None))

    tainted: set[str] = set()
    while True:
        cur_tainted = frozenset(tainted)
        newly_found: set[str] = set()
        for _rel, _raw_bytes, logical_lines in entries:
            if logical_lines is None:
                continue
            found, ok = _collect_tracked_macro_mutations(
                logical_lines, defined_macros, cur_tainted
            )
            newly_found |= found
            if not ok:
                for _raw_line, is_dir, dir_text in logical_lines:
                    if not is_dir:
                        continue
                    parts = dir_text.split(None, 1)
                    kw = parts[0] if parts else ""
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    if kw in ("define", "undef"):
                        m_id = arg.split("(", 1)[0].split(None, 1)[0] if arg else ""
                        if m_id in defined_macros or _is_l78_unset_macro(m_id):
                            newly_found.add(m_id)
        if newly_found <= tainted:
            break
        tainted |= newly_found

    tainted_frozen = frozenset(tainted)
    h = hashlib.sha256()
    h.update(json.dumps(list(eff_flags)).encode("utf-8"))
    for rel, raw_bytes, logical_lines in entries:
        h.update(rel.encode("utf-8"))
        if logical_lines is None:
            h.update(raw_bytes)
        else:
            h.update(
                _strip_inactive_branches_from_lines(
                    logical_lines, raw_bytes, defined_macros, tainted_frozen
                )
            )
    return h.hexdigest()


def build_chain(
    out_dir: Path | None = None,
    ver: int = 112830,
    union_sfa: bool = False,
    force: bool = False,
    enable_kernel_prof: bool = False,
) -> tuple[Path, Path]:
    if out_dir is None:
        base_dir = default_build_dir()
        out = base_dir / "prof" if enable_kernel_prof else base_dir
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cann = _cann_root()
    site = _site_root()
    defs = [f"-DKERNEL_VER={ver}", *BASE_FLAGS, *R4_FLAGS]
    if union_sfa:
        defs.append("-DK112840_UNION_SFA=1")
    if enable_kernel_prof:
        defs.append("-DENABLE_KERNEL_PROF=1")
    so_path = out / "chain.so"
    manifest_path = out / "chain.manifest.json"
    eff_source_sha = effective_source_digest(defs)
    if not force and so_path.is_file() and manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("ver") == ver
            and bool(cached.get("union_sfa")) == bool(union_sfa)
            and bool(cached.get("enable_kernel_prof", False)) == bool(enable_kernel_prof)
            and cached.get("flags") == defs
            and cached.get("effective_source_sha256") == eff_source_sha
            and cached.get("sha256") == hashlib.sha256(so_path.read_bytes()).hexdigest()
        ):
            return so_path, manifest_path

    inc = [
        "-I", str(ROOT / "local/ffn/body/source/routed_int4"),
        "-I", str(cann / "include"),
        "-I", str(ROOT / "local/ffn/bridge/vendor"),
        "-I", str(ROOT / "local/ffn/source"),
        "-I", str(ROOT / "vendor"),
        "-I", "/assets/shmem-lib/include",
        "-I", "/assets/shmem-lib/src/device",
    ]
    for sub in ASC_SUBDIRS:
        inc.extend(["-I", str(cann / "aarch64-linux" / sub)])
    host_inc = [
        "-I", str(site / "torch/include"),
        "-I", str(site / "torch/include/torch/csrc/api/include"),
        "-I", str(site / "torch_npu/include"),
        "-I", str(cann / "include"),
        "-I", "/assets/shmem-lib/include",
    ]
    libs = [
        "-L", "/assets/shmem-lib/lib", "-lshmem", "-lshmem_utils",
        "-L", str(cann / "aarch64-linux/lib64"), "-L", str(cann / "lib64"),
        "-lascendcl", "-lruntime", "-lunified_dlog", "-lc_sec",
        "-lregister", "-lprofapi", "-ltiling_api", "-lplatform",
        "-L", str(site / "torch/lib"), "-L", str(site / "torch_npu/lib"),
        "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu",
    ]
    k_obj = out / f"kernel_{ver}.o"
    h_obj = out / f"host_{ver}.o"
    c_obj = out / f"comm_{ver}.o"

    k_cmd = [
        str(cann / "bin/bisheng"), "-xcce", "--npu-arch=dav-2201",
        "-std=c++17", "-O2", "-w", "-ffp-contract=off", "-fPIC",
        *defs, *inc, "-c", str(ROOT / "source/kernel.cpp"), "-o", str(k_obj),
    ]
    h_cmd = [
        "c++", "-std=c++17", "-O2", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *defs, *host_inc, "-c", str(ROOT / "source/host.cpp"), "-o", str(h_obj),
    ]
    c_cmd = [
        "c++", "-std=c++17", "-O2", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *defs, *host_inc, "-c", str(ROOT / "communication/communication_host.cpp"),
        "-o", str(c_obj),
    ]
    link_cmd = [
        "c++", "-shared", "-Wl,--no-undefined", "-Wl,-Bsymbolic",
        str(k_obj), str(h_obj), str(c_obj), "-o", str(so_path), *libs,
    ]
    script = (
        f"set -euo pipefail\n"
        f"set +u; source {shlex.quote(str(cann / 'set_env.sh'))}; set -u\n"
        f"{' '.join(shlex.quote(x) for x in k_cmd)} &\n"
        f"{' '.join(shlex.quote(x) for x in h_cmd)} &\n"
        f"{' '.join(shlex.quote(x) for x in c_cmd)} &\n"
        f"wait\n"
        f"{' '.join(shlex.quote(x) for x in link_cmd)}\n"
    )
    try:
        subprocess.run(["bash", "-c", script], check=True)
    finally:
        for obj in (k_obj, h_obj, c_obj):
            if obj.exists():
                obj.unlink()

    sha256 = hashlib.sha256(so_path.read_bytes()).hexdigest()
    manifest = {
        "ver": ver,
        "union_sfa": bool(union_sfa),
        "enable_kernel_prof": bool(enable_kernel_prof),
        "flags": defs,
        "sha256": sha256,
        "source_sha256": source_digest(),
        "effective_source_sha256": eff_source_sha,
    }
    tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    return so_path, manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--ver", type=int, default=112830)
    parser.add_argument("--union-sfa", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--enable-kernel-prof", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out) if args.out else None
    so_path, manifest_path = build_chain(
        out_dir=out_dir,
        ver=args.ver,
        union_sfa=args.union_sfa,
        force=args.force,
        enable_kernel_prof=args.enable_kernel_prof,
    )
    print(f"built={so_path} manifest={manifest_path}")


if __name__ == "__main__":
    main()
