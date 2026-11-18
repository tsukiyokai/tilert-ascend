from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final


class VerificationError(Exception):
    pass


class IsolationError(Exception):
    pass


class HygieneError(Exception):
    pass


SCHEMA_VERSION: Final[str] = "glm52.e2e.v1"
ISL: Final[int] = 8192
OSL: Final[int] = 1024
DECODE_TOKENS: Final[int] = 1023
FINAL_POS: Final[int] = 9215
TARGET_LAYERS: Final[int] = 78
DRAFT_LAYERS: Final[int] = 1
WORLD_SIZE: Final[int] = 8
EP_SIZE: Final[int] = 8
VOCAB_SIZE: Final[int] = 154880
TPOT_TARGET_MS: Final[float] = 3.0
MIN_WARMUPS: Final[int] = 2
MIN_FORMALS: Final[int] = 5
AB_BA_WARM: Final[int] = 3
AB_BA_FORMAL: Final[int] = 5
MAX_COLLECTIVE_ROWS: Final[int] = 16
MAX_L78_CHUNK_ROWS: Final[int] = 8
DRAFT_SFA_TOPK_OPTIONS: Final[tuple[int, ...]] = (512, 1024, 2048)
TARGET_SFA_TOPK: Final[int] = 2048
WIDTHS: Final[tuple[int, ...]] = (1, 2, 4, 6, 8, 12, 16, 24, 32)
PROMPT_HEAD_TOKENS: Final[tuple[int, ...]] = (154822, 154824, 154827)
PROMPT_TAIL_TOKENS: Final[tuple[int, ...]] = (154828, 154841, 154842)
TPOT_FORMULA: Final[str] = "max(8-rank dt_ms) / 1023.0"
ALLOW_RELAXED_PARITY: Final[bool] = False
MILESTONES: Final[tuple[str, ...]] = ("M1", "M2", "M3", "M4", "M5")
STRICT_LT_MILESTONES: Final[frozenset[str]] = frozenset({"M4", "M5"})
VLLM_FREE_MILESTONES: Final[frozenset[str]] = frozenset({"M2", "M3", "M4", "M5"})
MILESTONE_TPOT_LIMITS: Final[dict[str, float]] = {
    "M1": 6.8,
    "M2": 6.0,
    "M3": 4.0,
    "M4": 3.0,
    "M5": 3.0,
}

FORBIDDEN_HOSTS: Final[tuple[str, ...]] = ("10.83.185.129", "7.244.1.12")
ALLOWED_A3_GENERATION: Final[str] = "A3"
ALLOWED_A3_CHIPS: Final[frozenset[str]] = frozenset({"Ascend910_9392"})
PRIMARY_A3_CHIP: Final[str] = "Ascend910_9392"
ALLOWED_KBENV_STATES: Final[frozenset[str]] = frozenset({"空闲", "部分"})
MIN_KBENV_RAM_PCT: Final[int] = 15
DEFAULT_RUNNER_TIMEOUT_S: Final[float] = 1800.0
REQUIRED_ISOLATION_VARS: Final[tuple[str, ...]] = (
    "TILERT_E2E_RUN_ID",
    "TILERT_RUN_DIR",
    "TILERT_JOB_DIR",
    "TILERT_BENCH_ROOT",
    "ATB_SHARE_MEMORY_NAME_SUFFIX",
    "MASTER_PORT",
    "TILERT_WHOLE_LAYER_ENDPOINT",
    "TILERT_ONESIDED_ENDPOINT",
    "HCCL_NPU_SOCKET_PORT_RANGE",
    "ASCEND_RT_VISIBLE_DEVICES",
)
PROFILER_ENV_VARS: Final[tuple[str, ...]] = (
    "PROFILER_SAMPLECONFIG",
    "PROFILING_MODE",
    "ASCEND_LAUNCH_BLOCKING",
)
INACTIVE_PROFILER_VALUES: Final[frozenset[str]] = frozenset({"", "0", "false", "False"})
SHARED_SHM_BLOCKLIST: Final[frozenset[str]] = frozenset(
    {"/dev/shm", "/dev/shm/", "/dev/shm/agy_tp8"}
)
FORBIDDEN_RUN_IDS: Final[frozenset[str]] = frozenset({".", "..", "agy_tp8"})
DEFAULT_KBENV_BIN: Final[str] = (
    "/home/shanshan/dotfiles/harness/skills/vibe-iris/scripts/experiment/kbenv"
)
DEFAULT_CASE_ROOT: Final[Path] = Path(__file__).resolve().parents[1] / "megakernel_tp8_ep8"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    scenario: str
    seq_len: int
    output_len: int
    first_token_id: int
    prompt_sha256: str
    output_sha256: str


GOLDEN_CASES: Final[dict[str, CaseSpec]] = {
    "case-01-coding": CaseSpec(
        case_id="case-01-coding",
        scenario="coding",
        seq_len=8192,
        output_len=1024,
        first_token_id=13534,
        prompt_sha256="a93d673b4c4138a49cb5db8a57a48abda4b84cf5adfcaf558cf6fd21de585f28",
        output_sha256="2f751372fc82cf290c8e35d3e7cc3cad0296fde4d8d188fba102ac34f82c2962",
    ),
    "case-02-review": CaseSpec(
        case_id="case-02-review",
        scenario="code_review",
        seq_len=8192,
        output_len=1024,
        first_token_id=334,
        prompt_sha256="dc019623c5b1f3f20fbd88ea6629c112e90df1dab1bbe7cee83d65a4d96943cb",
        output_sha256="b4f29fde51f38a2f3d4664a502f6626a02976d72bc4048ec78ca72b47c589ef1",
    ),
    "case-03-refactor": CaseSpec(
        case_id="case-03-refactor",
        scenario="refactoring",
        seq_len=8192,
        output_len=1024,
        first_token_id=1499,
        prompt_sha256="16f65c8119679eacac9aae6558906f0ee731015106acd4240814907d313a5bb1",
        output_sha256="65031e32e3d54f4c465c9af202a7684ffc314d322c2024779aa820f2d90510c8",
    ),
    "case-04-math": CaseSpec(
        case_id="case-04-math",
        scenario="math_reasoning",
        seq_len=8192,
        output_len=1024,
        first_token_id=8419,
        prompt_sha256="ba9be5287d130e052ab71d123d04ee00829de25ea34ca085da01f74c1f671b3f",
        output_sha256="68d439b6ff68308150d53b536ae88d0c13debd44b3d8790854f21aea832f461c",
    ),
}


def compact_tokens_sha256(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def default_tokens_sha256(tokens: list[int]) -> str:
    payload = json.dumps(tokens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def get_case_spec(case_id: str) -> CaseSpec:
    if case_id not in GOLDEN_CASES:
        raise VerificationError(f"Unknown case_id: {case_id}")
    return GOLDEN_CASES[case_id]


def milestone_tpot_limit(milestone: str) -> float:
    if milestone not in MILESTONE_TPOT_LIMITS:
        raise VerificationError(f"Unknown milestone: {milestone}")
    return MILESTONE_TPOT_LIMITS[milestone]


def is_valid_sha256_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)
