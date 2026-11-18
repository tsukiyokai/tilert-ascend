from pathlib import Path

import pytest

from .a3_runner import build_dry_run_bundle, run_a3_formal
from .contracts import DEFAULT_CASE_ROOT


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--a3-live", action="store_true", default=False)
    parser.addoption("--report-dir", action="store", default=None)
    parser.addoption("--runner-cmd", action="store", default=None)
    parser.addoption("--milestone", action="store", default="M5")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "a3_live: requires live Ascend A3 hardware")


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def case_root() -> Path:
    return DEFAULT_CASE_ROOT


@pytest.fixture
def e2e_pkg_dir() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture
def e2e_milestone(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--milestone"))


@pytest.fixture
def live_report_dir(request: pytest.FixtureRequest) -> Path | None:
    raw = request.config.getoption("--report-dir")
    return Path(raw) if raw else None


@pytest.fixture
def live_runner_cmd(request: pytest.FixtureRequest) -> str | None:
    raw = request.config.getoption("--runner-cmd")
    return str(raw) if raw else None


@pytest.fixture
def is_a3_live(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--a3-live"))


def resolve_case01_bundle(
    tmp_path: Path,
    case_root: Path,
    e2e_milestone: str,
    is_a3_live: bool,
    live_report_dir: Path | None,
    live_runner_cmd: str | None,
) -> Path:
    if is_a3_live:
        if live_report_dir is not None:
            return live_report_dir
        if live_runner_cmd is not None:
            out_dir = tmp_path / "case-01-coding"
            run_a3_formal(
                case_id="case-01-coding",
                output_dir=out_dir,
                case_root=case_root,
                runner_cmd=live_runner_cmd,
                milestone=e2e_milestone,
            )
            return out_dir
        pytest.fail("--a3-live requires --report-dir or --runner-cmd")
    return build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "case-01-coding",
        case_root=case_root,
        milestone=e2e_milestone,
    )


@pytest.fixture
def case01_bundle(
    tmp_path: Path,
    case_root: Path,
    e2e_milestone: str,
    is_a3_live: bool,
    live_report_dir: Path | None,
    live_runner_cmd: str | None,
) -> Path:
    return resolve_case01_bundle(
        tmp_path=tmp_path,
        case_root=case_root,
        e2e_milestone=e2e_milestone,
        is_a3_live=is_a3_live,
        live_report_dir=live_report_dir,
        live_runner_cmd=live_runner_cmd,
    )
