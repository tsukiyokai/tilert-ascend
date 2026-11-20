import argparse
import json
import math
from pathlib import Path
from typing import Any
import uuid

from .a3_runner import build_dry_run_bundle, run_a3_formal, validate_protocol_counts
from .contracts import (
    AB_BA_FORMAL,
    AB_BA_WARM,
    DEFAULT_CASE_ROOT,
    DEFAULT_KBENV_BIN,
    DEFAULT_RUNNER_TIMEOUT_S,
    DEFAULT_TRACK,
    GOLDEN_CASES,
    GOLDEN_TRACKS,
    HygieneError,
    IsolationError,
    MILESTONES,
    MIN_FORMALS,
    MIN_WARMUPS,
    WIDTHS,
    VerificationError,
    milestone_tpot_limit,
)
from .hygiene_gate import allocate_free_port_base, verify_root_hygiene
from .parity_gate import verify_e2e_run


def _resolve_cases(case_arg: str) -> list[str]:
    if case_arg == "all":
        return list(GOLDEN_CASES)
    if case_arg not in GOLDEN_CASES:
        raise VerificationError(f"Unsupported case: {case_arg}")
    return [case_arg]


def _validate_max_tpot(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise VerificationError(f"--max-tpot-ms must be finite positive, got {value!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tests.glm52.e2e_tp8_ep8.cli")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "verify-report", "a3-formal"),
        required=True,
    )
    parser.add_argument(
        "--case",
        choices=(*GOLDEN_CASES.keys(), "all"),
        default="case-01-coding",
    )
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--track", choices=tuple(GOLDEN_TRACKS), default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--formals", type=int, default=None)
    parser.add_argument("--ab-ba", action="store_true")
    parser.add_argument("--milestone", choices=MILESTONES, default="M5")
    parser.add_argument("--max-tpot-ms", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--runner-cmd", type=str, default=None)
    parser.add_argument("--kbenv-bin", type=str, default=DEFAULT_KBENV_BIN)
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--port-base", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_RUNNER_TIMEOUT_S)
    return parser


def _write_summary(output_path: Path | None, payload: dict[str, Any]) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    try:
        cases = _resolve_cases(args.case)
        raw_tpot_ms = (
            milestone_tpot_limit(args.milestone)
            if args.max_tpot_ms is None
            else float(args.max_tpot_ms)
        )
        max_tpot_ms = _validate_max_tpot(raw_tpot_ms)
        results: list[dict[str, Any]] = []

        if args.mode == "dry-run":
            resolved_track = args.track or DEFAULT_TRACK
            warmups, formals = validate_protocol_counts(
                MIN_WARMUPS if args.warmups is None else args.warmups,
                MIN_FORMALS if args.formals is None else args.formals,
                args.ab_ba,
            )
            width = 16 if args.width is None else args.width
            if width not in WIDTHS:
                raise VerificationError(f"Unsupported width: {width} (expected in {WIDTHS})")
            base_out = (
                args.output.parent
                if args.output is not None
                else repo_root / "tests" / ".artifacts" / "e2e_dry_run"
            )
            base_port = 29900 if args.port_base is None else args.port_base
            for idx, case_id in enumerate(cases):
                case_out = base_out / case_id
                build_dry_run_bundle(
                    case_id=case_id,
                    output_dir=case_out,
                    case_root=args.case_root,
                    warmups=warmups,
                    formals=formals,
                    ab_ba=args.ab_ba,
                    milestone=args.milestone,
                    width=width,
                    port_base=base_port + idx * 300,
                    track=resolved_track,
                )
                results.append(
                    verify_e2e_run(
                        case_out,
                        args.case_root / case_id,
                        max_tpot_ms=max_tpot_ms,
                        require_warm_head=True,
                        require_ab_ba=args.ab_ba,
                        milestone=args.milestone,
                        track=resolved_track,
                    )
                )
            hygiene = verify_root_hygiene(repo_root)
            summary: dict[str, Any] = {
                "pass": True,
                "mode": args.mode,
                "track": resolved_track,
                "milestone": args.milestone,
                "cases": results,
                "hygiene": hygiene,
            }
        elif args.mode == "verify-report":
            if args.report_dir is None:
                raise VerificationError("--report-dir is required for verify-report mode")
            req_warm = AB_BA_WARM if args.ab_ba else MIN_WARMUPS
            req_form = AB_BA_FORMAL if args.ab_ba else MIN_FORMALS
            if args.warmups is not None and args.warmups < req_warm:
                raise VerificationError(
                    f"warmups ({args.warmups}) must be >= {req_warm} for ab_ba={args.ab_ba}"
                )
            if args.formals is not None and args.formals < req_form:
                raise VerificationError(
                    f"formals ({args.formals}) must be >= {req_form} for ab_ba={args.ab_ba}"
                )
            if args.width is not None and args.width not in WIDTHS:
                raise VerificationError(
                    f"Unsupported width: {args.width} (expected in {WIDTHS})"
                )
            for case_id in cases:
                target_dir = args.report_dir / case_id if args.case == "all" else args.report_dir
                results.append(
                    verify_e2e_run(
                        target_dir,
                        args.case_root / case_id,
                        max_tpot_ms=max_tpot_ms,
                        require_warm_head=True,
                        require_ab_ba=args.ab_ba,
                        milestone=args.milestone,
                        track=args.track,
                    )
                )
                report = json.loads((target_dir / "e2e_result.json").read_text(encoding="utf-8"))
                if args.warmups is not None and report["protocol"]["warmups"] != args.warmups:
                    raise VerificationError(
                        f"Report protocol.warmups {report['protocol']['warmups']!r} "
                        f"!= expected {args.warmups}"
                    )
                if args.formals is not None and report["protocol"]["formals"] != args.formals:
                    raise VerificationError(
                        f"Report protocol.formals {report['protocol']['formals']!r} "
                        f"!= expected {args.formals}"
                    )
                if args.width is not None and report.get("width", 16) != args.width:
                    raise VerificationError(
                        f"Report width {report.get('width', 16)!r} != expected {args.width}"
                    )
            summary = {
                "pass": True,
                "mode": args.mode,
                "milestone": args.milestone,
                "cases": results,
            }
        else:
            resolved_track = args.track or DEFAULT_TRACK
            warmups, formals = validate_protocol_counts(
                MIN_WARMUPS if args.warmups is None else args.warmups,
                MIN_FORMALS if args.formals is None else args.formals,
                args.ab_ba,
            )
            width = 16 if args.width is None else args.width
            if width not in WIDTHS:
                raise VerificationError(f"Unsupported width: {width} (expected in {WIDTHS})")
            if args.output is not None:
                base_out = args.output.parent
            else:
                batch_tag = f"run_{uuid.uuid4().hex[:8]}"
                base_out = repo_root / "tests" / ".artifacts" / "e2e_a3_formal" / batch_tag
            reserved_ports: list[int] = []
            for idx, case_id in enumerate(cases):
                case_out = base_out / case_id
                if args.port_base is not None:
                    port_base = args.port_base + idx * 300
                else:
                    port_base = allocate_free_port_base(reserved_ports)
                    reserved_ports.append(port_base)
                results.append(
                    run_a3_formal(
                        case_id=case_id,
                        output_dir=case_out,
                        case_root=args.case_root,
                        runner_cmd=args.runner_cmd,
                        kbenv_bin=args.kbenv_bin,
                        warmups=warmups,
                        formals=formals,
                        ab_ba=args.ab_ba,
                        milestone=args.milestone,
                        max_tpot_ms=max_tpot_ms,
                        width=width,
                        target_host=args.target_host,
                        port_base=port_base,
                        timeout_s=float(args.timeout_s),
                        track=resolved_track,
                    )
                )
            summary = {
                "pass": True,
                "mode": args.mode,
                "track": resolved_track,
                "milestone": args.milestone,
                "cases": results,
            }
        _write_summary(args.output, summary)
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except (VerificationError, IsolationError, HygieneError, OSError, ValueError) as exc:
        err_payload = {"pass": False, "mode": args.mode, "error": str(exc)}
        _write_summary(args.output, err_payload)
        print(json.dumps(err_payload, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
