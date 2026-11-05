"""Require matching TP16 device precision evidence before performance runs."""

import hashlib
import json
from pathlib import Path

from run_native_decode import validate_reference_manifest


ROOT = Path(__file__).resolve().parents[2]
MODEL = "tilert/models/glm_5_2_ascend/"
CHECKPOINT_FILES = (
    "config.json", "quant_model_description.json",
    "quant_model_weights.safetensors.index.json",
)
MAIN_SOURCES = tuple(MODEL + name for name in (
    "native_engine.py", "config.py", "engine.py", "weights.py",
    "moe_plan.py", "ops/expert.py", "ops/indexer.py", "ops/mla.py",
    "ops/quantized.py",
))
REFERENCE_SOURCES = tuple(MODEL + name for name in (
    "reference.py", "reference_weights.py", "ops/indexer.py",
))
HF_SOURCES = REFERENCE_SOURCES + (
    "tests/glm52/run_hf_reference.py", "tests/glm52/run_native_decode.py",
    "tests/glm52/prefix_io.py",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_precision(
    directory, checkpoint, *, mtp_width, expert_backend="per_expert",
    source_root=None,
):
    """Validate one 8K case; return source and receipt identities.

    main/ and verify/ contain sixteen native numerical reports. MTP also
    requires mtp_seed/, mtp_chain/, mtp_verify/ and session/. This preflight
    checks saved numerical evidence; it does not execute a model or time it.
    """
    directory = Path(directory)
    source_root = ROOT if source_root is None else Path(source_root)
    require(mtp_width in (1, 2, 4), "Expected MTP width 1, 2 or 4")
    require(expert_backend in ("bank", "per_expert"),
            "Expected bank or per_expert backend")
    metadata = {name: digest(Path(checkpoint) / name)
                for name in CHECKPOINT_FILES}
    sources, receipts, session_prefixes, prefix_hashes = {}, {}, {}, {}
    package_versions = None
    prompt = None

    def check_sources(report, required):
        saved = report.get("source_sha256", {})
        for name in required:
            actual = digest(source_root / name)
            require(saved.get(name) == actual,
                    f"Precision source is absent or changed: {name}")
            if name.startswith("tilert/"):
                sources[name] = actual

    def load(stage, rank, statuses, required):
        nonlocal package_versions

        path = directory / stage / f"rank-{rank}.json"
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"Precision report is unavailable: {path}") \
                from error
        require(report.get("status") in statuses,
                f"Device precision has not passed: {path}")
        require(report.get("rank") == rank and report.get("world_size") == 16,
                f"Precision report is not from TP16 rank {rank}: {path}")
        require(report.get("device") == f"npu:{rank}",
                f"Precision report lacks the matching NPU device: {path}")
        checks = report.get("checks")
        require(isinstance(checks, list) and checks
                and all(isinstance(item, dict) and item.get("passed") is True
                        for item in checks),
                f"Precision checks are missing or failed: {path}")
        require(report.get("metadata_sha256", report.get("checkpoint_sha256"))
                == metadata, f"Precision checkpoint differs: {path}")
        require(not report.get("timed") and not report.get("profiled")
                and not report.get("performance_measured"),
                f"Expected a numerical run without profiling: {path}")
        versions = report.get("versions", report.get("packages", report))
        current = {name: versions.get(name) for name in ("torch", "torch_npu")}
        require(all(isinstance(value, str) and value
                    for value in current.values()),
                f"Precision report lacks Torch/NPU versions: {path}")
        if package_versions is None:
            package_versions = current
        require(current == package_versions,
                f"Precision reports use different packages: {path}")
        check_sources(report, required)
        receipts[str(path)] = digest(path)
        return report

    math_sources = MAIN_SOURCES + (
        (MODEL + "ops/expert_bank.py",) if expert_backend == "bank" else ()
    )
    for rank in range(16):
        main = load("main", rank, {"pass"} if rank == 0
                    else {"rank_forward_complete"}, MAIN_SOURCES + (
                        "tests/glm52/run_native_decode.py",
                    ))
        require(main.get("implementation") == "NativeDecodeEngine"
                and main.get("initial_position") == 8192
                and len(main.get("teacher_tokens", [])) >= 32,
                "Main precision requires an 8K prefix and 32 teacher rows")
        main_prefix = main.get("prefix_file_sha256")
        require(is_digest(main_prefix),
                "Main precision requires its frozen prefix file hash")
        prefix_hashes[str(rank)] = {"main": main_prefix, "mtp": None}
        verify = load("verify", rank, {"pass"}, math_sources + (
            "tests/glm52/run_native_verify.py",
            "tests/glm52/run_native_decode.py",
        ))
        require(verify.get("implementation") == "NativeDecodeEngine.verify"
                and verify.get("initial_position") == 8192
                and len(verify.get("teacher_tokens", [])) == 4
                and verify.get("expert_backend") == expert_backend,
                "Verification precision has the wrong shape or backend")
        require(main.get("reference_manifest")
                == verify.get("reference_manifest"),
                "Main and verification precision use different HF traces")
        for report in (main, verify):
            reference = report.get("reference_manifest", {})
            validate_reference_manifest(reference, metadata)
            check_sources(reference, HF_SOURCES)
            require(reference.get("initial_position") == 8192
                    and len(reference.get("teacher_tokens", [])) >= 32
                    and reference.get("prefix_files_sha256", {}).get(
                        f"rank-{rank}.pt"
                    ) == main_prefix,
                    "HF precision requires 32 rows after the 8K prefix")
            identity = {name: reference.get(name) for name in (
                "prompt_length", "prompt_sha256", "last_prompt_token",
            )}
            require(identity["prompt_length"] == 8192
                    and is_digest(identity["prompt_sha256"])
                    and type(identity["last_prompt_token"]) is int
                    and 0 <= identity["last_prompt_token"] < 154880,
                    "HF precision lacks its original 8K prompt identity")
            if prompt is None:
                prompt = identity
            require(identity == prompt,
                    "Precision ranks use different original prompts")
        require(main["teacher_tokens"] == reference["teacher_tokens"]
                and verify["teacher_tokens"]
                == reference["teacher_tokens"][:4],
                "Native and HF precision consume different teacher tokens")
        if mtp_width == 1:
            continue
        mtp_sources = math_sources + (MODEL + "native_mtp.py",)
        for stage, parent, position, width, external in (
            ("mtp_seed", "target_final_norm", 8191, 1, False),
            ("mtp_chain", "mtp_recycle_norm", 8192, 1, True),
            ("mtp_verify", "target_final_norm", 8192, mtp_width, False),
        ):
            report = load(stage, rank, {"parity"}, mtp_sources + (
                "tests/glm52/run_native_mtp.py", "tests/glm52/prefix_io.py",
            ))
            require(report.get("passed") is True
                    and report.get("parent_kind") == parent
                    and report.get("position") == position
                    and report.get("width") == width
                    and report.get("external_indices") is external
                    and report.get("expert_backend") == expert_backend
                    and is_digest(report.get("input_sha256"))
                    and is_digest(report.get("prefix_sha256")),
                    f"MTP precision has the wrong input role: {stage}")
            oracle = report.get("oracle", {})
            require(oracle.get("status") == "pass"
                    and oracle.get("numerical_execution") is True
                    and oracle.get("implementation")
                    == "transformers.GlmMoeDsaDecoderLayer.MTP"
                    and oracle.get("checkpoint_sha256") == metadata
                    and oracle.get("input_sha256")
                    == report.get("input_sha256")
                    and oracle.get("prefix_files_sha256", {}).get(
                        f"rank-{rank}.pt"
                    ) == report["prefix_sha256"]
                    and oracle.get("position_convention") == "parent",
                    f"MTP precision lacks its numerical HF oracle: {stage}")
            check_sources(oracle, REFERENCE_SOURCES + (
                MODEL + "mtp_reference.py", "tests/glm52/run_mtp_reference.py",
                "tests/glm52/prefix_io.py",
            ))
            tensor = directory / stage / f"rank-{rank}.pt"
            require(report.get("trace_sha256") == digest(tensor),
                    f"MTP device trace differs: {tensor}")
            if stage == "mtp_seed":
                prefix_hashes[str(rank)]["mtp"] = report.get("prefix_sha256")
        control = tuple(MODEL + name for name in (
            "mtp_round.py", "mtp_state.py", "mtp_session.py", "session.py",
            "tp_runtime.py",
        )) + ("tilert/pd_vllm/ascend_cache.py",
              "tilert/pd_vllm/ascend_server.py")
        session = load("session", rank, {"capture_replay"},
                       mtp_sources + control + (
                           "tests/glm52/run_native_mtp_session.py",
                           "tests/glm52/run_native_mtp_rounds.py",
                           "tests/glm52/prefix_io.py",
                       ))
        require(session.get("initial_position") == 8192
                and session.get("mtp_width") == mtp_width
                and session.get("expert_backend") == expert_backend
                and session.get("ignore_eos") is True
                and len(session.get("serial_tokens", [])) >= 16
                and len(session.get("rounds", [])) >= 2,
                "Session precision requires complete speculative requests")
        states = session.get("final_rank_states", [])
        require(len(states) == 16
                and all(len(state) == 3 and state[:2] == [0, 0]
                        and state[2] == states[0][2] for state in states),
                "Session precision lacks a complete TP16 reset")
        prefixes = session.get("prefix_sha256", {})
        require(len(prefixes) == 2
                and all(Path(name).name == f"rank-{rank}.pt"
                        for name in prefixes)
                and set(prefixes.values())
                == set(prefix_hashes[str(rank)].values()),
                "Session precision requires both rank-local prefix hashes")
        session_prefixes[str(rank)] = list(prefixes.values())
        token = session.get("last_prompt_token")
        require(token == prompt["last_prompt_token"],
                "Session replay token differs from the original prompt")
    return {
        "scope": "Saved TP16 8K numerical precision; no performance result",
        "checkpoint_sha256": metadata, "mtp_width": mtp_width,
        "expert_backend": expert_backend, "packages": package_versions,
        "source_sha256": sources, "reports": receipts,
        "session_prefix_sha256": session_prefixes,
        "prefix_sha256": prefix_hashes, "initial_position": 8192,
        **prompt,
    }


def validate_workload_precision(directory, checkpoint, **kwargs):
    cases = [validate_precision(Path(directory) / f"case-{case:02d}",
                                checkpoint, **kwargs) for case in range(8)]
    result = {name: cases[0][name] for name in (
        "checkpoint_sha256", "mtp_width", "expert_backend", "packages",
        "source_sha256",
    )}
    require(all(all(case[name] == result[name] for name in result)
                for case in cases),
            "Precision cases use different packages or implementations")
    prompts = [case["prompt_sha256"] for case in cases]
    require(len(set(prompts)) == 8,
            "Precision workload requires eight different 8K prompts")
    result.update(scope="Eight 8K/32-row TP16 numerical cases; no timing",
                  cases=cases, prompt_sha256s=prompts)
    return result
