"""Check real MTP checkpoint metadata and fixed-source contracts on CPU."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import tempfile

from tilert.models.glm_5_2_ascend.native_mtp import checkpoint_contract


SOURCES = {
    "vllm": {
        "commit": "c4df37dfcc7f72ca41cd43801a468526ecf99a2b",
        "checks": {
            "vllm/v1/spec_decode/llm_base_proposer.py": (
                "self.needs_extra_input_slots = "
                "self.net_num_new_slots_per_request > 0",
                '1 if (self.pass_hidden_states_to_model '
                'and self.method != "dflash") else 0',
            ),
            "vllm/model_executor/models/deepseek_mtp.py": (
                "previous_hidden_states = self.hnorm(previous_hidden_states)",
                "torch.cat([inputs_embeds, previous_hidden_states], dim=-1)",
                "return hidden_states, self.shared_head(hidden_states)",
                "torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)",
            ),
            "vllm/model_executor/models/deepseek_v2.py": (
                "hidden_states, _ = self.norm(hidden_states, residual)",
            ),
            "vllm/config/speculative.py": (
                "parallel_drafting: bool = False",
            ),
        },
    },
    "ascend": {
        "commit": "f558b58c177916424ae9dcf9bb8e66c0b9031236",
        "checks": {
            "vllm_ascend/spec_decode/eagle_proposer.py": (
                "device, True, runner=runner)",
            ),
            "vllm_ascend/models/deepseek_mtp.py": (
                "hidden_states = self.rot(hidden_states)",
            ),
            "vllm_ascend/spec_decode/llm_base_proposer.py": (
                "self.input_ids[: num_tokens - 1] = target_token_ids[1:]",
                "self._set_positions(num_tokens, target_positions)",
                "self.hidden_states[:num_tokens] = target_hidden_states.view",
                "share_embeddings = not self.use_compress",
                "draft_model.set_skip_topk(True)",
                "slot_mapping = block_ids * block_size "
                "+ clamped_positions % block_size",
            ),
            "vllm_ascend/attention/sfa_v1.py": (
                "compute_topk=not self.skip_topk and has_local_query",
                "self._get_indexcache_topk_indices"
                "(parallel_context.topk_num_tokens)",
            ),
        },
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_checks(roots):
    records = []
    for family, spec in SOURCES.items():
        root = roots[family]
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        records.append({"name": family + ".commit", "actual": commit,
                        "passed": commit == spec["commit"]})
        for name, fragments in spec["checks"].items():
            path = root / name
            lines = path.read_text().splitlines()
            matches = []
            for fragment in fragments:
                matches.append({"fragment": fragment, "lines": [
                    number for number, line in enumerate(lines, 1)
                    if fragment in line
                ]})
            records.append({"name": name, "sha256": digest(path),
                            "passed": all(item["lines"] for item in matches),
                            "anchors": matches})
    return records


def reference_hash_checks(root):
    runner = runpy.run_path(str(root / "tests/glm52/run_native_mtp.py"))
    validate = runner["validate_reference"]
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        artifact = directory / "rank-0.pt"
        artifact.write_bytes(b"Opaque bytes for the SHA256 contract only")
        checkpoint = {"config.json": "b" * 64}
        scorer = "tilert/models/glm_5_2_ascend/ops/indexer.py"
        manifest = {
            "status": "pass", "numerical_execution": True,
            "implementation": "transformers.GlmMoeDsaDecoderLayer.MTP",
            "source_sha256": {scorer: digest(root / scorer)},
            "input_sha256": "c" * 64, "checkpoint_sha256": checkpoint,
            "position_convention": "parent",
            "rank_sha256": {artifact.name: digest(artifact)},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        validate(directory, "c" * 64, checkpoint, 0)
        checks.append({"name": "reference_hash_accepts_matching_bytes",
                       "passed": True})
        for field, value in (
            ("status", "metadata_only"), ("numerical_execution", False),
            ("source_sha256", {"fixture.py": "a" * 64}),
            ("source_sha256", {scorer: "0" * 64}),
        ):
            previous = manifest[field]
            manifest[field] = value
            (directory / "manifest.json").write_text(json.dumps(manifest))
            try:
                validate(directory, "c" * 64, checkpoint, 0)
            except ValueError:
                rejected = True
            else:
                rejected = False
            checks.append({"name": f"reference_rejects.{field}.{value}",
                           "passed": rejected})
            manifest[field] = previous
        manifest["implementation"] = "NativeMtp"
        (directory / "manifest.json").write_text(json.dumps(manifest))
        try:
            validate(directory, "c" * 64, checkpoint, 0)
        except ValueError:
            rejected = True
        else:
            rejected = False
        checks.append({"name": "reference_rejects_production_implementation",
                       "passed": rejected})
        manifest["implementation"] = "transformers.GlmMoeDsaDecoderLayer.MTP"
        manifest["prefix_files_sha256"] = {artifact.name: "e" * 64}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        validate(directory, "c" * 64, checkpoint, 0, prefix_hash="e" * 64)
        checks.append({"name": "reference_prefix_accepts_matching_hash",
                       "passed": True})
        try:
            validate(directory, "c" * 64, checkpoint, 0, prefix_hash="f" * 64)
        except ValueError:
            rejected = True
        else:
            rejected = False
        checks.append({"name": "reference_prefix_rejects_mismatch",
                       "passed": rejected})
        manifest["prefix_files_sha256"] = {}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        artifact.write_bytes(b"Changed bytes")
        try:
            validate(directory, "c" * 64, checkpoint, 0)
        except ValueError:
            rejected = True
        else:
            rejected = False
        checks.append({"name": "reference_hash_rejects_changed_bytes",
                       "passed": rejected})
        manifest.pop("rank_sha256")
        (directory / "manifest.json").write_text(json.dumps(manifest))
        try:
            validate(directory, "c" * 64, checkpoint, 0)
        except ValueError:
            rejected = True
        else:
            rejected = False
        checks.append({"name": "reference_requires_rank_hash",
                       "passed": rejected})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--ascend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = checkpoint_contract(args.checkpoint)
    checks = source_checks({"vllm": args.vllm, "ascend": args.ascend})
    root = Path(__file__).resolve().parents[2]
    checks.extend(reference_hash_checks(root))
    sources = {}
    for name in (
        "tilert/models/glm_5_2_ascend/native_mtp.py",
        "tilert/models/glm_5_2_ascend/ops/indexer.py",
        "tests/glm52/run_native_mtp.py", "tests/glm52/check_mtp_contract.py",
    ):
        path = root / name
        text = path.read_text()
        ast.parse(text, feature_version=(3, 11))
        sources[name] = digest(path)
        checks.append({"name": name + ".line_length", "passed": all(
            len(line) <= 79 for line in text.splitlines()
        )})
    checks.append({"name": "ordinary_import_without_torch",
                   "passed": "torch" not in sys.modules})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": sys.version,
        "command": [sys.executable, *sys.argv],
        "scope": "Checkpoint headers, source anchors and Python syntax",
        "passed": all(item["passed"] for item in checks),
        "contract": contract, "source_sha256": sources,
        "checkpoint_sha256": {
            name: digest(args.checkpoint / name) for name in (
                "config.json", "quant_model_description.json",
                "quant_model_weights.safetensors.index.json",
            )
        },
        "device_execution": False, "numerical_parity": False,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"],
                      "checks": len(checks), "report": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
