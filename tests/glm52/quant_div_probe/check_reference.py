"""Check calibrated CPU DynamicQuant against frozen native A3 outputs."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from check_cann_prolog import prolog_reference  # noqa: E402
from replay_cann_indexer import (  # noqa: E402
    boundary, difference, score, selection_difference,
)
from tilert.models.glm_5_2_ascend.reference import (  # noqa: E402
    interleaved_rope,
)
from tilert.models.glm_5_2_ascend.reference_prolog import (  # noqa: E402
    project_codes,
)
from tilert.models.glm_5_2_ascend.reference_weights import (  # noqa: E402
    ReferenceWeights, quantize_dynamic_a8,
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reconstruct(saved, weights):
    normalized = saved["reference"]["input_norm"]
    codes, scale = quantize_dynamic_a8(normalized)
    prolog = prolog_reference(codes, scale, saved["prolog_weights"],
                              saved["cos"], saved["sin"])
    stem = "model.layers.0.self_attn.indexer"
    query = project_codes(
        weights, stem + ".wq_b", prolog["query_norm"],
        prolog["query_norm_scale"], scale_order="channel_first",
    ).bfloat16().reshape(1, 1, 32, 128)
    query = torch.cat((
        interleaved_rope(query[..., :64], saved["cos"].unsqueeze(0),
                         saved["sin"].unsqueeze(0)), query[..., 64:],
    ), dim=-1).reshape(1, 32, 128)
    # Key and head-weight projections are unchanged FLOAT operations.
    key, head_weights = (saved["reference"][name]
                         for name in ("key", "weights"))
    scores = score(query, torch.cat((saved["prefix_keys"], key)), head_weights)
    indices = scores.argsort(descending=True, stable=True)[:2048]
    return {"input_codes": codes, "input_scale": scale.flatten(),
            "q_codes": prolog["query_norm"],
            "q_scale": prolog["query_norm_scale"].flatten(),
            "query": query, "key": key, "weights": head_weights,
            "scores": scores, "indices": indices}


@torch.inference_mode()
def run(args):
    calibration_report = args.calibration.with_suffix(".json")
    capture_report = args.capture / "result.json"
    calibration = json.loads(calibration_report.read_text())
    capture = json.loads(capture_report.read_text())
    assert digest(args.calibration) == calibration["artifacts"][
        str(args.calibration)]
    assert digest(args.capture / "input.pt") == capture["prepared_sha256"]
    assert digest(args.capture / "device.pt") == capture["device_sha256"]
    raw = torch.load(args.calibration, map_location="cpu", weights_only=True)
    maximum = (raw["amax_bf16_bits"] << 16).view(torch.float32)
    rows = torch.stack((maximum, maximum * 0.5, -maximum * 0.5), -1)
    codes, _ = quantize_dynamic_a8(rows.bfloat16())
    native = raw["integers"][:, 0].reshape(-1).to(torch.int8)
    zero_codes, zero_scale = quantize_dynamic_a8(
        torch.zeros(2, 64, dtype=torch.bfloat16),
    )
    checks = {"positive_half": torch.equal(codes[:, 1], native),
              "negative_half": torch.equal(codes[:, 2], -native),
              "amax": bool((codes[:, 0] == 127).all()),
              "zero_rows": not bool(zero_codes.any() or zero_scale.any())}
    saved = torch.load(args.capture / "input.pt", map_location="cpu",
                       weights_only=True)
    actual = torch.load(args.capture / "device.pt", map_location="cpu",
                        weights_only=True)
    weights = ReferenceWeights(args.checkpoint)
    assert (weights.contract["metadata_sha256"]
            == saved["identity"]["checkpoint_sha256"])
    reference = reconstruct(saved, weights)
    comparisons = {name: difference(value, actual[name])
                   for name, value in reference.items()
                   if name not in ("scores", "indices")}
    for name in ("input_codes", "input_scale", "q_codes", "query",
                 "key", "weights"):
        checks[name] = comparisons[name]["equal"]
    choices = selection_difference(reference["indices"], actual["indices"])
    checks["choice_set"] = choices["equal"]
    module = sys.modules.get("torch_npu.npu")
    initialized = bool(module and module._initialized)
    checks["cpu_only"] = not initialized
    paths = [args.calibration, calibration_report, capture_report,
             args.capture / "input.pt", args.capture / "device.pt",
             Path(__file__)]
    paths.extend(Path(inspect.getsourcefile(function)) for function in (
        quantize_dynamic_a8, project_codes, prolog_reference,
    ))
    result = {"status": "pass" if all(checks.values()) else "fail",
              "checks": checks, "calibration_rows": maximum.numel(),
              "comparisons": comparisons, "choice_set": choices,
              "cpu_boundary": boundary(reference["scores"]),
              "npu_initialized": initialized, "performance_measured": False,
              "artifacts": {str(path): digest(path) for path in paths}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return result["status"] == "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration", "capture", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(not run(args))


if __name__ == "__main__":
    main()
