"""Check the CPU Prolog helper against a frozen single-NPU observation."""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402

from tilert.models.glm_5_2_ascend import reference_prolog as prolog  # noqa
from tilert.models.glm_5_2_ascend.reference_weights import (  # noqa: E402
    ReferenceWeights, quantize_dynamic_a8,
)


PREFIX = "model.layers.0.self_attn"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_equal(checks, name, actual, expected):
    compatible = (actual.shape == expected.shape
                  and actual.dtype == expected.dtype)
    checks.append({
        "name": name, "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "passed": compatible and torch.equal(actual, expected),
        "unequal": int(torch.count_nonzero(actual != expected))
        if compatible else None,
    })


def read_observation(path, weights):
    report = json.loads(path.read_text())
    artifact = Path(report["artifact"])
    if (report["status"] != "pass"
            or report["device_execution"] is not True
            or digest(artifact) != report["artifact_sha256"]
            or report["identity"]["checkpoint_sha256"]
            != weights.contract["metadata_sha256"]):
        raise ValueError("Frozen device/checkpoint identity differs")
    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    identity = saved["identity"]
    if (json.loads(json.dumps(identity)) != report["identity"]
            or identity["layer"] != 0
            or identity["position"] not in (0, 1)
            or identity["logical_rank"] != 0
            or identity["tp_size"] != 16
            or digest(Path(identity["trace"])) != identity["trace_sha256"]):
        raise ValueError("The observation must be the frozen layer-zero row")
    return report, saved, artifact


def observation_angles(saved):
    if "rotary_cos" not in saved and saved["identity"]["position"] == 0:
        return (torch.ones(1, 64, dtype=torch.bfloat16),
                torch.zeros(1, 64, dtype=torch.bfloat16))
    cos, sin = saved["rotary_cos"], saved["rotary_sin"]
    for name, value in (("cos", cos), ("sin", sin)):
        raw = value.contiguous().view(torch.uint8).numpy().tobytes()
        expected = saved["identity"]["rotary"][name]
        if (value.shape != (1, 64) or value.dtype != torch.bfloat16
                or hashlib.sha256(raw).hexdigest() != expected["sha256"]):
            raise ValueError("Frozen rotary input identity differs")
    return cos, sin


def compare_device(weights, actual, output, checks):
    for name, key in (("q_codes", "query_norm"),
                      ("q_scale", "query_norm_scale"),
                      ("latent", "latent"), ("k_rope", "rope")):
        check_equal(checks, "device." + name, output[name],
                    actual[key].reshape(output[name].shape))
    check_equal(checks, "device.q_rope", output["q_rope"][:, :4],
                actual["query_rope"])
    weight = weights.tensor(PREFIX + ".kv_b_proj.weight", rows=(0, 4 * 448))
    uk = weight.reshape(4, 448, 512)[:, :192]
    absorbed = torch.bmm(output["q_nope"][:, :4].transpose(0, 1).float(),
                         uk.float()).transpose(0, 1).to(torch.bfloat16)
    check_equal(checks, "device.q_nope_absorbed", absorbed,
                actual["query_nope"])


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def scalar_rope(value, cosine, sine):
    output = torch.empty_like(value)
    for row in range(value.shape[0]):
        for head in range(value.shape[1]):
            for lane in range(32):
                even = float(value[row, head, 2 * lane])
                odd = float(value[row, head, 2 * lane + 1])
                cos = float(cosine[row, lane])
                sin = float(sine[row, lane])
                output[row, head, lane] = fp32(
                    fp32(even * cos) + fp32(-odd * sin)
                )
                output[row, head, lane + 32] = fp32(
                    fp32(odd * cos) + fp32(even * sin)
                )
    return output.to(torch.bfloat16)


def check_rows(weights, hidden, raw_query, checks):
    from transformers import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaRotaryEmbedding,
    )

    rotary = GlmMoeDsaRotaryEmbedding(
        GlmMoeDsaConfig.from_dict(weights.config), device="cpu",
    )
    positions = torch.tensor([[0, 1, 8192, 8193]])
    rows = hidden.repeat(4, 1)
    rows = (rows.float() * torch.tensor([1, .75, -1, 1.25])[:, None]).to(
        torch.bfloat16
    )
    cos, sin = rotary(rows.unsqueeze(0), positions)
    cos, sin = cos[0], sin[0]
    before = rows.clone()
    batched = prolog.preprocess(weights, PREFIX, rows, cos, sin)
    for row in range(4):
        single = prolog.preprocess(weights, PREFIX, rows[row:row + 1],
                                   cos[row:row + 1], sin[row:row + 1])
        for name, value in single.items():
            check_equal(checks, f"rows.{row}.{name}",
                        batched[name][row:row + 1], value)
    check_equal(checks, "input_unchanged", rows, before)
    raw = raw_query[..., 192:].repeat(4, 16, 1)
    check_equal(checks, "rope.scalar_positions_0_1_8192_8193",
                prolog.rotary_half(raw, cos, sin), scalar_rope(raw, cos, sin))
    value = torch.tensor([[.988, 1.0]], dtype=torch.float32).repeat(1, 1024)
    gamma = torch.ones(2048, dtype=torch.bfloat16)
    direct, _ = prolog.quantize_prolog(prolog.rms_fp32(value, gamma))
    rounded, _ = prolog.quantize_prolog(prolog.rms_fp32(
        value.to(torch.bfloat16).float(), gamma,
    ))
    checks.append({"name": "early_bf16_changes_query_code",
                   "direct": int(direct[0, 0]),
                   "early_bf16": int(rounded[0, 0]),
                   "passed": int(direct[0, 0]) == 125
                   and int(rounded[0, 0]) == 126})
    return Path(inspect.getsourcefile(GlmMoeDsaRotaryEmbedding))


def check_indexer(weights, output, checks):
    codes, scale = prolog.quantize_prolog(output["q_norm"])
    check_equal(checks, "indexer.same_codes", codes, output["q_codes"])
    check_equal(checks, "indexer.same_scale", scale, output["q_scale"])
    prefix = PREFIX + ".indexer.wq_b"
    integer = weights._integer_weight(prefix)
    expected = weights._integer_product(
        integer, codes, scale, (0, 4096), (0, 2048),
    ).reshape(-1, 32, 128)
    actual = prolog.index_query(weights, PREFIX + ".indexer", output["q_norm"])
    check_equal(checks, "indexer.existing_w8_arithmetic", actual, expected)


def npu_initialized():
    module = sys.modules.get("torch_npu")
    return bool(module is not None and module.npu._initialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    before = npu_initialized()
    weights = ReferenceWeights(args.checkpoint)
    device_report, saved, artifact = read_observation(
        args.device_report, weights,
    )
    actual = saved["actual"]
    checks = []
    with torch.inference_mode():
        from tilert.models.glm_5_2_ascend.reference import prolog_input_norm

        norm = SimpleNamespace(weight=saved["weights"]["input_gamma"],
                               variance_epsilon=1e-5)
        check_equal(checks, "device.production_input_norm",
                    prolog_input_norm(norm, saved["hidden"]),
                    actual["input_norm"])
        hidden = actual["input_norm"]
        codes, scale = quantize_dynamic_a8(hidden)
        check_equal(checks, "input.codes", codes, actual["input_codes"])
        check_equal(checks, "input.scale", scale, actual["input_scale"])
        cos, sin = observation_angles(saved)
        output = prolog.preprocess(weights, PREFIX, hidden, cos, sin)
        compare_device(weights, actual, output, checks)
        reference = saved["cpu_reference"]
        check_equal(checks, "cpu_fp32.q_norm", output["q_norm"],
                    reference["q_a_normalized_fp32"])
        check_equal(checks, "cpu_fp32.q_nope", output["q_nope"][:, :4],
                    reference["q_nope_bf16"])
        installed = check_rows(weights, hidden, reference["q_b_fp32"], checks)
        check_indexer(weights, output, checks)
    after = npu_initialized()
    checks.append({"name": "no_npu_initialization",
                   "passed": not before and not after})
    paths = (Path(__file__), Path(prolog.__file__),
             Path(inspect.getsourcefile(ReferenceWeights)), installed,
             Path(inspect.getsourcefile(prolog_input_norm)))
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU helper versus frozen layer0/rank0 Prolog observation; "
                 "M4 and other positions are CPU formula checks",
        "precision_verified": False, "performance_measured": False,
        "indexer_device_verified": False,
        "checkpoint_sha256": weights.contract["metadata_sha256"],
        "device_identity": saved["identity"],
        "artifact_sha256": {
            str(args.device_report): digest(args.device_report),
            str(artifact): digest(artifact),
        },
        "source_sha256": {str(path): digest(path) for path in paths},
        "device_source_sha256": device_report["source_sha256"],
        "contract": prolog.CONTRACT, "torch": str(torch.__version__),
        "npu_initialized_before": before, "npu_initialized_after": after,
        "checks": checks, "passed": all(item["passed"] for item in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({"passed": result["passed"], "checks": len(checks),
                      "npu_initialized": after}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
