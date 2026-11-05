"""Audit the original DecoderLayer construction and postprocessing gate."""

import hashlib
import json
from pathlib import Path
import shutil

output = Path("docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-constructor-v2-a3-194")
load = lambda path: json.loads(path.read_text())
ranks = [load(output / f"rank-{rank:02d}.json") for rank in range(16)]
launcher = load(output / "summary.json")
assert launcher["success"] and launcher["exitcodes"] == [0] * 16
for rank, report in enumerate(ranks):
    assert report["rank"] == rank and not report.get("error")
    assert report["rank_zero_status"]["success"]
    assert report["npu_initialized_before"] is False
    assert report["npu_initialized_after"] == (rank == 0)
    assert all(item["success"] for item in report["stages"])
    group = next(item for item in report["stages"] if item["name"] == "actual_tp16_gloo_groups")
    assert group["tp_world_size"] == 16 and group["tp_rank"] == rank
    assert group["cpu_rank_sum"] == group["expected_rank_sum"] == 120
stages = {item["name"]: item for item in ranks[0]["stages"]}
metadata = stages["actual_rank_zero_hccl_metadata_group"]
assert metadata["ranks"] == [0] and metadata["world_size"] == 1
assert metadata["comm_name_nonempty"] and metadata["tp_collective_validation"] is False
assert "ProcessGroupHCCL" in metadata["backend_class"]
resolvers = stages["actual_official_extension"]["api_libraries"]
assert len(resolvers) == 6
for symbol, identity in resolvers.items():
    expected = ("347437ef3aaf109b4088af90763a2330fb66bb5278e63245366afa8fd7a2233a"
                if "GroupedMatmul" in symbol else
                "b24a676805eac8a528fd6534ad890c038fd68e01e28935f2cb9d22d9f6f53cda")
    assert identity["sha256"] == expected
shared = []
for layer in (6, 7):
    ctor = stages[f"actual_decoder_layer{layer}_constructor"]
    assert ctor["selection"]["layer_index"] == layer
    assert ctor["selection"]["ffn"] == "sparse"
    assert ctor["selection"]["indexer"] == ("full" if layer == 6 else "shared")
    assert ctor["actual_skip_topk"] == (layer == 7)
    assert ctor["attention_reduce_results"] is True
    assert ctor["tp"] == 16 and ctor["ep"] == ctor["dp"] == ctor["pcp"] == 1
    assert ctor["topk_owner_is_supplied_buffer"]
    assert ctor["scheme"]["class"].endswith("AscendW4A8DynamicFusedMoEMethod")
    assert ctor["communication"]["class"].endswith("AllGatherCommImpl")
    weights = stages[f"layer{layer}_processed_weight_contract"]
    for key, shape in (("w13_weight", [256, 6144, 32]), ("w2_weight", [256, 128, 768])):
        assert weights[key]["shape"] == shape
        assert weights[key]["dtype"] == "torch.int32" and weights[key]["format"] == 29
        assert weights[key]["device"] == "npu:0"
    parameters = stages[f"layer{layer}_move_parameters"]["parameters"]
    gu = next(value for value in parameters if value["name"] == "mlp.shared_experts.gate_up_proj.weight")
    assert gu["dtype"] == "torch.bfloat16"
    shared.append({"layer": layer, "gate_up_dtype": gu["dtype"]})
result = ranks[0]["result"]
assert result["constructed_layers"] == [6, 7] and result["official_postprocessing"]
assert result["full_decoder_forward_executed"] is False
assert result["tp16_collectives_executed"] == 0
summary = {
    "pass": True,
    "scope": "Original TP16 layer-6 and layer-7 constructors and official weight postprocessing with synthetic parameters; one NPU, no forward or TP16 collective.",
    "rank_successes": 16, "npu_rank_count": 1, "layers": [6, 7],
    "routed_weight_contract": "Packed INT4 stored in INT32 NZ, H6144 and per-rank I128",
    "shared_gate_up": shared,
    "target_shared_w8a8_fixture_contract_pass": False,
    "remaining_fixture_issue": "Merged shared gate_up_proj is BF16 in this fixture while down_proj is INT8.",
    "rank_zero_report_sha256": hashlib.sha256((output / "rank-00.json").read_bytes()).hexdigest(),
    "actual_api_libraries": resolvers,
}
(output / "acceptance-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
shutil.copy2(__file__, output / "audit.py")
print(json.dumps(summary, indent=2))
