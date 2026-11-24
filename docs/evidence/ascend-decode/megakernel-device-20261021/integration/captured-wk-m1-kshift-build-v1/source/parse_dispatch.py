#!/usr/bin/env python3
"""Decode the captured CANN 9.2 M1 WK MatMulV2 dispatch payload."""

import argparse
import json
from pathlib import Path
import struct

from contract import file_identity
from kshift_check import outer_order


FIELDS = (
    "k_ori m_ori n_ori k m n batch_single_core m_single_core n_single_core "
    "batch_dim n_dim m_dim k_dim m_al1 n_bl1 cub_n1 m_l0 k_l0 n_ub_l0_time "
    "kal0_factor kbl0_factor kal1_factor kbl1_factor kal1_16 kbl1_16 "
    "kl1_times batch_l1_factor batch_ub_l0_time batch_cub out_branch_flag "
    "bias_flag hf32_flag datatype_bf16 al1_db bl1_db l0c_db l2_cache_flag "
    "close_k_shift"
).split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plog", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = args.plog.read_text().splitlines()
    matches = [(number, line) for number, line in enumerate(lines, 1)
               if "PrintTilingData[0]" in line and "MatMulCommon" in line]
    if len(matches) != 1:
        raise ValueError("Expected one production MatMulCommon tiling payload")
    line_number, payload = matches[0]
    words = [int(value) for value in payload.split("data(uint32_t):", 1)[1].split()]
    if len(words) != 40 or words[38:] != [0, 0xa5a5a5a5]:
        raise ValueError("Expected 152 B tiling and the separate eight-byte DFX suffix")
    fields = dict(zip(FIELDS, words[:38], strict=True))
    expected = {"m_ori": 1, "n_ori": 160, "k_ori": 6144,
                "m_dim": 1, "n_dim": 10, "k_dim": 1, "k_l0": 8,
                "kal1_factor": 12, "kbl1_factor": 12,
                "kal1_16": 32, "kbl1_16": 32,
                "datatype_bf16": 1, "close_k_shift": 0}
    if any(fields[name] != value for name, value in expected.items()):
        raise ValueError("The observed M1 dispatch changed")
    relevant = [(number, line) for number, line in enumerate(lines, 1)
                if (("MatMulCommon" in line and any(token in line for token in (
                    "PrintAICErrorDFXInfo", "PrintTilingData", "Launch kernel engine")))
                    or "kernel_name=MatMulV2_ND_ND_FP16_FP16_false_true_all_98513" in line
                    or ("OpName:[MatMulV2]" in line and any(token in line for token in (
                        "98513", "GetCloseKShiftFlag", "tiling data size")))
                    or '"pad_flag: 0, nz_flag: 0' in line)]
    launches = [line for _, line in relevant if "StreamLaunchKernelV2: kernel info" in line]
    if len(launches) != 1 or "coreDim=10" not in launches[0]:
        raise ValueError("Expected one ten-core key-98513 launch")
    production = json.loads(args.report.read_text())
    if not (production.get("completed")
            and production.get("reproduction", {}).get("bitwise_equal")):
        raise ValueError("The production F.linear replay did not pass")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "tiling-152.bin").write_bytes(struct.pack("<38I", *words[:38]))
    (args.output / "plog-excerpts.txt").write_text(
        "\n".join(f"{number}: {line}" for number, line in relevant) + "\n")
    report = {
        "scope": "Actual single F.linear BF16[1,6144] x ND BF16[160,6144], CANN 9.2",
        "parser": file_identity(__file__), "plog": file_identity(args.plog),
        "production_report": file_identity(args.report), "tiling_line": line_number,
        "tiling_bytes": 152, "tiling_uint32": words[:38],
        "dfx_suffix_uint32": words[38:], "fields": fields,
        "tiling_key": 98513, "num_blocks": 10,
        "kernel": "MatMulV2_ND_ND_FP16_FP16_false_true_all_98513",
        "derived": {"mmad_K": fields["k_l0"] * 16,
                    "outer_K": fields["kal1_16"] * 16,
                    "outer_count": fields["kal1_factor"],
                    "rotation_per_pair": 12 // (10 // 2),
                    "formula": "q(i,j)=((j%2 ? 11-i : i)+2*floor(j/2))%12",
                    "orders": {str(j): outer_order(j, True) for j in range(10)}},
        "witness_mapping": [{"coordinate": [0, 25], "core": 1, "N_tile": 1}],
        "numerical_scope": "The stock replay is measured; the derived native K-order remains a device candidate.",
    }
    report["tiling_binary"] = file_identity(args.output / "tiling-152.bin")
    (args.output / "dispatch.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
