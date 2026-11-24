"""Decode local-layer records using offsets measured from their frozen C++ types."""

import json
from pathlib import Path
import struct
import subprocess
import tempfile


ATTENTION = ("rows pages table_pages blocks scale epsilon_cq epsilon_ckv "
             "s2_base mm_elements bmm_elements query_offset rope_offset "
             "work_offset prolog_bytes sfa_bytes bytes").split()
TAIL = "heads values quantized token_scales accumulators system bytes".split()
FRONT = ("rows capacity blocks quantized query_scale_bf16 query_acc key_acc "
         "query_projection key_projection key_normalized query_rope key_rope "
         "norm_stats query_hadamard key_hadamard system bytes").split()
INDEXER = ("rows pages table_pages blocks quantized mm_bytes merge_offset "
           "merge_bytes params_offset params_bytes weights_offset weights_bytes bytes").split()


def member_names():
    config = []
    for prefix, names in (
            ("indexed.attention", ATTENTION), ("indexed.front", FRONT),
            ("indexed.indexer", INDEXER), ("indexed.tail", TAIL),
            ("indexed", ("query_norm query_norm_scale query key weights query_scale "
                         "key_scale topk latent attention_offset front_offset "
                         "indexer_offset tail_offset bytes").split()),
            ("shared.attention", ATTENTION), ("shared.tail", TAIL),
            ("shared", "capacity reserved arena_offset bytes".split())):
        config.extend(prefix + "." + name for name in names)
    config.extend("input_epsilon ffn_epsilon bias_kind reserved".split())
    addresses = ["norm." + name for name in
                 "input residual gamma normalized retained rstd codes scales".split()]
    addresses.extend("attention." + name for name in "hidden token_x token_scale".split())
    for name, count in (("main_weights", 9), ("front_weights", 6), ("tail_weights", 3)):
        addresses.extend(f"attention.{name}[{i}]" for i in range(count))
    addresses.append("attention.bias")
    for name, count in (("main_metadata", 6), ("indexer_metadata", 6), ("caches", 4)):
        addresses.extend(f"attention.{name}[{i}]" for i in range(count))
    addresses.extend("attention." + name for name in "scratch tiling output".split())
    addresses.extend("shared_topk post_gamma gate correction_bias".split())
    for name, count in (("bridge", 12), ("body.weights", 6), ("body.scratch", 21)):
        addresses.extend(f"{name}[{i}]" for i in range(count))
    addresses.extend("body.dispatch." + name for name in
                     "rows scales inverse counts probabilities active_list bf16_rows expert_ids".split())
    addresses.extend("body.shared." + name for name in
                     ("input input_scale gate_weight gate_scale down_weight down_scale "
                      "payload tiling output capacity partial_offset middle_offset "
                      "scale_offset down_offset").split())
    addresses.extend(("body.rows", "body.blocks"))
    return {"config": config, "addresses": addresses}


def probe_source():
    source = '''#include <cstddef>
#include <iostream>
#include <type_traits>
#include "source/contract.h"
template<class T> void Field(const char* record, const char* name, size_t offset) {
    const char* kind = std::is_floating_point_v<T> ? "f" :
        (std::is_signed_v<T> ? "i" : "u");
    std::cout << record << " " << name << " " << offset << " "
              << sizeof(T) << " " << kind << "\\n";
}
int main() {
    using C = layer_local_chain::Config;
    using A = layer_local_chain::Addresses;
    std::cout << "size config " << sizeof(C) << "\\n";
    std::cout << "size addresses " << sizeof(A) << "\\n";
'''
    for record, fields in member_names().items():
        kind = "C" if record == "config" else "A"
        for name in fields:
            source += (f'    Field<decltype({kind}{{}}.{name})>("{record}", '
                       f'"{name}", offsetof({kind}, {name}));\n')
    return source + "}\n"


def measure_layout(headers):
    with tempfile.TemporaryDirectory(prefix="local-record-abi-") as temporary:
        root = Path(temporary)
        cpp, binary = root / "probe.cpp", root / "probe"
        cpp.write_text(probe_source())
        command = ["g++", "-std=c++17", "-DROUTED_ACTIVE_LIST", "-DROUTED_INPUT_FRONT",
                   "-DROUTED_ROW_BALANCED", "-DROUTED_SHARED_PIPELINE",
                   "-I", str(headers), str(cpp), "-o", str(binary)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    layout = {"config": {"fields": []}, "addresses": {"fields": []}}
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts[0] == "size":
            layout[parts[1]]["bytes"] = int(parts[2])
        else:
            record, name, offset, size, kind = parts
            layout[record]["fields"].append({
                "name": name, "offset": int(offset), "size": int(size), "kind": kind})
    if (layout["config"]["bytes"], layout["addresses"]["bytes"]) != (664, 928):
        raise RuntimeError("The frozen C++ record size changed")
    return layout


def decode(payload, layout):
    if len(payload) != layout["bytes"]:
        raise ValueError("A host record has the wrong byte extent")
    formats = {("u", 4): "I", ("u", 8): "Q", ("i", 4): "i",
               ("i", 8): "q", ("f", 4): "f"}
    result = {}
    for field in layout["fields"]:
        result[field["name"]] = struct.unpack_from(
            "<" + formats[field["kind"], field["size"]], payload, field["offset"])[0]
    return result


def compare_records(control, candidate, control_down, candidate_down, layout):
    if control_down == candidate_down:
        raise RuntimeError("The two Down tilings must have independent owners")
    before, after = bytes(control.addresses), bytes(candidate.addresses)
    expected = "body.scratch[7]"
    field = next(item for item in layout["addresses"]["fields"] if item["name"] == expected)
    if field["size"] != 8 or field["kind"] != "u":
        raise RuntimeError("The Down address field is not uint64")
    first = decode(before, layout["addresses"])
    second = decode(after, layout["addresses"])
    if (first[expected] != control_down or second[expected] != candidate_down
            or [key for key in first if first[key] != second[key]] != [expected]):
        raise RuntimeError("A local-layer address field differs outside Down tiling")
    start = field["offset"]
    if before[:start] + before[start + 8:] != after[:start] + after[start + 8:]:
        raise RuntimeError("A local-layer address record differs outside Down tiling")
    if bytes(control.config) != bytes(candidate.config):
        raise RuntimeError("The two local-layer configurations differ")
    return {"config": decode(bytes(control.config), layout["config"]),
            "addresses": {"control": first, "candidate": second},
            "only_different_field": expected, "byte_offset": start,
            "config_bytes_equal": True, "all_other_address_bytes_equal": True}


def frozen_layout():
    return json.loads(Path(__file__).with_name("record-layout.json").read_text())
