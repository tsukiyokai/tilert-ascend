#!/usr/bin/env python3
"""Extract static Torch schema evidence without importing a TileRT library.

Run on the CPU side of the research server. Requires Python 3.10+, nm,
readelf, and an objdump that understands the wheel's ELF architecture.
The four GLM5 constant-copy recipes are scoped to one SHA256; other
libraries receive only the complete-schema extraction.
"""

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import re
import struct
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


GLM5_SHA256 = "c09053e372a379e7575515e3be2ce84414b52fd082f3698b4e260464d4d5d632"
SCHEMA_PATTERN = re.compile(
    r"^([A-Za-z_]\w*)\(.*\)\s*->\s*"
    r"(?:\(\)|Tensor(?:\[\])?|str\[\]|int|bool|float)$",
    re.DOTALL,
)
PRINTABLE_PATTERN = re.compile(rb"[\x09\x0a\x0d\x20-\x7e]{5,}")
REFERENCE_PATTERN = re.compile(r"^\s*([0-9a-f]+):.*# ([0-9a-f]+)")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True)


def elf_sections(data):
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    if header[0][:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected a little-endian ELF64 file")
    offset, entry_size, count, names_index = (
        header[6], header[11], header[12], header[13]
    )
    entries = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * entry_size)
        for i in range(count)
    ]
    names = entries[names_index]
    name_data = data[names[4]:names[4] + names[5]]
    sections = {}
    for entry in entries:
        start = entry[0]
        name = name_data[start:name_data.index(b"\0", start)].decode()
        sections[name] = {
            "address": entry[3], "offset": entry[4], "size": entry[5]
        }
    return header[2], sections


def complete_schemas(data, section):
    offset = section["offset"]
    content = data[offset:offset + section["size"]]
    schemas = []
    for match in PRINTABLE_PATTERN.finditer(content):
        schema = " ".join(match.group().decode().split())
        parsed = SCHEMA_PATTERN.fullmatch(schema)
        if parsed is None or schema.count(" -> ") != 1:
            continue
        schemas.append({
            "name": parsed[1],
            "schema": schema,
            "file_offset": offset + match.start(),
            "virtual_address": section["address"] + match.start(),
            "recovery": "complete_rodata_string",
        })
    return schemas


def assembled_glm5_schemas(data):
    """Recover constant stores immediately before parseSchema/_def calls."""
    if sha256(data) != GLM5_SHA256:
        return []
    recipes = [
        (0x124FB0, 96, b"[]", 0x9CF10, 0x9CFDA, 0x9D00B),
        (0x125010, 80, b"", 0x9D585, 0x9D612, 0x9D640),
        (0x125060, 96, b"]", 0xA4900, 0xA49C5, 0xA49F6),
        (0x1250C0, 64, None, 0xA4F75, 0xA5002, 0xA5030),
    ]
    schemas = []
    for offset, length, suffix, start, parse_call, def_call in recipes:
        raw = bytearray(data[offset:offset + length])
        if suffix is None:
            # The last 16-byte store starts at destination +0x3f,
            # overlapping the fourth block by one byte.
            raw[63:79] = data[0x125050:0x125060]
        else:
            raw.extend(suffix)
        schema = raw.decode()
        parsed = SCHEMA_PATTERN.fullmatch(schema)
        if parsed is None:
            raise ValueError(f"Invalid recovered schema at {offset:#x}")
        schemas.append({
            "name": parsed[1], "schema": schema,
            "file_offset": offset, "virtual_address": offset,
            "recovery": "constant_copy_and_immediate_stores",
            "construction_start": start,
            "parse_schema_call": parse_call,
            "torch_library_def_call": def_call,
        })
    return schemas


def wrapper_references(wheel):
    references = {}
    dynamic = []
    for name in sorted(wheel.namelist()):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(wheel.read(name), filename=name)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and ast.unparse(node.value) == "torch.ops.tilert"):
                references.setdefault(node.attr, []).append({
                    "member": name, "line": node.lineno,
                    "kind": "torch_ops_attribute",
                })
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)
                  and node.func.id == "getattr" and len(node.args) >= 2
                  and ast.unparse(node.args[0]) == "torch.ops.tilert"):
                dynamic.append({
                    "member": name, "line": node.lineno,
                    "expression": ast.unparse(node.args[1]),
                })
    return references, dynamic


def verify_record(wheel):
    record = next(n for n in wheel.namelist() if n.endswith(".dist-info/RECORD"))
    checked = 0
    for name, digest, size in csv.reader(io.StringIO(wheel.read(record).decode())):
        if not digest:
            continue
        algorithm, expected = digest.split("=", 1)
        if algorithm != "sha256":
            raise ValueError(f"Unexpected RECORD algorithm: {algorithm}")
        data = wheel.read(name)
        actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        if actual.decode().rstrip("=") != expected or len(data) != int(size):
            raise ValueError(f"RECORD mismatch: {name}")
        checked += 1
    return checked


def inspect_variant(member, data, scratch, output, references):
    path = scratch / Path(member).name
    path.write_bytes(data)
    machine, sections = elf_sections(data)
    schemas = complete_schemas(data, sections[".rodata"])
    schemas.extend(assembled_glm5_schemas(data))
    schemas.sort(key=lambda item: item["name"])
    disassembly = command("objdump", "-d", "-C", "-M", "intel",
                          "--no-show-raw-insn", str(path))
    lines = disassembly.splitlines()
    xrefs = {}
    for line in lines:
        match = REFERENCE_PATTERN.match(line)
        if match:
            xrefs.setdefault(int(match[2], 16), []).append({
                "instruction": int(match[1], 16), "text": line.strip()
            })
    for schema in schemas:
        schema["code_references"] = xrefs.get(schema["virtual_address"], [])
        schema["python_references"] = references.get(schema["name"], [])
        schema["declared_mutated_arguments"] = re.findall(
            r"Tensor\([^)]*!\)(?:\[\])?\??\s+(\w+)", schema["schema"]
        )
    constraints = []
    if Path(member).name == "libtilert_glm52_rocm.so":
        markers = (
            "num_mtp must be", "npes must be 8", "glm52 MoE: S must",
            "glm5_sparse_select_s0123: samples must", "draft_tokens and",
            "x_out must not alias", "sym must be a contiguous device int64",
            "must be a u8 [batch, max_seq_len, 528]", "fp8 ki cache slot",
        )
        section = sections[".rodata"]
        start = section["offset"]
        for match in PRINTABLE_PATTERN.finditer(data[start:start + section["size"]]):
            value = " ".join(match.group().decode().split())
            if any(marker in value for marker in markers):
                address = section["address"] + match.start()
                constraints.append({
                    "text": value, "virtual_address": address,
                    "code_references": xrefs.get(address, []),
                })
    stem = Path(member).stem
    nm_defined = command("nm", "-D", "--defined-only", "-C", str(path))
    note_options = ("-n",) if ".note.gnu.build-id" in sections else ()
    elf_text = "\n".join([
        command("readelf", "-h", "-d", *note_options, str(path)),
        "Dynamic symbols (undefined imports are not Torch operators):",
        command("nm", "-D", "-C", str(path)),
        "Defined dynamic symbols:", nm_defined or "(none)\n",
    ])
    (output / f"{stem}-elf.txt").write_text(elf_text)
    with (output / f"{stem}-registration.txt").open("w") as stream:
        for schema in schemas:
            stream.write(f"{schema['name']}\n")
            stream.write(f"schema VA {schema['virtual_address']:#x}\n")
            for reference in schema["code_references"]:
                stream.write(reference["text"] + "\n")
            if "construction_start" in schema:
                start = schema["construction_start"]
                end = schema["torch_library_def_call"]
                for line in lines:
                    match = re.match(r"\s*([0-9a-f]+):", line)
                    if match and start <= int(match[1], 16) <= end:
                        stream.write(line + "\n")
            stream.write("\n")
    return {
        "member": member, "size_bytes": len(data), "sha256": sha256(data),
        "elf_machine": machine, "defined_dynamic_symbols": nm_defined.splitlines(),
        "schema_count": len(schemas), "schemas": schemas,
        "selected_native_constraint_strings": constraints,
    }


def write_inventory(wheel_path, output):
    output.mkdir(parents=True, exist_ok=True)
    wheel_data = wheel_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(wheel_data)) as wheel:
        metadata_name = next(
            n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(wheel.read(metadata_name))
        wheel_info_name = metadata_name.removesuffix("METADATA") + "WHEEL"
        references, dynamic = wrapper_references(wheel)
        with tempfile.TemporaryDirectory(prefix="tilert-abi-") as directory:
            variants = [
                inspect_variant(member, wheel.read(member), Path(directory),
                                output, references)
                for member in sorted(wheel.namelist()) if member.endswith(".so")
            ]
        union = {}
        for variant in variants:
            for schema in variant["schemas"]:
                union.setdefault(schema["name"], []).append(variant["member"])
        inventory = {
            "method": "static ELF and wheel Python inspection; no dlopen or execution",
            "wheel": {"filename": wheel_path.name, "sha256": sha256(wheel_data),
                      "size_bytes": len(wheel_data), "name": metadata["Name"],
                      "version": metadata["Version"],
                      "wheel_metadata": wheel.read(wheel_info_name).decode(),
                      "requires_dist": metadata.get_all("Requires-Dist"),
                      "record_members_verified": verify_record(wheel)},
            "unique_operator_name_count": len(union),
            "total_variant_schema_entries": sum(v["schema_count"] for v in variants),
            "operators_by_variant": union,
            "variants": variants,
            "dynamic_python_lookups": dynamic,
            "static_python_attributes_without_any_schema": sorted(set(references) - set(union)),
        }
        (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
        with (output / "schemas.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["variant", "operator", "schema", "file_offset",
                             "virtual_address", "recovery"])
            for variant in variants:
                for schema in variant["schemas"]:
                    writer.writerow([Path(variant["member"]).name, schema["name"],
                                     schema["schema"], hex(schema["file_offset"]),
                                     hex(schema["virtual_address"]), schema["recovery"]])
    print(json.dumps({
        "wheel_sha256": inventory["wheel"]["sha256"],
        "variant_counts": {v["member"]: v["schema_count"] for v in variants},
        "unique_operator_names": len(union),
        "record_members_verified": inventory["wheel"]["record_members_verified"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_inventory(args.wheel, args.output)


if __name__ == "__main__":
    main()
