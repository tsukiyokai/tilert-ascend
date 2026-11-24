#!/usr/bin/env python3
"""Decode retained CANN DumpData metadata and the A8W4 tiling prefix."""

import argparse
import hashlib
import json
from pathlib import Path
import struct


BASE_FIELDS = (
    "groupNum", "coreNum", "K", "N", "M", "baseM", "baseN", "mLimit",
    "workSpaceOffset1", "workSpaceOffset2", "quantGroupNum", "isSingleTensor",
    "groupListType", "smoothScaleDimNum", "singleN",
)
SWIGLU_FIELDS = ("maxProcessRowNum", "groupListLen", "tokenLen")
CUBE_FIELDS = (
    "usedCoreNum", "M", "N", "Ka", "Kb", "singleCoreM", "singleCoreN",
    "singleCoreK", "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
    "stepN", "isBias", "transLength", "iterateOrder", "shareMode", "shareL1Size",
    "shareL0CSize", "shareUbSize", "batchM", "batchN", "singleBatchM",
    "singleBatchN", "stepKa", "stepKb", "depthAL1CacheUB", "depthBL1CacheUB",
    "dbL0A", "dbL0B", "dbL0C", "ALayoutInfoB", "ALayoutInfoS", "ALayoutInfoN",
    "ALayoutInfoG", "ALayoutInfoD", "BLayoutInfoB", "BLayoutInfoS", "BLayoutInfoN",
    "BLayoutInfoG", "BLayoutInfoD", "CLayoutInfoB", "CLayoutInfoS1", "CLayoutInfoN",
    "CLayoutInfoG", "CLayoutInfoS2", "BatchNum", "mxTypePara",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def varint(data, offset):
    result = 0
    for shift in range(0, 70, 7):
        value = data[offset]
        offset += 1
        result |= (value & 127) << shift
        if value < 128:
            return result, offset
    raise ValueError("Malformed protobuf varint")


def fields(data):
    result = {}
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire == 2:
            size, offset = varint(data, offset)
            value = data[offset:offset + size]
            offset += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        if offset > len(data):
            raise ValueError("Truncated protobuf field")
        result.setdefault(field, []).append(value)
    return result


def shape(data):
    result = []
    for value in fields(data).get(1, []):
        if isinstance(value, int):
            result.append(value)
        else:
            offset = 0
            while offset < len(value):
                dimension, offset = varint(value, offset)
                result.append(dimension)
    return result


def tensor(data, output=False):
    values = fields(data)
    result = {
        "arg_index": values.get(12 if output else 10, [0])[0],
        "bytes": values.get(6 if output else 5, [0])[0],
        "data_type_enum": values.get(1, [0])[0],
        "format_enum": values.get(2, [0])[0],
        "shape": shape(values[3][0]) if 3 in values else [],
    }
    if not output:
        result["input_type"] = values.get(11, [0])[0]
    return result


def decode(path):
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        if header_size > 1 << 20:
            raise ValueError("Unexpected DumpData header length")
        header = fields(source.read(header_size))
        result = {
            "path": str(path), "sha256": digest(path),
            "file_bytes": path.stat().st_size, "header_bytes": header_size,
            "version": header[1][0].decode(),
            "inputs": [tensor(value) for value in header.get(4, [])],
            "outputs": [tensor(value, True) for value in header.get(3, [])],
            "spaces": [
                {"arg_index": fields(value).get(4, [0])[0],
                 "bytes": fields(value).get(3, [0])[0]}
                for value in header.get(8, [])
            ],
            "dfx_message": header.get(9, [b""])[0].decode(),
            "dtype_note": "DumpFile deliberately writes DT_UNDEFINED; use the retained kernel JSON for actual dtypes.",
        }
        offset = 8 + header_size
        for item in result["inputs"]:
            item["payload_offset"] = offset
            if item.get("input_type") == 7:
                source.seek(offset)
                payload = source.read(item["bytes"])
                if len(payload) != 352:
                    raise ValueError("This decoder requires the retained 352-byte A8W4 tiling")
                result["tiling"] = {
                    "offset": offset, "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "hex": payload.hex(),
                    "base": dict(zip(BASE_FIELDS, struct.unpack("<15q", payload[:120]))),
                    "swiglu": dict(zip(SWIGLU_FIELDS, struct.unpack("<3q", payload[120:144]))),
                    "cube": dict(zip(CUBE_FIELDS, struct.unpack("<50i", payload[144:344]))),
                    "uninterpreted_trailer_hex": payload[344:].hex(),
                    "schema_scope": "Base 120-byte common source prefix; Swiglu and Cube layout cross-checked against source and installed SDK. Final 8 bytes are not interpreted.",
                }
            offset += item["bytes"]
        for collection in (result["outputs"], result["spaces"]):
            for item in collection:
                item["payload_offset"] = offset
                offset += item["bytes"]
        if offset != result["file_bytes"]:
            raise ValueError(f"Payload lengths do not cover the file: {offset}")
        result["payload_lengths_exact"] = True
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(decode(args.dump), indent=2) + "\n")


if __name__ == "__main__":
    main()
