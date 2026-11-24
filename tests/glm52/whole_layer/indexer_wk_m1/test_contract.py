"""Reject operand-layout and producer-identity changes in the retained M1 case."""

import argparse
import copy
import json
from pathlib import Path

import contract


def rejected(function):
    try:
        function()
    except ValueError:
        return True
    raise AssertionError("An invalid captured contract was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", type=Path, required=True)
    args = parser.parse_args()
    if contract.file_identity(args.capture_report)["sha256"] != contract.REPORT_SHA256:
        raise ValueError("Unexpected capture report")
    report = json.loads(args.capture_report.read_text())
    descriptors = contract.capture_descriptors(report)
    checks = {"captured_descriptor_accepted": True}
    changed = copy.deepcopy(report)
    changed["vendor_device_descriptors"]["key_weight"]["format"] = 29
    checks["different_weight_format_rejected"] = rejected(
        lambda: contract.capture_descriptors(changed))
    changed = copy.deepcopy(report)
    changed["live_producer_checks"]["native_key_matches_written_cache"]["bitwise_equal"] = False
    checks["wrong_cache_producer_rejected"] = rejected(
        lambda: contract.capture_descriptors(changed))
    changed = copy.deepcopy(report)
    changed["original_failure_reproduction"]["pass"] = False
    checks["different_failure_rejected"] = rejected(
        lambda: contract.capture_descriptors(changed))
    changed = dict(descriptors["key_weight"], storage_offset=1)
    checks["device_offset_change_rejected"] = rejected(
        lambda: contract.require_device_descriptor(changed, descriptors["key_weight"]))
    changed = dict(descriptors["key_weight"], data_ptr=1, storage_ptr=1)
    contract.require_device_descriptor(changed, descriptors["key_weight"])
    checks["fresh_allocation_address_accepted"] = True
    print(json.dumps({"checks": checks, "pass": all(checks.values())}, indent=2))


if __name__ == "__main__":
    main()
