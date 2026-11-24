"""Check retained-index geometry and alias rejection without device access."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from topk_contract import validate_topk


class Owner:
    def __init__(self, rows=12):
        self.device = "npu:0"
        self.dtype = "int32"
        self.shape = (rows, 1, 2048)
        self.address = 0x100000
        self.offset = 0
        self.contiguous = True
        self.format = 2

    def data_ptr(self):
        return self.address

    def untyped_storage(self):
        return self

    def storage_offset(self):
        return self.offset

    def is_contiguous(self):
        return self.contiguous


class ContractTest(unittest.TestCase):
    def setUp(self):
        modules = {"torch": SimpleNamespace(int32="int32"),
                   "torch_npu": SimpleNamespace(
                       get_npu_format=lambda value: value.format)}
        self.mock_runtime = patch.dict(sys.modules, modules)
        self.mock_runtime.start()
        self.addCleanup(self.mock_runtime.stop)

    def test_all_widths(self):
        for rows in (1, 2, 12, 32):
            validate_topk(Owner(rows), rows, "npu:0", [()])

    def test_invalid_owner(self):
        changes = {"device": "npu:1", "dtype": "int64",
                   "shape": (32, 2048), "offset": 2048,
                   "contiguous": False, "format": 29,
                   "address": 0x100004}
        for field, value in changes.items():
            with self.subTest(field=field):
                owner = Owner()
                setattr(owner, field, value)
                with self.assertRaisesRegex(ValueError, "retained aligned"):
                    validate_topk(owner, 12, "npu:0", [()])
        with self.assertRaisesRegex(ValueError, "retained aligned"):
            validate_topk(None, 12, "npu:0", [()])

    def test_alias(self):
        owner = Owner()
        with self.assertRaisesRegex(ValueError, "separate storage"):
            validate_topk(owner, 12, "npu:0", [(None, Owner())])
        other = Owner()
        other.address += 0x100000
        validate_topk(owner, 12, "npu:0", [(None, other)])


if __name__ == "__main__":
    unittest.main()
