"""Retain validated parent tensors for the local-chain comparison harness."""

import ctypes
import hashlib
from pathlib import Path


U64_POINTER = ctypes.POINTER(ctypes.c_uint64)


def pointers(values):
    return tuple(0 if value is None else value.data_ptr() for value in values)


class Prepared:
    """Own both host records and every tensor referenced by the device record."""

    def __init__(self, library, norm, attention, post, routed, bridge, body,
                 shared, shared_topk, geometry, scalars, sync, indexed):
        groups = (norm, attention, post, routed, bridge, body, shared)
        if tuple(map(len, groups)) != (8, 41, 3, 6, 12, 21, 4):
            raise ValueError("Expected norm/attention/post/routed/bridge/body/shared slots 8/41/3/6/12/21/4")
        if len(geometry) != 9 or len(scalars) != 5:
            raise ValueError("Expected nine geometry and five scalar fields")
        if norm[0] is None or norm[0].device.type != "npu":
            raise ValueError("The local-chain input must be an NPU tensor")
        required_attention = (*range(12), *range(18, 21), *range(22, 28),
                              34, 35, 38, 39, 40)
        if indexed:
            required_attention += (*range(12, 18), *range(28, 34), 36)
            if geometry[6]:
                required_attention += (37,)
        if geometry[8]:
            required_attention += (21,)
        if any(attention[index] is None for index in required_attention):
            raise ValueError("A required attention tensor is missing")
        if any(value is None for group in groups if group is not attention
               for value in group):
            raise ValueError("A required norm or FFN tensor is missing")
        from topk_contract import validate_topk
        validate_topk(shared_topk, geometry[0], norm[0].device, groups)
        owners = tuple(value for group in groups for value in group
                       if value is not None)
        if shared_topk is not None:
            owners += (shared_topk,)
        if any(value.device != norm[0].device for value in owners):
            raise ValueError("All retained tensors must belong to one NPU")
        if geometry[0] != norm[0].shape[0] or norm[0].shape[1:] != (6144,):
            raise ValueError("Geometry must describe the retained [M,6144] input")
        self.library = library
        self.owners = owners
        self.owner_pointers = pointers(owners)
        self.device = norm[0].device.index
        self.sync = int(sync)
        self.indexed = bool(indexed)
        self.config = ctypes.create_string_buffer(library.config_bytes)
        self.addresses = ctypes.create_string_buffer(library.address_bytes)
        arrays = [(ctypes.c_uint64 * len(group))(*pointers(group))
                  for group in groups]
        status = library.handle.layer_topk_prepare_records(
            self.config, len(self.config), self.addresses, len(self.addresses),
            (ctypes.c_int64 * 9)(*geometry), (ctypes.c_float * 5)(*scalars),
            arrays[0], arrays[1],
            0 if shared_topk is None else shared_topk.data_ptr(),
            *arrays[2:])
        library.check(status)

    def launch(self):
        if pointers(self.owners) != self.owner_pointers:
            raise RuntimeError("A retained local-chain tensor changed its data address")
        self.library.check(self.library.handle.layer_topk_submit(
            self.device, self.sync, self.config, len(self.config),
            self.addresses, len(self.addresses), self.indexed))


class Library:
    """Load a frozen library after importing the matching TorchNPU runtime."""

    def __init__(self, path, expected_sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError("The local-chain library differs from its build record")
        self.handle = ctypes.CDLL(str(path.resolve()))
        for name in ("layer_topk_config_bytes", "layer_topk_address_bytes"):
            function = getattr(self.handle, name)
            function.argtypes = []
            function.restype = ctypes.c_size_t
        self.config_bytes = self.handle.layer_topk_config_bytes()
        self.address_bytes = self.handle.layer_topk_address_bytes()
        if (self.config_bytes, self.address_bytes) != (664, 928):
            raise RuntimeError("The local-chain record ABI changed")
        self.handle.layer_topk_last_error.argtypes = []
        self.handle.layer_topk_last_error.restype = ctypes.c_char_p
        self.handle.layer_topk_prepare_records.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float),
            U64_POINTER, U64_POINTER, ctypes.c_uint64,
            U64_POINTER, U64_POINTER, U64_POINTER, U64_POINTER, U64_POINTER,
        ]
        self.handle.layer_topk_prepare_records.restype = ctypes.c_int
        self.handle.layer_topk_submit.argtypes = [
            ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool,
        ]
        self.handle.layer_topk_submit.restype = ctypes.c_int

    def check(self, status):
        if status != 0:
            message = self.handle.layer_topk_last_error().decode("utf-8")
            raise RuntimeError(f"Local-chain status {status}: {message}")
