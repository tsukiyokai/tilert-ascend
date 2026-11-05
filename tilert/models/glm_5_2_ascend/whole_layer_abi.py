"""Own the frozen layer records and stream-bound SHMEM windows.

The record layouts are checked by the compiled metadata builder. Tensors remain
owned until every graph using them has been reset and the stream synchronized.
"""

import ctypes
import hashlib
from pathlib import Path


U64_POINTER = ctypes.POINTER(ctypes.c_uint64)


def pointers(values):
    return tuple(0 if value is None else value.data_ptr() for value in values)


class LocalRecords:
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
        if not indexed and shared_topk is None:
            raise ValueError("Shared-indexer attention requires its retained top-k owner")
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
        status = library.handle.layer_local_prepare_records(
            self.config, len(self.config), self.addresses, len(self.addresses),
            (ctypes.c_int64 * 9)(*geometry), (ctypes.c_float * 5)(*scalars),
            arrays[0], arrays[1],
            0 if shared_topk is None else shared_topk.data_ptr(),
            *arrays[2:])
        library.check(status)

    def launch(self):
        if pointers(self.owners) != self.owner_pointers:
            raise RuntimeError("A retained local-chain tensor changed its data address")
        self.library.check(self.library.handle.layer_local_submit(
            self.device, self.sync, self.config, len(self.config),
            self.addresses, len(self.addresses), self.indexed))


class MetadataLibrary:
    """Load a frozen library after importing the matching TorchNPU runtime."""

    def __init__(self, path, expected_sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError("The local-chain library differs from its build record")
        self.handle = ctypes.CDLL(str(path.resolve()))
        for name in ("layer_local_config_bytes", "layer_local_address_bytes"):
            function = getattr(self.handle, name)
            function.argtypes = []
            function.restype = ctypes.c_size_t
        self.config_bytes = self.handle.layer_local_config_bytes()
        self.address_bytes = self.handle.layer_local_address_bytes()
        if (self.config_bytes, self.address_bytes) != (664, 928):
            raise RuntimeError("The local-chain record ABI changed")
        self.handle.layer_local_last_error.argtypes = []
        self.handle.layer_local_last_error.restype = ctypes.c_char_p
        self.handle.layer_local_prepare_records.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float),
            U64_POINTER, U64_POINTER, ctypes.c_uint64,
            U64_POINTER, U64_POINTER, U64_POINTER, U64_POINTER, U64_POINTER,
        ]
        self.handle.layer_local_prepare_records.restype = ctypes.c_int
        self.handle.layer_local_submit.argtypes = [
            ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool,
        ]
        self.handle.layer_local_submit.restype = ctypes.c_int

    def check(self, status):
        if status != 0:
            message = self.handle.layer_local_last_error().decode("utf-8")
            raise RuntimeError(f"Local-chain status {status}: {message}")


class TreeLibrary:
    """Own one explicitly initialized SHMEM context in each rank process."""

    def __init__(self, path, expected_sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError("The TP layer library differs from its build record")
        self.handle = ctypes.CDLL(str(path.resolve()))
        signatures = {
            "record_bytes": ([ctypes.c_int], ctypes.c_size_t),
            "last_error": ([], ctypes.c_char_p),
            "initialize": ([ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_char_p, ctypes.c_uint64], ctypes.c_int),
            "window_create": ([ctypes.c_int64, ctypes.POINTER(ctypes.c_uint64)],
                              ctypes.c_int),
            "window_info": ([ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                             ctypes.c_size_t], ctypes.c_int),
            "window_destroy": ([ctypes.c_uint64], ctypes.c_int),
            "finalize": ([], ctypes.c_int),
            "submit": ([ctypes.c_int, ctypes.c_uint64,
                        ctypes.c_void_p, ctypes.c_size_t,
                        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool,
                        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.handle, "layer_tp_tree_" + name)
            function.argtypes = arguments
            function.restype = result
        sizes = tuple(self.handle.layer_tp_tree_record_bytes(index) for index in range(3))
        if sizes != (664, 928, 48):
            raise RuntimeError(f"The TP layer record ABI changed: {sizes}")

    def check(self, status):
        if status != 0:
            message = self.handle.layer_tp_tree_last_error().decode("utf-8")
            raise RuntimeError(f"TP layer status {status}: {message}")

    def initialize(self, device, rank, ranks, endpoint, heap_bytes):
        self.check(self.handle.layer_tp_tree_initialize(
            device, rank, ranks, endpoint.encode("utf-8"), heap_bytes))

    def finalize(self):
        self.check(self.handle.layer_tp_tree_finalize())


class Window:
    """Allocate collectively, on the stream that will capture every layer call."""

    def __init__(self, library, capacity):
        self.library = library
        handle = ctypes.c_uint64()
        library.check(library.handle.layer_tp_tree_window_create(capacity, ctypes.byref(handle)))
        self.handle = handle.value
        fields = (ctypes.c_uint64 * 6)()
        library.check(library.handle.layer_tp_tree_window_info(self.handle, fields, len(fields)))
        self.address, self.bytes, self.capacity, self.rank, self.ranks, self.device = fields

    def close(self):
        """Call after destroying all graphs and synchronizing all ranks."""
        self.library.check(self.library.handle.layer_tp_tree_window_destroy(self.handle))
        self.handle = 0


class PreparedLayer:
    """Extend a validated local-layer owner with distinct BF16 reduction outputs."""

    def __init__(self, local, window, attention_sum, output):
        import torch
        import torch_npu

        source = local.owners[0]
        outputs = (attention_sum, output)
        if window.handle == 0 or local.device != window.device:
            raise ValueError("The layer and live communication window must share a device")
        if source.shape[0] > window.capacity:
            raise ValueError("The communication window is smaller than the layer")
        for value in outputs:
            if (value.dtype != torch.bfloat16 or value.device != source.device
                    or value.shape != source.shape or not value.is_contiguous()
                    or torch_npu.get_npu_format(value) != 2):
                raise ValueError("Reduction outputs must be contiguous physical-ND BF16 [M,6144]")
        output_storage = tuple(value.untyped_storage().data_ptr() for value in outputs)
        local_storage = {value.untyped_storage().data_ptr() for value in local.owners}
        if (len(set(output_storage)) != 2
                or any(pointer in local_storage for pointer in output_storage)):
            raise ValueError("Reduction outputs need storage distinct from local-layer tensors")
        self.output_pointers = tuple(value.data_ptr() for value in outputs)
        self.local = local
        self.window = window
        self.outputs = outputs

    def launch(self):
        local = self.local
        if self.window.handle == 0:
            raise RuntimeError("The layer communication window was released")
        if tuple(value.data_ptr() for value in local.owners) != local.owner_pointers:
            raise RuntimeError("A retained local-layer tensor changed its data address")
        if tuple(value.data_ptr() for value in self.outputs) != self.output_pointers:
            raise RuntimeError("A retained reduction output changed its data address")
        library = self.window.library
        library.check(library.handle.layer_tp_tree_submit(
            local.device, local.sync, local.config, len(local.config),
            local.addresses, len(local.addresses), local.indexed,
            self.window.handle, *self.output_pointers))
