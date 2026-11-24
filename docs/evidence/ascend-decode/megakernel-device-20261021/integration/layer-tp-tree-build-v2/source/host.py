"""Keep TP communication windows and layer tensors alive through graph replay."""

import ctypes
import hashlib
from pathlib import Path


class Library:
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


class Prepared:
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
