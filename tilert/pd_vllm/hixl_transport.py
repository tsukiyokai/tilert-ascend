"""HIXL writes from Torch-NPU allocations on one rank."""

import ctypes
import ipaddress
import os
from pathlib import Path
import threading

from .transport import Transport


class HixlTransport(Transport):
    """Keep registered allocations alive until close after all peers drain.

    TILERT_HIXL_PORT must identify this rank's unique listening endpoint.
    Import and init do not load HIXL; the first register binds its device.
    The caller owns ACL initialization and the lifetime of registered memory.
    Use non-expandable Torch-NPU allocations; IPC registers their full blocks.
    """

    name = "hixl"

    def __init__(self, library_path=None, port=None, timeout_ms=30000):
        self._library_path = Path(
            library_path or os.environ.get("TILERT_HIXL_LIBRARY")
            or Path(__file__).with_name("libtilert_hixl.so")
        )
        self._port = port
        self._timeout_ms = int(timeout_ms)
        if self._timeout_ms <= 0 or self._timeout_ms > 2147483647:
            raise ValueError("HIXL timeout must fit a positive int32")
        self._endpoint = None
        self._device_id = None
        self._library = None
        self._handle = ctypes.c_void_p()
        self._regions = []
        self._allocations = []
        self._sealed = False
        self._lock = threading.RLock()

    def init(self, host: str) -> None:
        with self._lock:
            if self._endpoint is not None:
                raise RuntimeError("HIXL transport is already initialized")
            port = (self._port if self._port is not None
                    else os.environ.get("TILERT_HIXL_PORT"))
            if port is None or not 1 <= int(port) <= 65535:
                raise ValueError("Set a unique TILERT_HIXL_PORT for each rank")
            address = ipaddress.ip_address(host.strip("[]"))
            host = f"[{address}]" if address.version == 6 else str(address)
            self._endpoint = f"{host}:{int(port)}"

    def _open(self, device_id):
        if self._endpoint is None:
            raise RuntimeError("Call init before registering memory")
        if self._handle.value:
            if device_id != self._device_id:
                raise ValueError("One HIXL transport supports one NPU device")
            return
        library = ctypes.CDLL(str(self._library_path))
        address = ctypes.POINTER(ctypes.c_uint64)
        signatures = {
            "open": [ctypes.c_char_p, ctypes.c_int,
                     ctypes.POINTER(ctypes.c_void_p)],
            "register": [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64],
            "write": [ctypes.c_void_p, ctypes.c_char_p, address, address,
                      address, ctypes.c_size_t, ctypes.c_int],
            "close": [ctypes.c_void_p],
        }
        for name, arguments in signatures.items():
            function = getattr(library, "tilert_hixl_" + name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        library.tilert_hixl_last_error.argtypes = []
        library.tilert_hixl_last_error.restype = ctypes.c_char_p
        self._library = library
        self._call("open", self._endpoint.encode(), device_id,
                   ctypes.byref(self._handle))
        self._device_id = device_id

    def _call(self, operation, *arguments):
        result = getattr(self._library, "tilert_hixl_" + operation)(*arguments)
        if result:
            message = self._library.tilert_hixl_last_error()
            raise RuntimeError(f"HIXL {operation}: {message.decode()}")

    def register(self, ptr, nbytes, dev_id, location=None, host=False):
        if host or location is not None:
            raise ValueError("HIXL requires MEM_DEVICE and no rail override")
        ptr, nbytes, dev_id = int(ptr), int(nbytes), int(dev_id)
        self._validate_interval(ptr, nbytes)
        if not 0 <= dev_id <= 2147483647:
            raise ValueError("NPU device id must fit a nonnegative int32")
        with self._lock:
            region = {"base": ptr, "bytes": nbytes}
            if self._sealed and region not in self._regions:
                raise RuntimeError(
                    "Register all regions before publishing metadata"
                )
            for current in self._regions:
                if (current != region
                        and ptr < current["base"] + current["bytes"]
                        and current["base"] < ptr + nbytes):
                    raise ValueError("Overlapping HIXL memory regions")
            self._open(dev_id)
            allocation = self._allocation(ptr, nbytes)
            if allocation is None:
                import torch

                self._allocations = [
                    (item["address"], item["total_size"],
                     item["is_expandable"])
                    for item in torch.npu.memory_snapshot()
                    if item["device"] == dev_id
                ]
                allocation = self._allocation(ptr, nbytes)
            if allocation is None:
                raise ValueError(
                    "HIXL memory is outside Torch-NPU allocations"
                )
            base, size, expandable = allocation
            if expandable:
                raise ValueError(
                    "HIXL requires PYTORCH_NPU_ALLOC_CONF="
                    "expandable_segments:False"
                )
            self._call("register", self._handle, base, size)
            if region not in self._regions:
                self._regions.append(region)

    def _allocation(self, address, length):
        return next((item for item in self._allocations
                     if item[0] <= address
                     and address + length <= item[0] + item[1]), None)

    def local_meta(self) -> dict:
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("Register device memory before local_meta")
            self._sealed = True
            return {
                "hixl_engine": self._endpoint,
                "hixl_device": self._device_id,
                "hixl_regions": [dict(region) for region in self._regions],
            }

    @staticmethod
    def _validate_interval(address, length):
        if address <= 0 or length <= 0 or address + length >= 2 ** 64:
            raise ValueError("Memory interval exceeds uint64 addresses")

    def write(self, remote_meta, srcs, dsts, lens) -> None:
        srcs, dsts, lens = ([int(value) for value in items]
                            for items in (srcs, dsts, lens))
        if not len(srcs) == len(dsts) == len(lens):
            raise ValueError("Source, destination and length counts differ")
        remote_regions = remote_meta["hixl_regions"]
        for source, destination, length in zip(srcs, dsts, lens):
            self._validate_interval(source, length)
            self._validate_interval(destination, length)
            if not any(
                region["base"] <= source
                and source + length <= region["base"] + region["bytes"]
                for region in self._regions
            ):
                raise ValueError("Source exceeds registered local memory")
            if not any(
                int(region["base"]) <= destination
                and destination + length <= (
                    int(region["base"]) + int(region["bytes"])
                )
                for region in remote_regions
            ):
                raise ValueError("Destination exceeds registered peer memory")
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("Register memory before writing")
            self._sealed = True
            array = ctypes.c_uint64 * len(lens)
            self._call(
                "write", self._handle, remote_meta["hixl_engine"].encode(),
                array(*srcs), array(*dsts), array(*lens), len(lens),
                self._timeout_ms,
            )

    def close(self) -> None:
        """Release registrations after peer writes and consumers drain."""
        with self._lock:
            if self._handle.value:
                self._call("close", self._handle)
                self._handle = ctypes.c_void_p()
                self._regions.clear()
