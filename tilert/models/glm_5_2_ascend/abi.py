import ctypes
from dataclasses import dataclass
import gc
import hashlib
import json
from operator import index as _index
import os
from pathlib import Path
import struct

from .config import (
    FINAL_POS,
    FULL_INDEX_LAYERS,
    HIDDEN_SIZE,
    INT8_INDEX_LAYERS,
    MAIN_LAYERS,
    MTP_PREFIX_LEN,
    PAGE_SIZE,
    PREFIX_LEN,
    TOTAL_LAYERS,
)
from .ops.build_megakernel import (
    R4_FLAGS,
    ROOT as _MK_ROOT,
    effective_source_digest,
    source_digest,
)

U64_POINTER = ctypes.POINTER(ctypes.c_uint64)
FFN_NAMESPACE = "tilert_ffn_original_shared_gu_static"
LAYOUT_NAME = "glm5_2_ascend_w4a8c8"
LAYOUT_VERSION = 2
PARENT_HIDDEN_LAYOUT_VERSION = 3
PARENT_HIDDEN_REGION = "mtp_parent_hidden"
PARENT_HIDDEN_BYTES = HIDDEN_SIZE * 2
NUM_RANKS = 16
NUM_LAYERS = TOTAL_LAYERS

LIBRARY_HASHES = {
    "indexed": {
        "fea3c0c6c8d1c1daa97dddc1c954ca1f61ffc1a8f579aeccb11a078bb211f403",
        "8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f",
        "aa10bf55d65f34a9aa6d1627703794303dc7b1ce1e3119856ce6a6abf3b9d6f9",
    },
    "shared": {
        "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c",
        "0e985f72fda85389bcc6c11915cef5255a71475633089a82bea8639fd59e1d5d",
    },
    "ffn": {
        "8c031c70c2ccef93a7425ef06ff457923e139889903caaeea21cdbe4ed21cf42",
        "eb565d83aceb23dc00be837042dfb009d0cbc58c9dff531dfec3df34bfb969ac",
        "358034c6b45ad37a270944b0918ceb91084d25e4ed1621a1745a3edbf7400ed9",
        "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2",
        "9a22fae0d4c2aedde083b2d413d6706e33ce5b941508b7d7b7a86814acd611f4",
        "168f1c812bfc9a0a3774d6436004a7d416e18aa17efb18f63e631d6005af9a54",
        "74838084ddcc204cbb01e62d2d612716dc23e3f5e0a6b33232470674e155a199",
        "8ec2d71bc522efbf287bf70cb24bccc6e0c4b1c1556a0335d5529232c4676b0f",
    },
    "metadata": {
        "421775a060c02bbb112d94f17601522d10606d8b506476d900bb03e438e6d5c3",
        "cb5aaf655bab06b7b5e2304acb5a053030de121df4ba8f8735354bd6e0e52708",
        "3349e9978c38a9f34b77895325edda331d165d4be2aaaad09edf2f87f097370f",
        "c8f5cf5a4b54eaffe7d21fafd796e3431a4dc0de16ede4e624d1cd39cfc7cf5e",
        "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f",
        "d9083176f84ca43205b0ec46206d5ee4a3f37c94557baa96cd4e8df76f5c2e66",
        "b8eadb136b308801921430fc1760f2c45a4f4c6ba5b93de54a846bcbb26bfea0",
        "c573611d5cf22c7e9446b5c356c19560bb7be9a0b1910d7588ffed3616069673",
    },
    "tree": {
        "bd380142f6c125c44b9ab5f79e3934241e16d38f5ae206e08333c56b926af6cc",
        "8db9ecf2bef487b90b8611c164b760494d4a9d25d1e6aa9afd9ace20ccb966cd",
        "f0d69486c76a9d5d1b4e7e5407066209ee51818e3a02540c24dbd64dbe720715",
        "ae3c46867c9923cf1646a6e77af4804a64d3f145c7359494a0f9cc01826c73c9",
        "3fed5d1704e880be62f5f3f96de285eae27c47cb9b8bc770454d1b22d8c9b434",
        "bf7f17c12eeda21169c933a90d091c5b1d55d25ebb748ae1fa14d6d8f1395d10",
        "15f2d17d9cd6cf9eb14c3abf4d9cde1d63773515e35b29ddf37716cedc60aea1",
        "11fa8bfb0ea274fd5473e2687960c487dfa298c492c639799abeed0af1c88ba7",
        "af13b91eb626ab9ece7582153b09caacdf7001898fc9341a781eeee44ef5cc9f",
        "7ec3e8fdd6155d99d98a9b69d2d4b02dc43c9417695f7632ad6559f42ecb96b3",
        "4671998072ae15e8efe69cecf5a3cd88757a431b72ff713dae4d596ca3d9191e",
        "06aef9dd18c662c04679cb976e636ff8804c5feacf7d1331862cacda74d718fa",
        "4905da958f67a17a02096d638616b63092c9dc0bfb6449987fa267152c43c447",
        "21ebf23d459e0819ba09b761f9ab066a65ea6cc68902085abfef2f34c852c0bf",
    },
}

SHMEM_LIBRARY_DIR = os.environ.get(
    "TILERT_SHMEM_LIB_DIR",
    "/assets/shmem-lib/lib"
    if os.path.isdir("/assets/shmem-lib/lib")
    else "/workspace/src/shmem-build/shmem/build/lib",
)
SHMEM_HEAP_BYTES = 1024 * 1024 * 1024
SHMEM_DEPENDENCIES = (
    "libshmem_utils.so",
    "libshmem.so",
    "libtp_push_kernel.so",
    "libkv_shuffle_kernel.so",
    "liballgather_kernel.so",
)


def pointers(values):
    return tuple(0 if value is None else value.data_ptr() for value in values)


def tensor_identity(value):
    return (
        value.device,
        value.data_ptr(),
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        value.dtype,
    )


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value, dtype, shape, name, device=None, physical=2):
    import torch_npu

    if (
        value.dtype != dtype
        or tuple(value.shape) != tuple(shape)
        or not value.is_contiguous()
        or value.device.type != "npu"
        or (device is not None and value.device != device)
        or torch_npu.get_npu_format(value) != physical
    ):
        raise ValueError(f"{name} has an unsupported dtype, shape or storage")


def kernel_prof_enabled(tree_manifest: dict | None) -> bool:
    if tree_manifest is None:
        return False
    flags = tree_manifest.get("flags") or ()
    return bool(
        tree_manifest.get("enable_kernel_prof", False)
        or "-DENABLE_KERNEL_PROF=1" in flags
    )


def extract_prof_acc(raw, *, device=None, sync: bool = True, cpu: bool = True):
    import torch

    nbytes = raw.numel() * raw.element_size()
    if nbytes < 512:
        raise RuntimeError("Arena body[6] scratch is smaller than 512 bytes")
    if nbytes % 8 != 0:
        raise RuntimeError(f"Arena body[6] scratch size {nbytes} is not 8-byte aligned")
    if sync and getattr(getattr(raw, "device", None), "type", None) == "npu":
        torch.npu.synchronize(device if device is not None else raw.device)
    words = raw if raw.dtype == torch.int64 else raw.view(torch.int64)
    prof = words.reshape(-1)[40:64]
    return prof.detach().cpu().clone() if cpu else prof


def capture_prof_acc(source, width: int | None = None, *, sync: bool = True, cpu: bool = True):
    if isinstance(source, PreparedChain):
        return source.capture_prof_acc(sync=sync, cpu=cpu)
    if hasattr(source, "capture_prof_acc"):
        return source.capture_prof_acc(width, sync=sync, cpu=cpu)
    return extract_prof_acc(source, sync=sync, cpu=cpu)


@dataclass(frozen=True)
class Libraries:
    indexed: Path
    shared: Path
    ffn: Path
    metadata: Path
    tree: Path

    @classmethod
    def from_json(cls, path):
        path = Path(path).resolve()
        values = json.loads(path.read_text())
        if set(values) != set(LIBRARY_HASHES):
            raise ValueError("Whole-layer libraries require exactly five paths")
        return cls(
            **{name: (path.parent / value).resolve() for name, value in values.items()}
        )

    def tree_manifest(self):
        manifest_path = self.tree.with_suffix(".manifest.json")
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text())
        return None

    def tree_flags(self) -> tuple[str, ...]:
        manifest = self.tree_manifest()
        if manifest is not None and manifest.get("flags"):
            return tuple(manifest["flags"])
        return R4_FLAGS

    def verify(self):
        for name, expected in LIBRARY_HASHES.items():
            if name == "tree":
                manifest = self.tree_manifest()
                if manifest is not None:
                    if _digest(self.tree) != manifest.get("sha256"):
                        raise ValueError("The built whole-layer tree library changed")
                    if "enable_kernel_prof" in manifest:
                        flag_prof = "-DENABLE_KERNEL_PROF=1" in (
                            manifest.get("flags") or ()
                        )
                        if bool(manifest.get("enable_kernel_prof")) != flag_prof:
                            raise ValueError("The built whole-layer tree prof flag changed")
                    if _MK_ROOT.is_dir():
                        if "effective_source_sha256" in manifest:
                            eff = effective_source_digest(manifest.get("flags", ()))
                            if manifest.get("effective_source_sha256") != eff:
                                raise ValueError(
                                    "The built whole-layer tree source changed"
                                )
                        elif manifest.get("source_sha256") != source_digest():
                            raise ValueError("The built whole-layer tree source changed")
                    continue
            allowed = {expected} if isinstance(expected, str) else set(expected)
            if _digest(getattr(self, name)) not in allowed:
                raise ValueError(f"The frozen whole-layer {name} library changed")

    def paths(self):
        return tuple(Path(getattr(self, name)) for name in LIBRARY_HASHES)


class LocalRecords:

    def __init__(
        self,
        library,
        norm,
        attention,
        post,
        routed,
        bridge,
        body,
        shared,
        shared_topk,
        geometry,
        scalars,
        sync,
        indexed,
    ):
        groups = (norm, attention, post, routed, bridge, body, shared)
        if tuple(map(len, groups)) != (8, 41, 3, 6, 12, 21, 4):
            raise ValueError(
                "Expected norm/attention/post/routed/bridge/body/shared "
                "slots 8/41/3/6/12/21/4"
            )
        if len(geometry) != 9 or len(scalars) != 5:
            raise ValueError("Expected nine geometry and five scalar fields")
        if norm[0] is None or norm[0].device.type != "npu":
            raise ValueError("The local-chain input must be an NPU tensor")
        required_attention = (
            *range(12),
            *range(18, 21),
            *range(22, 28),
            34,
            35,
            38,
            39,
            40,
        )
        if indexed:
            required_attention += (*range(12, 18), *range(28, 34), 36)
            if geometry[6]:
                required_attention += (37,)
        if geometry[8]:
            required_attention += (21,)
        if any(attention[index] is None for index in required_attention):
            raise ValueError("A required attention tensor is missing")
        if any(
            value is None
            for group in groups
            if group is not attention
            for value in group
        ):
            raise ValueError("A required norm or FFN tensor is missing")
        if not indexed and shared_topk is None:
            raise ValueError("Shared-indexer attention requires its retained top-k owner")
        owners = tuple(
            value for group in groups for value in group if value is not None
        )
        if shared_topk is not None:
            owners += (shared_topk,)
        if any(value.device != norm[0].device for value in owners):
            raise ValueError("All retained tensors must belong to one NPU")
        if geometry[0] != norm[0].shape[0] or norm[0].shape[1:] != (HIDDEN_SIZE,):
            raise ValueError("Geometry must describe the retained [M,6144] input")
        self.library = library
        self.body = tuple(body)
        self.owners = owners
        self.owner_pointers = pointers(owners)
        self.device = norm[0].device.index
        self.sync = int(sync)
        self.indexed = bool(indexed)
        self.config = ctypes.create_string_buffer(library.config_bytes)
        self.addresses = ctypes.create_string_buffer(library.address_bytes)
        arrays = [(ctypes.c_uint64 * len(group))(*pointers(group)) for group in groups]
        status = library.handle.layer_local_prepare_records(
            self.config,
            len(self.config),
            self.addresses,
            len(self.addresses),
            (ctypes.c_int64 * 9)(*geometry),
            (ctypes.c_float * 5)(*scalars),
            arrays[0],
            arrays[1],
            0 if shared_topk is None else shared_topk.data_ptr(),
            *arrays[2:],
        )
        library.check(status)

    def launch(self):
        if pointers(self.owners) != self.owner_pointers:
            raise RuntimeError("A retained local-chain tensor changed its data address")
        self.library.check(
            self.library.handle.layer_local_submit(
                self.device,
                self.sync,
                self.config,
                len(self.config),
                self.addresses,
                len(self.addresses),
                self.indexed,
            )
        )


class MetadataLibrary:

    def __init__(self, path, expected_sha256):
        path = Path(path)
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        allowed = (
            {expected_sha256}
            if isinstance(expected_sha256, str)
            else set(expected_sha256 or ())
        )
        if allowed and actual_sha256 not in allowed:
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
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float),
            U64_POINTER,
            U64_POINTER,
            ctypes.c_uint64,
            U64_POINTER,
            U64_POINTER,
            U64_POINTER,
            U64_POINTER,
            U64_POINTER,
        ]
        self.handle.layer_local_prepare_records.restype = ctypes.c_int
        self.handle.layer_local_submit.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_bool,
        ]
        self.handle.layer_local_submit.restype = ctypes.c_int

    def check(self, status):
        if status != 0:
            message = self.handle.layer_local_last_error().decode("utf-8")
            raise RuntimeError(f"Local-chain status {status}: {message}")


class TreeLibrary:

    def __init__(self, path, expected_sha256):
        path = Path(path)
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        allowed = (
            {expected_sha256}
            if isinstance(expected_sha256, str)
            else set(expected_sha256 or ())
        )
        if allowed and actual_sha256 not in allowed:
            raise RuntimeError("The TP layer library differs from its build record")
        self.handle = ctypes.CDLL(str(path.resolve()))
        signatures = {
            "record_bytes": ([ctypes.c_int], ctypes.c_size_t),
            "last_error": ([], ctypes.c_char_p),
            "initialize": (
                [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint64,
                ],
                ctypes.c_int,
            ),
            "window_create": (
                [ctypes.c_int64, ctypes.POINTER(ctypes.c_uint64)],
                ctypes.c_int,
            ),
            "window_info": (
                [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
                ctypes.c_int,
            ),
            "window_destroy": ([ctypes.c_uint64], ctypes.c_int),
            "finalize": ([], ctypes.c_int),
            "submit": (
                [
                    ctypes.c_int,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_bool,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                ],
                ctypes.c_int,
            ),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.handle, "layer_tp_tree_" + name)
            function.argtypes = arguments
            function.restype = result
        self.has_chain = hasattr(self.handle, "layer_tp_tree_submit_chain")
        self._set_mask_ptr = getattr(
            self.handle, "layer_tp_tree_set_mask_ptr", None
        )
        self._set_peer_weights_ptr = getattr(
            self.handle, "layer_tp_tree_set_peer_weights_ptr", None
        )
        if self.has_chain:
            submit_chain = self.handle.layer_tp_tree_submit_chain
            submit_chain.argtypes = [
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            submit_chain.restype = ctypes.c_int
            if self._set_mask_ptr is not None:
                self._set_mask_ptr.argtypes = [ctypes.c_uint64]
                self._set_mask_ptr.restype = None
            if self._set_peer_weights_ptr is not None:
                self._set_peer_weights_ptr.argtypes = [ctypes.c_uint64]
                self._set_peer_weights_ptr.restype = None
        sizes = tuple(
            self.handle.layer_tp_tree_record_bytes(index) for index in range(3)
        )
        if sizes != (664, 928, 48):
            raise RuntimeError(f"The TP layer record ABI changed: {sizes}")

    def set_mask_ptr(self, ptr):
        if self._set_mask_ptr is not None:
            self._set_mask_ptr(int(ptr))

    def set_peer_weights_ptr(self, ptr):
        if self._set_peer_weights_ptr is not None:
            self._set_peer_weights_ptr(int(ptr))

    def check(self, status):
        if status != 0:
            message = self.handle.layer_tp_tree_last_error().decode("utf-8")
            raise RuntimeError(f"TP layer status {status}: {message}")

    def initialize(self, device, rank, ranks, endpoint, heap_bytes):
        self.check(
            self.handle.layer_tp_tree_initialize(
                device, rank, ranks, endpoint.encode("utf-8"), heap_bytes
            )
        )

    def finalize(self):
        self.check(self.handle.layer_tp_tree_finalize())


class Window:

    def __init__(self, library, capacity):
        self.library = library
        handle = ctypes.c_uint64()
        library.check(
            library.handle.layer_tp_tree_window_create(
                capacity, ctypes.byref(handle)
            )
        )
        self.handle = handle.value
        fields = (ctypes.c_uint64 * 6)()
        library.check(
            library.handle.layer_tp_tree_window_info(
                self.handle, fields, len(fields)
            )
        )
        (
            self.address,
            self.bytes,
            self.capacity,
            self.rank,
            self.ranks,
            self.device,
        ) = fields

    def close(self):
        self.library.check(
            self.library.handle.layer_tp_tree_window_destroy(self.handle)
        )
        self.handle = 0


class PreparedLayer:

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
            if (
                value.dtype != torch.bfloat16
                or value.device != source.device
                or value.shape != source.shape
                or not value.is_contiguous()
                or torch_npu.get_npu_format(value) != 2
            ):
                raise ValueError(
                    "Reduction outputs must be contiguous physical-ND BF16 [M,6144]"
                )
        output_storage = tuple(
            value.untyped_storage().data_ptr() for value in outputs
        )
        local_storage = {value.untyped_storage().data_ptr() for value in local.owners}
        if len(set(output_storage)) != 2 or any(
            pointer in local_storage for pointer in output_storage
        ):
            raise ValueError(
                "Reduction outputs need storage distinct from local-layer tensors"
            )
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
        library.check(
            library.handle.layer_tp_tree_submit(
                local.device,
                local.sync,
                local.config,
                len(local.config),
                local.addresses,
                len(local.addresses),
                local.indexed,
                self.window.handle,
                *self.output_pointers,
            )
        )


class PreparedChain:

    def __init__(
        self,
        steps,
        steps_device,
        copy_stream=None,
        *,
        mask_ptr=0,
        peer_weights_ptr=0,
    ):
        import torch

        if not steps:
            raise ValueError("PreparedChain requires at least one PreparedLayer")
        first = steps[0]
        window = first.window
        device = first.local.device
        raw_steps = []
        for step in steps:
            local = step.local
            if step.window is not window or local.device != device:
                raise ValueError("All chain steps must share one device and window")
            raw_steps.append(
                struct.pack("<QQ", local.sync, int(local.indexed))
                + local.config.raw
                + local.addresses.raw
                + struct.pack(
                    "<QQQQqii",
                    window.address,
                    window.bytes,
                    *step.output_pointers,
                    window.capacity,
                    window.rank,
                    window.ranks,
                )
            )
        payload = b"".join(raw_steps)
        if len(payload) != len(steps) * 1656:
            raise RuntimeError("ChainStep byte length mismatch")
        if (
            steps_device.device.type != "npu"
            or steps_device.device.index != device
            or steps_device.numel() < len(payload)
            or steps_device.data_ptr() % 64 != 0
        ):
            raise ValueError("Chain device buffer must be 64B-aligned NPU uint8")
        host_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        if copy_stream is not None:
            acl = ctypes.CDLL("libascendcl.so")
            mode = ctypes.c_int(2)
            acl.aclmdlRICaptureThreadExchangeMode(ctypes.byref(mode))
            try:
                with torch.npu.stream(copy_stream):
                    steps_device[: len(payload)].copy_(host_tensor)
                copy_stream.synchronize()
            finally:
                acl.aclmdlRICaptureThreadExchangeMode(ctypes.byref(mode))
        else:
            steps_device[: len(payload)].copy_(host_tensor)
        self.device = device
        self.window = window
        self.count = len(steps)
        self.steps_host = ctypes.create_string_buffer(payload, len(payload))
        self.steps_bytes = len(payload)
        self.steps_device = steps_device
        self.steps_device_ptr = int(steps_device.data_ptr())
        self.steps = tuple(steps)
        self.mask_ptr = int(mask_ptr)
        self.peer_weights_ptr = int(peer_weights_ptr)

    def capture_prof_acc(self, *, sync: bool = True, cpu: bool = True):
        last_body = self.steps[-1].local.body
        if len(last_body) <= 6 or last_body[6] is None:
            raise RuntimeError("Last chain step is missing body[6] scratch")
        return extract_prof_acc(
            last_body[6], device=self.device, sync=sync, cpu=cpu
        )

    def launch(self):
        if self.window.handle == 0:
            raise RuntimeError("The layer communication window was released")
        if int(self.steps_device.data_ptr()) != self.steps_device_ptr:
            raise RuntimeError("The chain device descriptor buffer moved")
        for step in self.steps:
            local = step.local
            if pointers(local.owners) != local.owner_pointers:
                raise RuntimeError("A retained local-layer tensor changed its data address")
            if pointers(step.outputs) != step.output_pointers:
                raise RuntimeError("A retained reduction output changed its data address")
        library = self.window.library
        library.set_mask_ptr(self.mask_ptr)
        library.set_peer_weights_ptr(self.peer_weights_ptr)
        library.check(
            library.handle.layer_tp_tree_submit_chain(
                self.device,
                self.window.handle,
                self.steps_host,
                self.steps_bytes,
                self.steps_device_ptr,
                self.count,
            )
        )


def _host_input_layout(width):
    positions = (4 * width + 7) // 8 * 8
    slots = positions + 8 * width
    length = (slots + 4 * width + 7) // 8 * 8
    return (0, positions, slots, length), length + 8


def _host_input_views(packet, width):
    import torch

    offsets, _ = _host_input_layout(width)
    return tuple(
        packet[offset : offset + count * size].view(dtype)
        for offset, count, size, dtype in zip(
            offsets,
            (width, width, width, 1),
            (4, 8, 4, 4),
            (torch.int32, torch.int64, torch.int32, torch.int32),
            strict=True,
        )
    )


class HostInputs:

    def __init__(self, device, widths, slots=64, pin_memory=True):
        import torch

        self._packets = {}
        self._device_views = {}
        self._host_rows = {}
        self._host_views = {}
        self._next = {}
        for width in widths:
            _, size = _host_input_layout(width)
            packet = torch.empty(size, dtype=torch.uint8, device=device)
            host = torch.zeros(
                slots, size, dtype=torch.uint8, device="cpu", pin_memory=pin_memory
            )
            rows = tuple(host.unbind())
            self._packets[width] = packet
            self._device_views[width] = _host_input_views(packet, width)
            self._host_rows[width] = rows
            self._host_views[width] = tuple(_host_input_views(row, width) for row in rows)
            self._next[width] = 0

    def views(self, width):
        return self._device_views[width]

    def stage(self, tokens, position, depths=None):
        import torch

        width = len(tokens)
        index = self._next[width]
        self._next[width] = (index + 1) % len(self._host_rows[width])
        input_ids, positions, slots, length = self._host_views[width][index]
        input_ids.copy_(torch.tensor(tokens, dtype=torch.int32, device="cpu"))
        torch.arange(position, position + width, dtype=torch.int32, device="cpu", out=slots)
        if depths is None:
            positions.copy_(slots)
        else:
            positions.copy_(torch.tensor(depths, dtype=torch.int64, device="cpu"))
            positions.add_(position)
        length.fill_(position + width)
        self._packets[width].copy_(self._host_rows[width][index], non_blocking=True)
        return length, positions


def _bind_shmem_device(device):
    import torch

    torch.npu.set_device(device)
    index = int(str(device).rsplit(":", 1)[-1])
    runtime = ctypes.CDLL("libascendcl.so")
    bound = runtime.aclrtSetDevice(ctypes.c_int(index))
    if bound != 0:
        raise RuntimeError(f"aclrtSetDevice({index}) returned {bound}")


def _load_shmem_libs():
    import torch

    for name in SHMEM_DEPENDENCIES:
        ctypes.CDLL(f"{SHMEM_LIBRARY_DIR}/{name}", mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(f"{SHMEM_LIBRARY_DIR}/aclshmem_torch.so")
    bf16_lib = f"{SHMEM_LIBRARY_DIR}/tilert_shmem_bf16.so"
    if os.path.exists(bf16_lib):
        torch.ops.load_library(bf16_lib)


class OneSidedReduce:

    def __init__(self, rank, device, world_size, endpoint):
        import torch

        _bind_shmem_device(device)
        _load_shmem_libs()
        self.manager = torch.classes.ShmemOps.Manager()
        status = self.manager.attr_init(rank, world_size, SHMEM_HEAP_BYTES, endpoint)
        if status != 0:
            raise RuntimeError(f"aclshmem init returned {status}")
        self.has_bf16_op = hasattr(
            getattr(torch.ops, "tilert", None), "onesided_all_reduce"
        )
        self.has_bf16_out_op = hasattr(
            getattr(torch.ops, "tilert", None), "onesided_all_reduce_out"
        )
        self.target_out = None
        if self.has_bf16_op:
            self.reducer = None
            probe = torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
            _ = torch.ops.tilert.onesided_all_reduce(probe)
            torch.npu.synchronize()
        else:
            self.reducer = torch.classes.ShmemOps.AllReduce()

    def sum(self, value):
        import torch

        out = self.target_out
        self.target_out = None
        if (
            out is not None
            and self.has_bf16_out_op
            and value.dtype in (torch.bfloat16, torch.float32)
            and out.dtype == value.dtype
            and out.numel() == value.numel()
            and out.is_contiguous()
        ):
            return torch.ops.tilert.onesided_all_reduce_out(value, out)
        if self.has_bf16_op and value.dtype in (torch.bfloat16, torch.float32):
            return torch.ops.tilert.onesided_all_reduce(value)
        out = torch.empty_like(value)
        self.reducer.compute(out, value)
        return out

    def close(self):
        if self.manager is not None:
            self.manager.finalize()
        self.manager = None
        self.reducer = None


@dataclass(frozen=True)
class CacheRegion:
    name: str
    dtype: str
    row_bytes: int
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def cache_layout(capacity_tokens: int = 9344, *, parent_hidden: bool = True) -> dict:
    if type(parent_hidden) is not bool:
        raise ValueError("parent_hidden must be a boolean")
    capacity_tokens = _index(capacity_tokens)
    if capacity_tokens < 1:
        raise ValueError("cache capacity must be positive")
    pages = (capacity_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    padded_capacity = pages * PAGE_SIZE
    regions = {}
    offset = 0

    def add(name, dtype, row_bytes, shape):
        nonlocal offset
        nbytes = padded_capacity * row_bytes
        regions[name] = {
            "dtype": dtype,
            "row_bytes": row_bytes,
            "shape": shape,
            "offset": offset,
            "nbytes": nbytes,
        }
        offset += nbytes

    for layer in range(NUM_LAYERS):
        add(f"mla_latent.{layer}", "bfloat16", 1024, (pages, PAGE_SIZE, 1, 512))
        add(f"mla_rope.{layer}", "bfloat16", 128, (pages, PAGE_SIZE, 1, 64))
    for layer in FULL_INDEX_LAYERS:
        quantized = layer in INT8_INDEX_LAYERS
        add(
            f"index.{layer}",
            "int8" if quantized else "bfloat16",
            128 if quantized else 256,
            (pages, PAGE_SIZE, 1, 128),
        )
        if quantized:
            add(f"index_scale.{layer}", "float16", 2, (pages, PAGE_SIZE, 1))
    if parent_hidden:
        regions[PARENT_HIDDEN_REGION] = {
            "dtype": "bfloat16",
            "row_bytes": PARENT_HIDDEN_BYTES,
            "shape": (1, HIDDEN_SIZE),
            "offset": offset,
            "nbytes": PARENT_HIDDEN_BYTES,
        }
        offset += PARENT_HIDDEN_BYTES
    return {
        "name": LAYOUT_NAME,
        "version": (
            PARENT_HIDDEN_LAYOUT_VERSION if parent_hidden else LAYOUT_VERSION
        ),
        "num_ranks": NUM_RANKS,
        "page_size": PAGE_SIZE,
        "capacity_tokens": padded_capacity,
        "nbytes": offset,
        "regions": regions,
    }


def compute_cache_digest(cache, position: int) -> dict:
    import torch

    layers = []
    for layer in cache:
        fields = {}
        for name, value in layer.items():
            valid = value[:position].contiguous()
            fields[name] = {
                "dtype": str(valid.dtype),
                "shape": list(valid.shape),
                "sha256": hashlib.sha256(
                    valid.view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest(),
            }
        layers.append(fields)
    return {"position": position, "layers": layers}


def _alloc_aligned(nbytes: int, align: int, device, *, zero: bool = False):
    import torch

    raw = torch.empty(nbytes + align, dtype=torch.uint8, device=device)
    offset = (-int(raw.data_ptr())) % align
    view = raw[offset : offset + nbytes]
    if zero:
        view.zero_()
    return raw, view


class FrozenPrefixArena:

    def __init__(
        self,
        device,
        capacity_tokens: int = 9280,
        parent_hidden: bool = True,
    ):
        import torch

        self.device = torch.device(device)
        self.capacity_tokens = int(capacity_tokens)
        self.layout = cache_layout(
            self.capacity_tokens, parent_hidden=parent_hidden
        )
        self.padded_capacity = int(self.layout["capacity_tokens"])
        self.pages = self.padded_capacity // PAGE_SIZE
        self._raw_storage, self.storage = _alloc_aligned(
            int(self.layout["nbytes"]), 512, self.device, zero=True
        )
        self.views = {}
        for name, region in self.layout["regions"].items():
            start = region["offset"]
            self.views[name] = (
                self.storage[start : start + region["nbytes"]]
                .view(getattr(torch, region["dtype"]))
                .view(region["shape"])
            )
        self.main_cache = []
        for layer in range(MAIN_LAYERS):
            entry = {
                "latent": self.views[f"mla_latent.{layer}"].view(-1, 512),
                "rope": self.views[f"mla_rope.{layer}"].view(-1, 64),
            }
            if layer in FULL_INDEX_LAYERS:
                entry["index_keys"] = self.views[f"index.{layer}"].view(-1, 128)
            if layer in INT8_INDEX_LAYERS:
                entry["index_scales"] = self.views[f"index_scale.{layer}"].view(-1)
            self.main_cache.append(entry)
        self.draft_cache = {
            "latent": self.views["mla_latent.78"].view(-1, 512),
            "rope": self.views["mla_rope.78"].view(-1, 64),
            "index_keys": self.views["index.78"].view(-1, 128),
        }
        self.parent_hidden = self.views.get(PARENT_HIDDEN_REGION)
        self.block_table = torch.arange(
            self.pages, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        cache_tensors = [
            tensor for layer in self.main_cache for tensor in layer.values()
        ]
        self.num_cache_tensors = len(cache_tensors)
        descs_cpu = torch.zeros((512,), dtype=torch.int64)
        for idx_t, tensor in enumerate(cache_tensors):
            row_bytes = (
                tensor.stride(0) * tensor.element_size()
                if tensor.dim() > 1
                else tensor.element_size()
            )
            descs_cpu[2 * idx_t] = int(tensor.data_ptr())
            descs_cpu[2 * idx_t + 1] = int(row_bytes)
        self.descs_dev = descs_cpu.to(self.device)
        self._snapshot_buf = None
        self._snapshot_copies = ()
        self.prompt_token_ids = ()
        self.first_token_id = None
        self.prefix_len = 0
        self.mtp_prefix_len = 0
        self.main_pos = 0
        self.draft_pos = 0

    def install_snapshot(
        self,
        main_layers,
        mtp_layer,
        parent_hidden,
        first_token_id: int,
        prompt_token_ids=(),
    ) -> None:
        import torch

        if len(main_layers) != MAIN_LAYERS:
            raise ValueError(f"Expected {MAIN_LAYERS} main prefix layers")
        p_len = int(main_layers[0]["latent"].shape[0])
        if not 1 <= p_len < self.padded_capacity:
            raise ValueError(f"Prefix length {p_len} exceeds capacity {self.padded_capacity}")
        m_len = int(mtp_layer["latent"].shape[0])
        if m_len not in (p_len - 1, p_len):
            raise ValueError(f"MTP prefix length {m_len} must be P-1 or P ({p_len})")
        if self.parent_hidden is not None:
            if (
                tuple(parent_hidden.shape) != (1, HIDDEN_SIZE)
                or parent_hidden.dtype != torch.bfloat16
            ):
                raise ValueError("parent_hidden must be BF16 [1, 6144]")

        parts = []
        dst_views = []
        for layer, (src, dst) in enumerate(zip(main_layers, self.main_cache)):
            clean = {k: v for k, v in src.items() if v is not None}
            if set(clean) != set(dst):
                raise ValueError(f"Layer {layer}: prefix keys mismatch")
            for name, dst_t in dst.items():
                val = clean[name]
                if tuple(val.shape) != tuple(dst_t[:p_len].shape) or val.dtype != dst_t.dtype:
                    raise ValueError(f"Layer {layer}.{name} prefix shape/dtype mismatch")
                raw_src = val.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
                parts.append(raw_src)
                dst_views.append(dst_t[:p_len].view(torch.uint8).reshape(-1))
        for name, dst_t in self.draft_cache.items():
            val = mtp_layer[name]
            if tuple(val.shape) != tuple(dst_t[:m_len].shape) or val.dtype != dst_t.dtype:
                raise ValueError(f"Draft {name} prefix shape/dtype mismatch")
            raw_src = val.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
            parts.append(raw_src)
            dst_views.append(dst_t[:m_len].view(torch.uint8).reshape(-1))
        if self.parent_hidden is not None:
            raw_par = (
                parent_hidden.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
            )
            parts.append(raw_par)
            dst_views.append(self.parent_hidden.view(torch.uint8).reshape(-1))

        total_bytes = sum(part.numel() for part in parts)
        pin_ok = self.device.type == "npu"
        buf = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=pin_ok)
        copies = []
        cursor = 0
        for part, dst_view in zip(parts, dst_views):
            size = part.numel()
            src_slice = buf[cursor : cursor + size]
            src_slice.copy_(part)
            copies.append((src_slice, dst_view))
            cursor += size
        self._snapshot_buf = buf
        self._snapshot_copies = tuple(copies)
        self.prefix_len = p_len
        self.mtp_prefix_len = m_len
        self.first_token_id = int(first_token_id)
        self.prompt_token_ids = tuple(int(t) for t in prompt_token_ids)
        self.restore()

    @classmethod
    def from_case(
        cls,
        case_dir: str | Path,
        mtp_prefix_path: str | Path,
        device,
        capacity_tokens: int = 9280,
        main_prefix_path: str | Path | None = None,
    ) -> "FrozenPrefixArena":
        import torch

        case_path = Path(case_dir)
        main_path = (
            Path(main_prefix_path)
            if main_prefix_path is not None
            else case_path / "main-prefix" / "rank-0.pt"
        )
        main_blob = torch.load(main_path, map_location="cpu", weights_only=False)
        mtp_blob = torch.load(
            Path(mtp_prefix_path), map_location="cpu", weights_only=False
        )
        prompt_ids = list(main_blob["prompt_token_ids"])
        prompt_json = case_path / "prompt.json"
        if prompt_json.is_file():
            expected_prompt = json.loads(prompt_json.read_text())["prompt_token_ids"]
            if prompt_ids != expected_prompt:
                raise ValueError(f"Prompt IDs in {main_path} differ from {prompt_json}")
        arena = cls(device=device, capacity_tokens=capacity_tokens, parent_hidden=True)
        arena.install_snapshot(
            main_blob["layers"],
            mtp_blob,
            main_blob["parent_hidden"],
            int(main_blob["first_token_id"]),
            prompt_ids,
        )
        return arena

    def restore(self) -> None:
        import torch

        if self._snapshot_buf is None:
            raise RuntimeError("No frozen prefix snapshot installed")
        if self.device.type == "npu":
            with torch.npu.device(self.device):
                self.storage.zero_()
                for src_slice, dst_view in self._snapshot_copies:
                    dst_view.copy_(src_slice, non_blocking=True)
                torch.npu.synchronize(self.device)
        else:
            self.storage.zero_()
            for src_slice, dst_view in self._snapshot_copies:
                dst_view.copy_(src_slice)
        self.main_pos = self.prefix_len
        self.draft_pos = self.mtp_prefix_len

    def main_state(self, position: int = FINAL_POS) -> dict:
        import torch

        if self.device.type == "npu":
            torch.npu.synchronize(self.device)
        return compute_cache_digest(self.main_cache, int(position))

    def draft_state(self, position: int = FINAL_POS) -> dict:
        import torch

        if self.device.type == "npu":
            torch.npu.synchronize(self.device)
        return compute_cache_digest([self.draft_cache], int(position))


class RoundState:
    POS_IDX = 0
    DRAFT_POS_IDX = 1
    DONE_IDX = 2
    EMITTED_IDX = 3
    MAX_NEW_TOKENS_IDX = 4
    ROUND_IDX = 5
    SEEDED_IDX = 6
    PENDING_TOKEN_IDX = 7
    HORIZON_LEVEL_IDX = 8
    W_ACT_IDX = 9

    def __init__(self, device):
        import torch

        self.device = torch.device(device)
        self._raw, self.buf = _alloc_aligned(256, 64, self.device, zero=True)
        self.words = self.buf.view(torch.int64)

    @property
    def data_ptr(self) -> int:
        return int(self.buf.data_ptr())

    def reset(
        self,
        prefix_len: int = PREFIX_LEN,
        first_token: int = 0,
        max_new_tokens: int = 1024,
        draft_prefix_len: int = MTP_PREFIX_LEN,
    ) -> None:
        import torch

        init = torch.zeros((32,), dtype=torch.int64)
        init[self.POS_IDX] = int(prefix_len)
        init[self.DRAFT_POS_IDX] = int(draft_prefix_len)
        init[self.DONE_IDX] = 0
        init[self.EMITTED_IDX] = 1 if max_new_tokens > 0 else 0
        init[self.MAX_NEW_TOKENS_IDX] = int(max_new_tokens)
        init[self.ROUND_IDX] = 0
        init[self.SEEDED_IDX] = 0
        init[self.PENDING_TOKEN_IDX] = int(first_token)
        init[self.HORIZON_LEVEL_IDX] = 1
        self.words.copy_(init.to(self.device))
        if self.device.type == "npu":
            torch.npu.synchronize(self.device)


class HostReceipt:

    def __init__(
        self,
        device,
        slots: int = 4,
        slot_i32_len: int = 16,
        token_bytes: int = 512,
    ):
        import torch

        self.device = torch.device(device)
        self.slots = int(slots)
        self.slot_i32_len = int(slot_i32_len)
        self.token_bytes = int(token_bytes)
        pin_ok = self.device.type == "npu"
        self.host_ring = torch.zeros(
            (self.slots, self.slot_i32_len), dtype=torch.int32, pin_memory=pin_ok
        )
        self.host_tokens = torch.zeros(
            (self.slots, self.token_bytes), dtype=torch.uint8, pin_memory=pin_ok
        )
        self._raw_meta, meta_u8 = _alloc_aligned(
            self.slot_i32_len * 4, 64, self.device, zero=True
        )
        self.device_meta = meta_u8.view(torch.int32)
        self._raw_tok, self.device_tokens = _alloc_aligned(
            self.token_bytes, 64, self.device, zero=True
        )

    def reset(self) -> None:
        self.host_ring.zero_()
        self.host_tokens.zero_()
        self.device_meta.zero_()
        self.device_tokens.zero_()

    def copy_slot_async(self, slot: int, stream=None) -> None:
        import torch

        idx = int(slot) % self.slots
        if stream is not None and self.device.type == "npu":
            with torch.npu.stream(stream):
                self.host_ring[idx].copy_(self.device_meta, non_blocking=True)
                self.host_tokens[idx].copy_(self.device_tokens, non_blocking=True)
        else:
            self.host_ring[idx].copy_(self.device_meta, non_blocking=True)
            self.host_tokens[idx].copy_(self.device_tokens, non_blocking=True)

    def read_slot(self, slot: int) -> tuple[list[int], list[int]]:
        import torch

        idx = int(slot) % self.slots
        meta = self.host_ring[idx].tolist()
        count = max(0, int(meta[1]))
        toks = self.host_tokens[idx].view(torch.int32)[:count].tolist()
        return meta, toks


@dataclass
class LayerWeights:
    layer_idx: int
    mode: str
    main: tuple
    front: tuple
    tail: tuple
    routed: tuple
    shared: tuple
    input_gamma: object
    post: tuple
    dense_mlp: tuple | None
    scalars: tuple
    topk: object
    owners: tuple


@dataclass
class DraftWeights:
    cal_wt: object
    head_q8_nz: object
    head_scale: object
    head_w_shard: object
    input_gamma: object
    main: tuple
    front: tuple
    tail: tuple
    post: tuple
    routed: tuple
    routed_folded: tuple
    shared: tuple
    scalars: tuple
    rot_w_shard: object
    w_eh_shard: object
    enorm_w: object
    hnorm_w: object
    shared_head_norm_w: object


class ModelArena:

    def __init__(
        self,
        *,
        device,
        rank: int,
        world_size: int,
        nd_raw,
        nd_arena,
        nz_holders: tuple,
        embed_w,
        norm_w,
        lm_head_w,
        cos_cache,
        sin_cache,
        layers: tuple[LayerWeights, ...],
        draft: DraftWeights | None,
        peer_weights_buf,
        peer_weights_ptr: int,
        tree_mask_raw,
        tree_mask_i32,
        topk_bufs: tuple = (),
    ):
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._nd_raw = nd_raw
        self.nd_arena = nd_arena
        self.nz_holders = tuple(nz_holders)
        self.embed_w = embed_w
        self.norm_w = norm_w
        self.lm_head_w = lm_head_w
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.layers = tuple(layers)
        self.draft = draft
        self.peer_weights_buf = peer_weights_buf
        self.peer_weights_ptr = int(peer_weights_ptr)
        self._tree_mask_raw = tree_mask_raw
        self.tree_mask_i32 = tree_mask_i32
        self._topk_bufs = tuple(topk_bufs)
        self.round_state = RoundState(self.device)
        self.host_receipt = HostReceipt(self.device)
        self._chain_step_buffers = []

    def bind_chain_step_buffers(self, step_count: int = MAIN_LAYERS):
        chain_bytes = int(step_count) * 1656
        raw, buf = _alloc_aligned(chain_bytes, 64, self.device, zero=False)
        self._chain_step_buffers.append((raw, buf))
        return buf

    def close(self) -> None:
        self._chain_step_buffers.clear()
        self.nz_holders = ()
        self.layers = ()
        self.draft = None
        self._topk_bufs = ()
        self._nd_raw = None
        self.nd_arena = None
        self.peer_weights_buf = None
        self.peer_weights_ptr = 0
        self._tree_mask_raw = None
        self.tree_mask_i32 = None
        gc.collect()
        if getattr(self.device, "type", None) == "npu":
            import torch

            torch.npu.empty_cache()
