"""A3 streaming GM reads with two UB buffers and a live checksum."""

import torch_npu  # noqa: F401
import tilelang
from tilelang import language as T


def compile_read_kernel(workers: int, tiles_per_worker: int, tile_size: int):
    """Compile 24-by-2 AIV work groups for the default 48-worker shape."""
    return tilelang.compile(
        _read_prim_func(workers, tiles_per_worker, tile_size),
        out_idx=[],
        target="ascendc",
        platform="A3",
        execution_backend="cython",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
        },
    )


def _read_prim_func(workers: int, tiles_per_worker: int, tile_size: int):
    @T.prim_func
    def streaming_read(
        data: T.Tensor((workers, tiles_per_worker, tile_size), "float32"),
        checksum: T.Tensor((workers, 8), "float32"),
    ):
        with T.Kernel(workers // 2, is_npu=True) as (block, vector):
            worker = block * 2 + vector
            incoming = T.alloc_ub((2, tile_size), "float32")
            accumulator = T.alloc_ub((tile_size,), "float32")
            reduced = T.alloc_ub((8,), "float32")

            with T.Scope("V"):
                T.tile.fill(accumulator, 0.0)
                T.tile.fill(reduced, 0.0)
                T.pipe_barrier("v")
                T.set_flag("v", "mte2", 0)
                T.set_flag("v", "mte2", 1)
                T.wait_flag("v", "mte2", 0)
                T.copy(data[worker, 0, :], incoming[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.serial(tiles_per_worker):
                    current = tile % 2
                    following = (tile + 1) % 2
                    if tile + 1 < tiles_per_worker:
                        T.wait_flag("v", "mte2", following)
                        T.copy(data[worker, tile + 1, :],
                               incoming[following, :])
                        T.set_flag("mte2", "v", following)

                    T.wait_flag("mte2", "v", current)
                    T.tile.add(accumulator, accumulator, incoming[current, :])
                    T.pipe_barrier("v")
                    T.set_flag("v", "mte2", current)

                T.wait_flag("v", "mte2", 0)
                T.wait_flag("v", "mte2", 1)
                T.reduce_sum(accumulator, reduced[0:1], dim=0)
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.copy(reduced, checksum[worker, :])
                T.barrier_all()

    return streaming_read
