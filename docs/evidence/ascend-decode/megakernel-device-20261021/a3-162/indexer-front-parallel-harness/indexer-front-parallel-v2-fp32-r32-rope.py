# Source-replayed vLLM-Ascend Apache-2.0 RoPE function.
import triton
import triton.language as tl

@triton.jit
def _triton_rope_siso(
    qk_ptr,
    qk_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    cos_sin_ptr,
    cos_sin_row_stride,
    pos_ptr,
    num_tokens,
    n_h: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_n_h: tl.constexpr,
    pad_rope_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    USE_COS_SIN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)

    for row_idx in tl.range(pid, num_tokens, row_block_size):
        qk_start_ptr = qk_ptr + row_idx * qk_row_stride

        # ####################################################################
        # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
        # m of this program instance
        # ####################################################################
        cos_offsets = tl.arange(0, pad_rope_dim // 2)
        sin_offsets = cos_offsets + (rope_dim // 2)
        cos_mask = cos_offsets < (rope_dim // 2)
        if USE_COS_SIN:
            pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
            cos_start_ptr = cos_sin_ptr + pos_idx * cos_sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(cos_start_ptr + sin_offsets, mask=cos_mask, other=0).to(tl.float32)
        else:
            cos_start_ptr = cos_ptr + row_idx * cos_row_stride
            sin_start_ptr = sin_ptr + row_idx * sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(sin_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)

        # ####################################################################
        # Load the left and right half of q and k for the current
        # program instance (i.e. for the current token) separately
        # ####################################################################
        # left half of the head
        if IS_NEOX_STYLE:
            first_half_offsets = tl.arange(0, pad_n_h)[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
        else:
            first_half_offsets = tl.arange(0, pad_n_h)[:, None] * hd + (2 * tl.arange(0, pad_rope_dim // 2)[None, :])

        first_mask = (tl.arange(0, pad_n_h)[:, None] < n_h) & (
            tl.arange(0, pad_rope_dim // 2)[None, :] < (rope_dim // 2)
        )
        qk_tile_1 = tl.load(qk_start_ptr + first_half_offsets, mask=first_mask, other=0).to(sin_row.dtype)

        # right half of the head
        if IS_NEOX_STYLE:
            second_half_offsets = first_half_offsets + (rope_dim // 2)
        else:
            second_half_offsets = first_half_offsets + 1
        second_mask = first_mask
        qk_tile_2 = tl.load(qk_start_ptr + second_half_offsets, mask=second_mask, other=0).to(sin_row.dtype)

        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        new_qk_tile_1 = qk_tile_1 * cos_row - qk_tile_2 * sin_row
        tl.store(qk_start_ptr + first_half_offsets, new_qk_tile_1, mask=first_mask)

        new_qk_tile_2 = qk_tile_2 * cos_row + qk_tile_1 * sin_row
        tl.store(qk_start_ptr + second_half_offsets, new_qk_tile_2, mask=second_mask)
