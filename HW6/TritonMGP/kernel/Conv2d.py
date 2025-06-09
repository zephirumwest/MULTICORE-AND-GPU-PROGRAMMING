import triton
import triton.language as tl

# Ensure mgp is available
try:
    from mgp import empty
except ImportError:
    import torch as mgp_torch_fallback # Fallback for reasoning
    empty = mgp_torch_fallback.empty

@triton.jit
def _conv2d_forward_kernel_direct(
    x_ptr, weight_ptr, y_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT, KH, KW,
    SH, SW, PH_TOP, PW_LEFT,
    H_OUT, W_OUT,
    stride_xn, stride_xc_in, stride_xh_in, stride_xw_in,
    stride_wc_out, stride_wc_in, stride_wkh, stride_wkw,
    stride_yn, stride_yc_out, stride_yh_out, stride_yw_out,
    BLOCK_N_SIZE: tl.constexpr, BLOCK_COUT_SIZE: tl.constexpr,
    BLOCK_HOUT_SIZE: tl.constexpr, BLOCK_WOUT_SIZE: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_cout_group = tl.program_id(1)
    pid_spatial_group = tl.program_id(2)

    offs_n_curr = pid_n * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
    offs_cout_curr = pid_cout_group * BLOCK_COUT_SIZE + tl.arange(0, BLOCK_COUT_SIZE)

    num_wout_blocks_total = tl.cdiv(W_OUT, BLOCK_WOUT_SIZE)
    pid_hout_block = pid_spatial_group // num_wout_blocks_total
    pid_wout_block = pid_spatial_group % num_wout_blocks_total

    offs_hout_curr = pid_hout_block * BLOCK_HOUT_SIZE + tl.arange(0, BLOCK_HOUT_SIZE)
    offs_wout_curr = pid_wout_block * BLOCK_WOUT_SIZE + tl.arange(0, BLOCK_WOUT_SIZE)

    acc = tl.zeros((BLOCK_N_SIZE, BLOCK_COUT_SIZE, BLOCK_HOUT_SIZE, BLOCK_WOUT_SIZE), dtype=tl.float32)

    for c_in_idx in range(C_IN):
        for kh_idx in range(KH):
            for kw_idx in range(KW):
                h_in_coords = offs_hout_curr[None, None, :, None] * SH - PH_TOP + kh_idx
                w_in_coords = offs_wout_curr[None, None, None, :] * SW - PW_LEFT + kw_idx

                x_block_ptrs = x_ptr + (
                    offs_n_curr[:, None, None, None] * stride_xn +
                    c_in_idx * stride_xc_in +
                    h_in_coords * stride_xh_in +
                    w_in_coords * stride_xw_in
                )

                mask_out_n_valid = offs_n_curr[:, None, None, None] < N
                mask_out_h_valid = offs_hout_curr[None, None, :, None] < H_OUT
                mask_out_w_valid = offs_wout_curr[None, None, None, :] < W_OUT
                mask_in_h_valid = (h_in_coords >= 0) & (h_in_coords < H_IN)
                mask_in_w_valid = (w_in_coords >= 0) & (w_in_coords < W_IN)

                final_combined_mask_for_x_load = mask_out_n_valid & \
                                                 mask_out_h_valid & \
                                                 mask_out_w_valid & \
                                                 mask_in_h_valid & \
                                                 mask_in_w_valid

                x_vals_loaded_block = tl.load(x_block_ptrs, mask=final_combined_mask_for_x_load, other=0.0)
                # x_vals_loaded_block has shape (N_B, 1, H_B, W_B)
                x_vals = tl.reshape(x_vals_loaded_block, (BLOCK_N_SIZE, BLOCK_HOUT_SIZE, BLOCK_WOUT_SIZE))


                weight_element_ptrs = weight_ptr + \
                                      offs_cout_curr * stride_wc_out + \
                                      c_in_idx * stride_wc_in + \
                                      kh_idx * stride_wkh + \
                                      kw_idx * stride_wkw
                mask_for_w_load_cout = offs_cout_curr < C_OUT
                w_vals = tl.load(weight_element_ptrs, mask=mask_for_w_load_cout, other=0.0)

                acc += x_vals[:, None, :, :] * w_vals[None, :, None, None]

    y_block_ptrs = y_ptr + (
        offs_n_curr[:,None,None,None] * stride_yn +
        offs_cout_curr[None,:,None,None] * stride_yc_out +
        offs_hout_curr[None,None,:,None] * stride_yh_out +
        offs_wout_curr[None,None,None,:] * stride_yw_out
    )

    y_store_mask = (offs_n_curr[:,None,None,None] < N) & \
                   (offs_cout_curr[None,:,None,None] < C_OUT) & \
                   (offs_hout_curr[None,None,:,None] < H_OUT) & \
                   (offs_wout_curr[None,None,None,:] < W_OUT)

    tl.store(y_block_ptrs, acc.to(y_ptr.type.element_ty), mask=y_store_mask)


def triton_conv2d(x, weight, bias, stride: tuple, padding: tuple, dilation: tuple): # types are torch.Tensor
    N, C_IN, H_IN, W_IN = x.shape
    C_OUT, C_IN_w, KH, KW = weight.shape

    assert C_IN == C_IN_w, f"Input channels mismatch: {C_IN} vs {C_IN_w}"
    assert dilation == (1,1), "Dilation not supported by this kernel"

    SH, SW = stride
    PH_TOP, PW_LEFT = padding[0], padding[1]

    H_OUT = (H_IN + 2 * PH_TOP - KH) // SH + 1
    W_OUT = (W_IN + 2 * PW_LEFT - KW) // SW + 1

    y = empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
    if H_OUT <= 0 or W_OUT <=0:
        return y


    stride_xn, stride_xc_in, stride_xh_in, stride_xw_in = x.stride()
    stride_wc_out, stride_wc_in, stride_wkh, stride_wkw = weight.stride()
    stride_yn, stride_yc_out, stride_yh_out, stride_yw_out = y.stride()

    # Default block sizes, can be tuned with @triton.autotune
    BLOCK_N_SIZE = 1
    # Determine BLOCK_COUT_SIZE
    if C_OUT == 0: BLOCK_COUT_SIZE = 1 # Avoid error with next_power_of_2(0)
    elif C_OUT < 16 : BLOCK_COUT_SIZE = triton.next_power_of_2(C_OUT)
    elif C_OUT < 32: BLOCK_COUT_SIZE = 16
    else: BLOCK_COUT_SIZE = 32

    # Determine BLOCK_HOUT_SIZE
    if H_OUT == 0: BLOCK_HOUT_SIZE = 1
    elif H_OUT < 8 : BLOCK_HOUT_SIZE = triton.next_power_of_2(H_OUT)
    else: BLOCK_HOUT_SIZE = 8

    # Determine BLOCK_WOUT_SIZE
    if W_OUT == 0: BLOCK_WOUT_SIZE = 1
    elif W_OUT < 8 : BLOCK_WOUT_SIZE = triton.next_power_of_2(W_OUT)
    else: BLOCK_WOUT_SIZE = 8

    GROUP_M = 1

    grid_n_blocks = triton.cdiv(N, BLOCK_N_SIZE)
    grid_cout_groups = triton.cdiv(C_OUT, BLOCK_COUT_SIZE)
    grid_hout_blocks_total = triton.cdiv(H_OUT, BLOCK_HOUT_SIZE)
    grid_wout_blocks_total = triton.cdiv(W_OUT, BLOCK_WOUT_SIZE)
    grid_spatial_groups = grid_hout_blocks_total * grid_wout_blocks_total

    if H_OUT <= 0 or W_OUT <= 0: # Already handled by early return of y
        pass

    # Ensure grid dimensions are positive if output is expected
    if N > 0 and C_OUT > 0 and H_OUT > 0 and W_OUT > 0:
        if grid_n_blocks == 0: grid_n_blocks = 1
        if grid_cout_groups == 0: grid_cout_groups = 1
        if grid_spatial_groups == 0: grid_spatial_groups = 1 # Can happen if H_OUT < BLOCK_HOUT_SIZE etc.
    else: # If any critical dimension is zero, output is empty or invalid
        return y


    grid = (grid_n_blocks, grid_cout_groups, grid_spatial_groups)

    if grid[0] == 0 or grid[1] == 0 or grid[2] == 0: # Final check
        return y


    _conv2d_forward_kernel_direct[grid](
        x, weight, y,
        N, C_IN, H_IN, W_IN, C_OUT, KH, KW,
        SH, SW, PH_TOP, PW_LEFT,
        H_OUT, W_OUT,
        stride_xn, stride_xc_in, stride_xh_in, stride_xw_in,
        stride_wc_out, stride_wc_in, stride_wkh, stride_wkw,
        stride_yn, stride_yc_out, stride_yh_out, stride_yw_out,
        BLOCK_N_SIZE=BLOCK_N_SIZE, BLOCK_COUT_SIZE=BLOCK_COUT_SIZE,
        BLOCK_HOUT_SIZE=BLOCK_HOUT_SIZE, BLOCK_WOUT_SIZE=BLOCK_WOUT_SIZE,
        GROUP_M=GROUP_M
    )
    return y