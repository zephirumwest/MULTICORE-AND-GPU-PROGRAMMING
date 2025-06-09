import triton
import triton.language as tl

# Ensure mgp is available
try:
    from mgp import empty
except ImportError:
    import torch as mgp_torch_fallback # Fallback for reasoning
    empty = mgp_torch_fallback.empty

@triton.jit
def _maxpool2d_forward_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    KH, KW,
    SH, SW,
    PH_TOP, PH_BOTTOM, PW_LEFT, PW_RIGHT, 
    OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, 
    BLOCK_SIZE_OH: tl.constexpr, BLOCK_SIZE_OW: tl.constexpr
):
    pid_n = tl.program_id(0) 
    pid_c = tl.program_id(1) 
    pid_oh_ow_group = tl.program_id(2)

    pid_oh = (pid_oh_ow_group // tl.cdiv(OW, BLOCK_SIZE_OW))
    pid_ow = (pid_oh_ow_group % tl.cdiv(OW, BLOCK_SIZE_OW))

    global_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    global_offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    global_offs_oh = pid_oh * BLOCK_SIZE_OH + tl.arange(0, BLOCK_SIZE_OH)
    global_offs_ow = pid_ow * BLOCK_SIZE_OW + tl.arange(0, BLOCK_SIZE_OW)

    y_ptrs = y_ptr + (global_offs_n[:, None, None, None] * stride_yn +
                      global_offs_c[None, :, None, None] * stride_yc +
                      global_offs_oh[None, None, :, None] * stride_yh +
                      global_offs_ow[None, None, None, :] * stride_yw)

    y_mask = (global_offs_n[:, None, None, None] < N) & \
             (global_offs_c[None, :, None, None] < C) & \
             (global_offs_oh[None, None, :, None] < OH) & \
             (global_offs_ow[None, None, None, :] < OW)

    max_val_block = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_OH, BLOCK_SIZE_OW), 
                            value=float("-inf"), dtype=x_ptr.type.element_ty)

    for kh_idx in range(KH):
        for kw_idx in range(KW):
            current_h_in_window = global_offs_oh[None, None, :, None] * SH - PH_TOP + kh_idx
            current_w_in_window = global_offs_ow[None, None, None, :] * SW - PW_LEFT + kw_idx

            input_mask_h = (current_h_in_window >= 0) & (current_h_in_window < H)
            input_mask_w = (current_w_in_window >= 0) & (current_w_in_window < W)
            valid_input_mask_for_window = input_mask_h & input_mask_w
            
            load_mask_final = y_mask & valid_input_mask_for_window

            x_input_ptrs = x_ptr + (global_offs_n[:, None, None, None] * stride_xn +
                                    global_offs_c[None, :, None, None] * stride_xc +
                                    current_h_in_window * stride_xh +
                                    current_w_in_window * stride_xw)
            
            input_val = tl.load(x_input_ptrs, mask=load_mask_final, other=float("-inf"))
            
            max_val_block = tl.maximum(max_val_block, tl.where(load_mask_final, input_val, float("-inf")))

    tl.store(y_ptrs, max_val_block.to(y_ptr.type.element_ty), mask=y_mask)

def triton_maxpool2d(x, kernel_size: tuple, stride: tuple, padding: tuple = (0,0)): # x is torch.Tensor
    N, C, H, W = x.shape
    KH, KW = kernel_size
    SH, SW = stride
    
    PH_TOP = padding[0]
    PH_BOTTOM = padding[0]
    PW_LEFT = padding[1]
    PW_RIGHT = padding[1]

    OH = (H + PH_TOP + PH_BOTTOM - KH) // SH + 1
    OW = (W + PW_LEFT + PW_RIGHT - KW) // SW + 1
    
    y = empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    BLOCK_SIZE_N = 1 
    BLOCK_SIZE_C = min(16, triton.next_power_of_2(C)) 
    BLOCK_SIZE_OH = min(16, triton.next_power_of_2(OH))
    BLOCK_SIZE_OW = min(16, triton.next_power_of_2(OW))
    
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    grid_c = triton.cdiv(C, BLOCK_SIZE_C)
    grid_oh_blocks = triton.cdiv(OH, BLOCK_SIZE_OH)
    grid_ow_blocks = triton.cdiv(OW, BLOCK_SIZE_OW)
    
    grid = (grid_n, grid_c, grid_oh_blocks * grid_ow_blocks)

    _maxpool2d_forward_kernel[grid](
        x, y,
        N, C, H, W, KH, KW, SH, SW,
        PH_TOP, PH_BOTTOM, PW_LEFT, PW_RIGHT,
        OH, OW,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_yn, stride_yc, stride_yh, stride_yw,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_OH=BLOCK_SIZE_OH, BLOCK_SIZE_OW=BLOCK_SIZE_OW
    )
    return y