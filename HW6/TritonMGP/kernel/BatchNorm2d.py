import triton
import triton.language as tl

# Ensure mgp is available
try:
    from mgp import empty
except ImportError:
    import torch as mgp_torch_fallback # Fallback for reasoning
    empty = mgp_torch_fallback.empty

@triton.jit
def _bn2d_forward_inference_kernel(
    x_ptr, y_ptr,
    mean_ptr, var_ptr, 
    weight_ptr, bias_ptr, 
    N, C, H, W,
    eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_mean_c, stride_var_c, 
    stride_weight_c, stride_bias_c, 
    BLOCK_SIZE_N_GROUP: tl.constexpr, # How many N items are processed by one program ID on axis 0
    BLOCK_SIZE_H_PIXELS: tl.constexpr, # How many H pixels a program handles
    BLOCK_SIZE_W_PIXELS: tl.constexpr  # How many W pixels a program handles
):
    # This kernel processes a slice of (N, H, W) for a single channel C.
    # pid_n_group_idx: iterates over groups of N.
    # pid_c_idx: iterates over channels C.
    # pid_spatial_idx: iterates over spatial blocks of HxW.
    
    pid_n_group_idx = tl.program_id(0)
    pid_c_idx = tl.program_id(1)
    pid_spatial_idx = tl.program_id(2)

    # Calculate current N offsets for this program
    n_offsets = pid_n_group_idx * BLOCK_SIZE_N_GROUP + tl.arange(0, BLOCK_SIZE_N_GROUP)

    # Calculate current H, W offsets for this program
    num_h_blocks = tl.cdiv(H, BLOCK_SIZE_H_PIXELS)
    # num_w_blocks = tl.cdiv(W, BLOCK_SIZE_W_PIXELS) # Not directly needed for decomposition

    h_block_idx = pid_spatial_idx // num_h_blocks
    w_block_idx = pid_spatial_idx % num_h_blocks # Corrected: should be num_w_blocks for modulus if defined that way
                                                # Or, if pid_spatial_idx covers all HxW blocks:
                                                # num_w_pixel_blocks = tl.cdiv(W, BLOCK_SIZE_W_PIXELS)
                                                # h_block_idx = pid_spatial_idx // num_w_pixel_blocks
                                                # w_block_idx = pid_spatial_idx % num_w_pixel_blocks

    # Let's use a simpler spatial decomposition if pid_spatial_idx covers all HxW blocks
    # This assumes grid for axis 2 is cdiv(H*W, BLOCK_SIZE_H_PIXELS*BLOCK_SIZE_W_PIXELS)
    # Or, if grid axis 2 is cdiv(H,H_BLOCK) * cdiv(W,W_BLOCK)
    # The current structure implies grid axis 2 is for HxW blocks.
    
    h_offsets = h_block_idx * BLOCK_SIZE_H_PIXELS + tl.arange(0, BLOCK_SIZE_H_PIXELS)
    w_offsets = w_block_idx * BLOCK_SIZE_W_PIXELS + tl.arange(0, BLOCK_SIZE_W_PIXELS)
    
    # Input pointers: x_ptrs[N_GROUP, H_PIXELS, W_PIXELS]
    x_ptrs = x_ptr + (n_offsets[:, None, None] * stride_xn +
                      pid_c_idx * stride_xc +
                      h_offsets[None, :, None] * stride_xh +
                      w_offsets[None, None, :] * stride_xw)

    y_ptrs = y_ptr + (n_offsets[:, None, None] * stride_yn +
                      pid_c_idx * stride_yc +
                      h_offsets[None, :, None] * stride_yh +
                      w_offsets[None, None, :] * stride_yw)

    mask_n = n_offsets[:, None, None] < N
    mask_c = pid_c_idx < C # Single channel for this program
    mask_h = h_offsets[None, :, None] < H
    mask_w = w_offsets[None, None, :] < W
    
    load_store_mask = mask_n & mask_c & mask_h & mask_w

    x_block = tl.load(x_ptrs, mask=load_store_mask, other=0.0)

    # Load params for current channel pid_c_idx
    mean_val = tl.load(mean_ptr + pid_c_idx * stride_mean_c, mask=mask_c)
    var_val = tl.load(var_ptr + pid_c_idx * stride_var_c, mask=mask_c)
    
    weight_val = 1.0 # Default for affine=False
    if weight_ptr is not None:
        weight_val = tl.load(weight_ptr + pid_c_idx * stride_weight_c, mask=mask_c)
        
    bias_val = 0.0 # Default for affine=False
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_c_idx * stride_bias_c, mask=mask_c)

    x_normalized = (x_block - mean_val) / tl.sqrt(var_val + eps)
    y_block = weight_val * x_normalized + bias_val
    
    tl.store(y_ptrs, y_block, mask=load_store_mask)


def triton_bn2d(x, weight, bias, running_mean, running_var, momentum, eps): # Types are torch.Tensor
    N, C, H, W = x.shape
    y = empty(x.shape, device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()
    
    stride_mean_c = running_mean.stride(0) if running_mean is not None else 0
    stride_var_c = running_var.stride(0) if running_var is not None else 0
    stride_weight_c = weight.stride(0) if weight is not None else 0
    stride_bias_c = bias.stride(0) if bias is not None else 0

    BLOCK_SIZE_N_GROUP = min(4, triton.next_power_of_2(N)) 
    BLOCK_SIZE_H_PIXELS = min(32, triton.next_power_of_2(H))
    BLOCK_SIZE_W_PIXELS = min(32, triton.next_power_of_2(W))

    grid_n_groups = triton.cdiv(N, BLOCK_SIZE_N_GROUP)
    grid_c_channels = C 
    
    num_h_pixel_blocks = triton.cdiv(H, BLOCK_SIZE_H_PIXELS)
    num_w_pixel_blocks = triton.cdiv(W, BLOCK_SIZE_W_PIXELS)
    grid_spatial_blocks = num_h_pixel_blocks * num_w_pixel_blocks

    grid = (grid_n_groups, grid_c_channels, grid_spatial_blocks)
    
    _bn2d_forward_inference_kernel[grid](
        x, y,
        running_mean, running_var,
        weight, bias,
        N, C, H, W, eps,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_yn, stride_yc, stride_yh, stride_yw,
        stride_mean_c, stride_var_c,
        stride_weight_c, stride_bias_c,
        BLOCK_SIZE_N_GROUP=BLOCK_SIZE_N_GROUP,
        BLOCK_SIZE_H_PIXELS=BLOCK_SIZE_H_PIXELS,
        BLOCK_SIZE_W_PIXELS=BLOCK_SIZE_W_PIXELS
    )
    return y