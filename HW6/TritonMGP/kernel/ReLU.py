import triton
import triton.language as tl
from mgp import empty

@triton.jit
def _relu_forward_kernel(
    in_ptr,
    out_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(in_ptr + offsets, mask=offsets < num_elements, other=0.)
    a = tl.where(a > 0, a, 0)
    tl.store(out_ptr + offsets, a, mask=offsets < num_elements)
    
    
def triton_relu(x, inplace=False):
    """
    Applies the ReLU (Rectified Linear Unit) function element-wise to the input tensor `x` using Triton for GPU acceleration.
    Args:
        x (torch.Tensor): The input tensor on which to apply the ReLU function.
        inplace (bool, optional): If True, modifies the input tensor in-place, otherwise creates a new tensor. Default is False.
    Returns:
        torch.Tensor: The tensor after applying the ReLU function.
    """
    
    if inplace:
        out = x
    else:
        out = empty(x.shape, device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    def grid(meta): return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _relu_forward_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out