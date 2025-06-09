import triton
import triton.language as tl

# Ensure mgp is available
try:
    from mgp import empty
except ImportError:
    import torch as mgp_torch_fallback # Fallback for reasoning
    empty = mgp_torch_fallback.empty


@triton.jit
def _linear_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    bias_ptr, 
    stride_biasn
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M) # Use tl.minimum
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) # % M implicitly by mask
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) # % N implicitly by mask
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_loop_iter in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_loop_iter * BLOCK_SIZE_K
        a_mask = (offs_am[:, None] < M) & ((offs_k[None, :] + k_loop_iter * BLOCK_SIZE_K) < K)
        b_mask = ((offs_k[:, None] + k_loop_iter * BLOCK_SIZE_K) < K) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=True) # allow_tf32 might need to be configurable
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(c_ptr.type.element_ty)

    if ADD_BIAS:
        bias_vals = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        c += bias_vals[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_linear(a, b, bias = None): # a, b, bias are torch.Tensors
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, "Inner dimensions of A and B must match"
    K = K_a

    c = empty((M, N), device=a.device, dtype=a.dtype) 

    BLOCK_SIZE_M = 32 
    BLOCK_SIZE_N = 32 
    BLOCK_SIZE_K = 32 
    GROUP_SIZE_M = 8  

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    
    bias_ptr_arg = bias if bias is not None else None 
    stride_biasn_arg = bias.stride(0) if bias is not None else 0

    _linear_forward_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ADD_BIAS=(bias is not None),
        bias_ptr=bias_ptr_arg,
        stride_biasn=stride_biasn_arg
    )
    return c