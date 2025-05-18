#include "cuda_runtime.h"

#define CONV_DIRECT_OUTPUT_TILE_Y 8
#define CONV_DIRECT_OUTPUT_TILE_X 64
#define CONV_DIRECT_THREADS_Y 8
#define CONV_DIRECT_THREADS_X 64

#define MAX_KERNEL_SIDE 7

#define CONV_DIRECT_INPUT_TILE_H_MAX (CONV_DIRECT_OUTPUT_TILE_Y + MAX_KERNEL_SIDE - 1)
#define CONV_DIRECT_INPUT_TILE_W_MAX (CONV_DIRECT_OUTPUT_TILE_X + MAX_KERNEL_SIDE - 1)
#define CONV_DIRECT_WEIGHT_SHARED_MAX_ELEMENTS (MAX_KERNEL_SIDE * MAX_KERNEL_SIDE)

#define IM2COL_BLOCK_THREADS 512

#define MATMUL_M1_BLOCK_THREADS 512

#define MATMUL_TILED_DIM 32

__global__ void im2col_kernel(const float* __restrict__ x, int N_in, int C_in, int H_in, int W_in,
                              int kH, int kW, int padH, int padW,
                              int strideH, int strideW, int dilH, int dilW,
                              float* __restrict__ out, int H_out, int W_out) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_output_cols = H_out * W_out;

    if (col_idx >= num_output_cols) {
        return;
    }

    int patch_out_y = col_idx / W_out;
    int patch_out_x = col_idx % W_out;

    for (int c_channel = 0; c_channel < C_in; ++c_channel) {
        int h_offset_in_input = patch_out_y * strideH - padH;
        int w_offset_in_input = patch_out_x * strideW - padW;

        const float* x_channel_ptr = x + c_channel * H_in * W_in;

        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int in_y = h_offset_in_input + kh * dilH;
                int in_x = w_offset_in_input + kw * dilW;

                float val_to_write;
                if (in_y >= 0 && in_y < H_in && in_x >= 0 && in_x < W_in) {
                    val_to_write = x_channel_ptr[in_y * W_in + in_x];
                } else {
                    val_to_write = 0.0f;
                }
                out[( (c_channel * kH + kh) * kW + kw) * num_output_cols + col_idx] = val_to_write;
            }
        }
    }
}

void launch_im2col(const float* x, int N, int C, int H, int W,
                   int kH, int kW, int padH, int padW, int strideH, int strideW,
                   int dilH, int dilW, float* out)
{
    int H_out_calc = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    int W_out_calc = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

    if (H_out_calc <= 0 || W_out_calc <=0) return;

    int num_output_cols = H_out_calc * W_out_calc;

    if (num_output_cols == 0) return;

    dim3 threadsPerBlock(IM2COL_BLOCK_THREADS);
    dim3 numBlocks((num_output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

    im2col_kernel<<<numBlocks, threadsPerBlock>>>(
        x, N, C, H, W, kH, kW, padH, padW, strideH, strideW, dilH, dilW,
        out, H_out_calc, W_out_calc
    );
}

__global__ void matmul_kernel_optimized_M1_shared_A(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                           int N_dim, int K_dim) {
    __shared__ float A_s[MATMUL_M1_BLOCK_THREADS];


    for(int i = threadIdx.x; i < K_dim; i += blockDim.x) {
        A_s[i] = A[i];
    }
    __syncthreads();

    int col_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (col_c >= N_dim) {
        return;
    }

    float c_val = 0.0f;
    for (int k = 0; k < K_dim; ++k) {
        c_val += A_s[k] * B[k * N_dim + col_c];
    }
    C[col_c] = c_val;
}

__global__ void matmul_kernel_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                    int M_dim, int N_dim, int K_dim) {
    __shared__ float As[MATMUL_TILED_DIM][MATMUL_TILED_DIM];
    __shared__ float Bs[MATMUL_TILED_DIM][MATMUL_TILED_DIM];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row_c = by * MATMUL_TILED_DIM + ty;
    int col_c = bx * MATMUL_TILED_DIM + tx;
    float c_val = 0.0f;

    for (int t = 0; t < (K_dim + MATMUL_TILED_DIM - 1) / MATMUL_TILED_DIM; ++t) {
        int a_load_row = row_c;
        int a_load_col = t * MATMUL_TILED_DIM + tx;
        if (a_load_row < M_dim && a_load_col < K_dim) {
            As[ty][tx] = A[a_load_row * K_dim + a_load_col];
        } else { As[ty][tx] = 0.0f; }

        int b_load_row = t * MATMUL_TILED_DIM + ty;
        int b_load_col = col_c;
        if (b_load_row < K_dim && b_load_col < N_dim) {
            Bs[ty][tx] = B[b_load_row * N_dim + b_load_col];
        } else { Bs[ty][tx] = 0.0f; }
        __syncthreads();

        for (int k_inner = 0; k_inner < MATMUL_TILED_DIM; ++k_inner) {
            c_val += As[ty][k_inner] * Bs[k_inner][tx];
        }
        __syncthreads();
    }
    if (row_c < M_dim && col_c < N_dim) {
        C[row_c * N_dim + col_c] = c_val;
    }
}

void launch_matmul(const float* A, const float* B, float* C, int M, int N_param, int K_param)
{
    if (M == 0 || N_param == 0 || K_param == 0) {
         if (M > 0 && N_param > 0) { 

         }
         return; 
    }

    if (M == 1) {

        dim3 threadsPerBlock_M1(MATMUL_M1_BLOCK_THREADS);
        dim3 numBlocks_M1((N_param + threadsPerBlock_M1.x - 1) / threadsPerBlock_M1.x);
        matmul_kernel_optimized_M1_shared_A<<<numBlocks_M1, threadsPerBlock_M1>>>(A, B, C, N_param, K_param);
    } else {
        dim3 threadsPerBlock(MATMUL_TILED_DIM, MATMUL_TILED_DIM);
        dim3 numBlocks((N_param + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_kernel_tiled<<<numBlocks, threadsPerBlock>>>(A, B, C, M, N_param, K_param);
    }
}

__global__ void conv2d_direct_optimized_final_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
    int N_in, int C_in, int H_in, int W_in,
    int K_filters, int kH, int kW,
    int padH, int padW, int strideH, int strideW,
    int dilH, int dilW,
    int H_out, int W_out)
{
    __shared__ float x_s[CONV_DIRECT_INPUT_TILE_H_MAX][CONV_DIRECT_INPUT_TILE_W_MAX];
    __shared__ float w_s[MAX_KERNEL_SIDE][MAX_KERNEL_SIDE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_tile_y_base_block = blockIdx.y * CONV_DIRECT_OUTPUT_TILE_Y;
    int out_tile_x_base_block = blockIdx.x * CONV_DIRECT_OUTPUT_TILE_X;

    float p_val = 0.0f;

    for (int k_filter_idx = 0; k_filter_idx < K_filters; ++k_filter_idx) {
        const float* w_current_filter = w + k_filter_idx * C_in * kH * kW;
        p_val = 0.0f; 

        for (int c_channel = 0; c_channel < C_in; ++c_channel) {
            const float* x_current_channel = x + c_channel * H_in * W_in;
            const float* w_current_channel_in_filter = w_current_filter + c_channel * kH * kW;

            int thread_1d_idx_in_block = ty * CONV_DIRECT_THREADS_X + tx;
            if (thread_1d_idx_in_block < kH * kW) {
                 w_s[thread_1d_idx_in_block / kW][thread_1d_idx_in_block % kW] = w_current_channel_in_filter[thread_1d_idx_in_block];
            }

            int input_tile_effective_h = (CONV_DIRECT_OUTPUT_TILE_Y -1) * strideH + dilH * (kH-1) + 1;
            int input_tile_effective_w = (CONV_DIRECT_OUTPUT_TILE_X -1) * strideW + dilW * (kW-1) + 1;


            int input_load_y_start_in_image = out_tile_y_base_block * strideH - padH;
            int input_load_x_start_in_image = out_tile_x_base_block * strideW - padW;

            for (int r_s = ty; r_s < input_tile_effective_h; r_s += CONV_DIRECT_THREADS_Y) {
                int global_y = input_load_y_start_in_image + r_s;
                 for (int c_s = tx; c_s < input_tile_effective_w; c_s += CONV_DIRECT_THREADS_X) {
                    int global_x = input_load_x_start_in_image + c_s;
                    if (global_y >= 0 && global_y < H_in && global_x >=0 && global_x < W_in) {
                        x_s[r_s][c_s] = x_current_channel[global_y * W_in + global_x];
                    } else {
                        x_s[r_s][c_s] = 0.0f;
                    }
                }
            }
            __syncthreads();

            int out_y_local = ty;
            int out_x_local = tx;

            int out_y_global = out_tile_y_base_block + out_y_local;
            int out_x_global = out_tile_x_base_block + out_x_local;

            if (out_y_global < H_out && out_x_global < W_out) {
                float current_channel_sum = 0.0f;
                #pragma unroll
                for (int kh_f = 0; kh_f < kH; ++kh_f) {
                    #pragma unroll
                    for (int kw_f = 0; kw_f < kW; ++kw_f) {
                         int x_s_y_idx = out_y_local * strideH + kh_f * dilH;
                         int x_s_x_idx = out_x_local * strideW + kw_f * dilW;
                         current_channel_sum += x_s[x_s_y_idx][x_s_x_idx] * w_s[kh_f][kw_f];
                    }
                }
                p_val += current_channel_sum;
            }
             __syncthreads();
        }
        int out_y_global = out_tile_y_base_block + ty;
        int out_x_global = out_tile_x_base_block + tx;
        if (out_y_global < H_out && out_x_global < W_out) {
             y[(k_filter_idx * H_out + out_y_global) * W_out + out_x_global] = p_val;
        }
    }
}


void launch_conv2d_direct(const float* x, const float* w, float* y,
                          int N, int C, int H, int W,
                          int K_filters, int kH, int kW,
                          int padH, int padW, int strideH, int strideW,
                          int dilH, int dilW)
{
    int H_out_calc = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    int W_out_calc = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

    if (H_out_calc <= 0 || W_out_calc <=0 || K_filters == 0) return;

    dim3 threadsPerBlock(CONV_DIRECT_THREADS_X, CONV_DIRECT_THREADS_Y);

    dim3 numBlocks( (W_out_calc + CONV_DIRECT_OUTPUT_TILE_X - 1) / CONV_DIRECT_OUTPUT_TILE_X,
                    (H_out_calc + CONV_DIRECT_OUTPUT_TILE_Y - 1) / CONV_DIRECT_OUTPUT_TILE_Y );

    conv2d_direct_optimized_final_kernel<<<numBlocks, threadsPerBlock>>>(
        x, w, y, N, C, H, W, K_filters, kH, kW,
        padH, padW, strideH, strideW, dilH, dilW,
        H_out_calc, W_out_calc
    );
}
