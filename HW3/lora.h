#ifndef LORA_H
#define LORA_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_DIM 16

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float val = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        int tiledRow = t * TILE_DIM + threadIdx.x;
        int tiledCol = t * TILE_DIM + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && tiledRow < K) ? A[row * K + tiledRow] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && tiledCol < K) ? B[col * K + tiledCol] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

__global__ void add_scaled_kernel(const float* A, const float* B, float* C, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + scale * B[idx];
    }
}

void lora(float* d_x, float* d_W, float* d_A, float* d_B, float* d_y,
          int BATCH, int IN_DIM, int OUT_DIM, int RANK, float scale) {
    float *d_linear, *d_tmp, *d_lora;
    size_t out_size = BATCH * OUT_DIM * sizeof(float);
    size_t tmp_size = BATCH * RANK * sizeof(float);

    cudaMalloc(&d_linear, out_size);  // x @ W.T
    cudaMalloc(&d_tmp, tmp_size);     // x @ A.T
    cudaMalloc(&d_lora, out_size);    // (x @ A.T) @ B.T

    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 grid_linear((OUT_DIM + TILE_DIM - 1) / TILE_DIM, (BATCH + TILE_DIM - 1) / TILE_DIM);
    dim3 grid_tmp((RANK + TILE_DIM - 1) / TILE_DIM, (BATCH + TILE_DIM - 1) / TILE_DIM);

    // 1. out_linear = x @ W.T
    matmul_shared_kernel<<<grid_linear, blockSize>>>(d_x, d_W, d_linear, BATCH, OUT_DIM, IN_DIM);

    // 2. tmp = x @ A.T
    matmul_shared_kernel<<<grid_tmp, blockSize>>>(d_x, d_A, d_tmp, BATCH, RANK, IN_DIM);

    // 3. out_lora = tmp @ B.T
    matmul_shared_kernel<<<grid_linear, blockSize>>>(d_tmp, d_B, d_lora, BATCH, OUT_DIM, RANK);

    // 4. y = out_linear + scale * out_lora
    int total = BATCH * OUT_DIM;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_scaled_kernel<<<blocks, threads>>>(d_linear, d_lora, d_y, total, scale);

    cudaFree(d_linear);
    cudaFree(d_tmp);
    cudaFree(d_lora);
}

#endif
