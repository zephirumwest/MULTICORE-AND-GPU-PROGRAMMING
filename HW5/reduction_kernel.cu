#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "reduction.h"

void allocateDeviceMemory(void** M, int size)
{
    cudaError_t err = cudaMalloc(M, size);
    assert(err==cudaSuccess);
}

void deallocateDeviceMemory(void* M)
{
    cudaError_t err = cudaFree(M);
    assert(err==cudaSuccess);
}

void cudaMemcpyToDevice(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyHostToDevice);
    assert(err==cudaSuccess);
}

void cudaMemcpyToHost(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToHost);
    assert(err==cudaSuccess);
}

void reduce_ref(const int* const g_idata, int* const g_odata, const int n) {
    g_odata[0] = 0;
    for (int i = 0; i < n; i++)
        g_odata[0] += g_idata[i];
}

#define REDUCE_OPTIMIZE_CUDA_CHECK(err)                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in reduce_optimize %s at %s:%d\n",     \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static __global__ void kernel1_interleaved_divergent_opt(const int *g_idata, int *g_odata, unsigned int n_elements_in_pass) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_elements_in_pass) {
        sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            if (tid + s < blockDim.x) {
                 sdata[tid] += sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

static __global__ void kernel2_interleaved_nondivergent_opt(const int *g_idata, int *g_odata, unsigned int n_elements_in_pass) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_elements_in_pass) {
        sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            if (index + s < blockDim.x) {
                sdata[index] += sdata[index + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

static __global__ void kernel3_sequential_addressing_opt(const int *g_idata, int *g_odata, unsigned int n_elements_in_pass) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_elements_in_pass) {
        sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

static __global__ void kernel4_first_add_during_load_opt(const int *g_idata, int *g_odata, unsigned int n_elements_in_pass) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i1 = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int i2 = i1 + blockDim.x;

    if (i1 < n_elements_in_pass && i2 < n_elements_in_pass) {
        sdata[tid] = g_idata[i1] + g_idata[i2];
    } else if (i1 < n_elements_in_pass) {
        sdata[tid] = g_idata[i1];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

static __global__ void kernel5_unroll_last_warp_opt(const int *g_idata, int *g_odata, unsigned int n_elements_in_pass) {
    extern __shared__ volatile int sdata_volatile[];

    unsigned int tid = threadIdx.x;
    unsigned int i1 = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int i2 = i1 + blockDim.x;

    if (i1 < n_elements_in_pass && i2 < n_elements_in_pass) {
        sdata_volatile[tid] = g_idata[i1] + g_idata[i2];
    } else if (i1 < n_elements_in_pass) {
        sdata_volatile[tid] = g_idata[i1];
    } else {
        sdata_volatile[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata_volatile[tid] += sdata_volatile[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (blockDim.x >= 64 && tid < 32) sdata_volatile[tid] += sdata_volatile[tid + 32]; // tid < 32 is redundant here
        if (blockDim.x >= 32 && tid < 16) sdata_volatile[tid] += sdata_volatile[tid + 16];
        if (blockDim.x >= 16 && tid < 8)  sdata_volatile[tid] += sdata_volatile[tid + 8];
        if (blockDim.x >= 8  && tid < 4)  sdata_volatile[tid] += sdata_volatile[tid + 4];
        if (blockDim.x >= 4  && tid < 2)  sdata_volatile[tid] += sdata_volatile[tid + 2];
        if (blockDim.x >= 2  && tid < 1)  sdata_volatile[tid] += sdata_volatile[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata_volatile[0];
    }
}

template <unsigned int blockSize>
static __device__ void warpReduce_k6_opt(volatile int* sdata_param, int tid) {
    if (blockSize >= 64) sdata_param[tid] += sdata_param[tid + 32];
    if (blockSize >= 32) sdata_param[tid] += sdata_param[tid + 16];
    if (blockSize >= 16) sdata_param[tid] += sdata_param[tid + 8];
    if (blockSize >= 8)  sdata_param[tid] += sdata_param[tid + 4];
    if (blockSize >= 4)  sdata_param[tid] += sdata_param[tid + 2];
    if (blockSize >= 2)  sdata_param[tid] += sdata_param[tid + 1];
}

template <unsigned int blockSize>
static __global__ void kernel6_completely_unrolled_opt(const int *g_idata, int *g_odata, unsigned int n_elements_in_pass) {
    extern __shared__ volatile int sdata_volatile[];

    unsigned int tid = threadIdx.x;
    unsigned int i1 = blockIdx.x * (blockSize * 2) + tid;
    unsigned int i2 = i1 + blockSize;

    if (i1 < n_elements_in_pass && i2 < n_elements_in_pass) {
        sdata_volatile[tid] = g_idata[i1] + g_idata[i2];
    } else if (i1 < n_elements_in_pass) {
        sdata_volatile[tid] = g_idata[i1];
    } else {
        sdata_volatile[tid] = 0;
    }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata_volatile[tid] += sdata_volatile[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) { sdata_volatile[tid] += sdata_volatile[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) { sdata_volatile[tid] += sdata_volatile[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128)  { if (tid <  64) { sdata_volatile[tid] += sdata_volatile[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        warpReduce_k6_opt<blockSize>(sdata_volatile, tid);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata_volatile[0];
    }
}

template <unsigned int blockSize>
static __global__ void kernel7_final_optimized_opt(const int *g_idata, int *g_odata, const int n_total_elements_in_kernel) {
    extern __shared__ volatile int sdata_volatile[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize_elements_processed_per_block_per_iteration = gridDim.x * blockSize * 2;

    sdata_volatile[tid] = 0;

    while (i < n_total_elements_in_kernel) {
        sdata_volatile[tid] += g_idata[i];
        if (i + blockSize < n_total_elements_in_kernel) {
            sdata_volatile[tid] += g_idata[i + blockSize];
        }
        i += gridSize_elements_processed_per_block_per_iteration;
    }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata_volatile[tid] += sdata_volatile[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) { sdata_volatile[tid] += sdata_volatile[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) { sdata_volatile[tid] += sdata_volatile[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128)  { if (tid <  64) { sdata_volatile[tid] += sdata_volatile[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        if (blockSize >= 64) sdata_volatile[tid] += sdata_volatile[tid + 32];
        if (blockSize >= 32) sdata_volatile[tid] += sdata_volatile[tid + 16];
        if (blockSize >= 16) sdata_volatile[tid] += sdata_volatile[tid + 8];
        if (blockSize >=  8) sdata_volatile[tid] += sdata_volatile[tid + 4];
        if (blockSize >=  4) sdata_volatile[tid] += sdata_volatile[tid + 2];
        if (blockSize >=  2) sdata_volatile[tid] += sdata_volatile[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata_volatile[0];
    }
}

void reduce_optimize(const int* const g_idata,
                     int* const g_odata,
                     const int* d_idata,
                     int* d_odata,
                     const int n) {

    if (n == 0) {
        int zero_val = 0;
        REDUCE_OPTIMIZE_CUDA_CHECK(cudaMemcpy(d_odata, &zero_val, sizeof(int), cudaMemcpyHostToDevice));
        return;
    }
    if (n == 1) {
        REDUCE_OPTIMIZE_CUDA_CHECK(cudaMemcpy(d_odata, d_idata, sizeof(int), cudaMemcpyDeviceToDevice));
        return;
    }

    const unsigned int k7_chosen_blockSize = 256;
    const unsigned int trace_blockSize = 256; 
    size_t trace_shared_mem_int = trace_blockSize * sizeof(int);
    size_t trace_shared_mem_volatile_int = trace_blockSize * sizeof(volatile int);


    /*
    // Kernel 1 
    {
        unsigned int num_blocks = (n + trace_blockSize - 1) / trace_blockSize;
        int* target_output_k1 = (num_blocks == 1) ? d_odata : nullptr;
        if (target_output_k1) { 
            kernel1_interleaved_divergent_opt<<<num_blocks, trace_blockSize, trace_shared_mem_int>>>(
                d_idata, target_output_k1, n);
            REDUCE_OPTIMIZE_CUDA_CHECK(cudaGetLastError());
        }
    }
    
    // Kernel 2 
    {
        unsigned int num_blocks = (n + trace_blockSize - 1) / trace_blockSize;
        int* target_output_k2 = (num_blocks == 1) ? d_odata : nullptr;
        if (target_output_k2) {
             kernel2_interleaved_nondivergent_opt<<<num_blocks, trace_blockSize, trace_shared_mem_int>>>(
                d_idata, target_output_k2, n);
            REDUCE_OPTIMIZE_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Kernel 3 
    {
        unsigned int num_blocks = (n + trace_blockSize - 1) / trace_blockSize;
        int* target_output_k3 = (num_blocks == 1) ? d_odata : nullptr;
        if (target_output_k3) {
            kernel3_sequential_addressing_opt<<<num_blocks, trace_blockSize, trace_shared_mem_int>>>(
                d_idata, target_output_k3, n);
            REDUCE_OPTIMIZE_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Kernel 4 
    {
        unsigned int elements_per_block = trace_blockSize * 2;
        unsigned int num_blocks = (n + elements_per_block - 1) / elements_per_block;
        int* target_output_k4 = (num_blocks == 1) ? d_odata : nullptr;
        if (target_output_k4) {
            kernel4_first_add_during_load_opt<<<num_blocks, trace_blockSize, trace_shared_mem_int>>>(
                d_idata, target_output_k4, n);
            REDUCE_OPTIMIZE_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Kernel 5 
    {
        unsigned int elements_per_block = trace_blockSize * 2;
        unsigned int num_blocks = (n + elements_per_block - 1) / elements_per_block;
        int* target_output_k5 = (num_blocks == 1) ? d_odata : nullptr;
        if (target_output_k5) {
            kernel5_unroll_last_warp_opt<<<num_blocks, trace_blockSize, trace_shared_mem_volatile_int>>>(
                d_idata, target_output_k5, n);
            REDUCE_OPTIMIZE_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Kernel 6 
    {
        const unsigned int k6_bs = trace_blockSize;
        unsigned int elements_per_block = k6_bs * 2;
        unsigned int num_blocks = (n + elements_per_block - 1) / elements_per_block;
        int* target_output_k6 = (num_blocks == 1) ? d_odata : nullptr;

        if (target_output_k6) {
            if (k6_bs == 64) {
                 kernel6_completely_unrolled_opt<64><<<num_blocks, k6_bs, trace_shared_mem_volatile_int>>>(
                     d_idata, target_output_k6, n);
            } else if (k6_bs == 128) {
                 kernel6_completely_unrolled_opt<128><<<num_blocks, k6_bs, trace_shared_mem_volatile_int>>>(
                     d_idata, target_output_k6, n);
            } else if (k6_bs == 256) {
                 kernel6_completely_unrolled_opt<256><<<num_blocks, k6_bs, trace_shared_mem_volatile_int>>>(
                     d_idata, target_output_k6, n);
            } else if (k6_bs == 512) {
                 kernel6_completely_unrolled_opt<512><<<num_blocks, k6_bs, trace_shared_mem_volatile_int>>>(
                     d_idata, target_output_k6, n);
            } else if (k6_bs == 1024) {
                 kernel6_completely_unrolled_opt<1024><<<num_blocks, k6_bs, trace_shared_mem_volatile_int>>>(
                     d_idata, target_output_k6, n);
            }
            REDUCE_OPTIMIZE_CUDA_CHECK(cudaGetLastError());
        }
    }
    */
    
    
    const int* d_current_pass_input = d_idata;
    int* d_current_pass_output = nullptr;
    int* d_temp_buffer = nullptr;
    int current_elements_to_reduce = n;
    bool is_first_pass_for_temp_alloc = true;
    size_t k7_shared_mem_size = k7_chosen_blockSize * sizeof(volatile int);


    auto internal_allocate = [&](void** ptr, size_t size) {
        REDUCE_OPTIMIZE_CUDA_CHECK(cudaMalloc(ptr, size));
    };
    auto internal_free = [&](void* ptr) {
        if (ptr) {
            REDUCE_OPTIMIZE_CUDA_CHECK(cudaFree(ptr));
        }
    };

    while (current_elements_to_reduce > 1) {
        unsigned int elements_processed_per_block_k7 = k7_chosen_blockSize * 2;
        unsigned int num_blocks_for_this_pass = (current_elements_to_reduce + elements_processed_per_block_k7 - 1) / elements_processed_per_block_k7;

        if (num_blocks_for_this_pass == 0 && current_elements_to_reduce > 0) {
            num_blocks_for_this_pass = 1;
        }

        if (num_blocks_for_this_pass == 1) {
            d_current_pass_output = d_odata;
        } else {
            if (is_first_pass_for_temp_alloc) {
                internal_allocate((void**)&d_temp_buffer, num_blocks_for_this_pass * sizeof(int));
                d_current_pass_output = d_temp_buffer;
                is_first_pass_for_temp_alloc = false;
            } else {
                d_current_pass_output = d_temp_buffer;
            }
        }

        if (d_current_pass_input == d_current_pass_output && num_blocks_for_this_pass > 1) {
        }


        switch (k7_chosen_blockSize) {
            case 64:
                kernel7_final_optimized_opt<64><<<num_blocks_for_this_pass, k7_chosen_blockSize, k7_shared_mem_size>>>(
                    d_current_pass_input, d_current_pass_output, current_elements_to_reduce);
                break;
            case 128:
                kernel7_final_optimized_opt<128><<<num_blocks_for_this_pass, k7_chosen_blockSize, k7_shared_mem_size>>>(
                    d_current_pass_input, d_current_pass_output, current_elements_to_reduce);
                break;
            case 256:
                kernel7_final_optimized_opt<256><<<num_blocks_for_this_pass, k7_chosen_blockSize, k7_shared_mem_size>>>(
                    d_current_pass_input, d_current_pass_output, current_elements_to_reduce);
                break;
            case 512:
                kernel7_final_optimized_opt<512><<<num_blocks_for_this_pass, k7_chosen_blockSize, k7_shared_mem_size>>>(
                    d_current_pass_input, d_current_pass_output, current_elements_to_reduce);
                break;
            case 1024:
                kernel7_final_optimized_opt<1024><<<num_blocks_for_this_pass, k7_chosen_blockSize, k7_shared_mem_size>>>(
                    d_current_pass_input, d_current_pass_output, current_elements_to_reduce);
                break;
            default:
                fprintf(stderr, "reduce_optimize: Unsupported k7_chosen_blockSize %u\n", k7_chosen_blockSize);
                exit(EXIT_FAILURE);
        }
        REDUCE_OPTIMIZE_CUDA_CHECK(cudaGetLastError());
        REDUCE_OPTIMIZE_CUDA_CHECK(cudaDeviceSynchronize());


        d_current_pass_input = d_current_pass_output;
        current_elements_to_reduce = num_blocks_for_this_pass;
    }
    

    if (d_current_pass_input != d_odata && current_elements_to_reduce == 1) {
        REDUCE_OPTIMIZE_CUDA_CHECK(cudaMemcpy(d_odata, d_current_pass_input, sizeof(int), cudaMemcpyDeviceToDevice));
    } else if (current_elements_to_reduce != 1 && n > 0) {
        int error_val = 0; 
        REDUCE_OPTIMIZE_CUDA_CHECK(cudaMemcpy(d_odata, &error_val, sizeof(int), cudaMemcpyHostToDevice));
    }


    if (d_temp_buffer) {
        internal_free(d_temp_buffer);
    }
    
}
