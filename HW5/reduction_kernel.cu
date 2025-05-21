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
    for (int i = 0; i < n; i++)
        g_odata[0] += g_idata[i];
}


void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, int* const d_odata, const int n) {
    // TODO: Implement your CUDA code
    // Reduction result must be stored in d_odata[0] 
    // You should run the best kernel in here but you must remain other kernels as evidence.
}
