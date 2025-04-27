#include "lora.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void load_txt(const std::string &path, std::vector<float> &data,
              int expected_size) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Cannot open: " << path << std::endl;
    exit(1);
  }
  float val;
  while (file >> val)
    data.push_back(val);
  if (data.size() != expected_size) {
    std::cerr << "Unexpected size in " << path << ": got " << data.size()
              << ", expected " << expected_size << std::endl;
    exit(1);
  }
}

float max_abs_error(const std::vector<float> &ref,
                    const std::vector<float> &out) {
  assert(ref.size() == out.size());
  float max_err = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    float err = std::fabs(ref[i] - out[i]);
    if (err > max_err)
      max_err = err;
  }
  return max_err;
}

int main() {
  int gpu_count = 0;
  cudaGetDeviceCount(&gpu_count);

  if (gpu_count == 0) {
    std::cerr << "No CUDA devices found.\n";
    return 1;
  }

  // Random Selection of GPU
  std::srand(std::time(nullptr));
  int selected = std::rand() % gpu_count;

  cudaSetDevice(selected);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, selected);
  std::cout << "Using GPU " << selected << ": " << prop.name << std::endl;

  const int BATCH = 32, IN_DIM = 4096, OUT_DIM = 4096, RANK = 8;
  const float alpha = 16.0f;
  const float scale = alpha / RANK;

  // Load data
  std::vector<float> h_x, h_W, h_A, h_B, h_ref;
  load_txt("/data/hw3/x.txt", h_x, BATCH * IN_DIM);
  load_txt("/data/hw3/W.txt", h_W, OUT_DIM * IN_DIM);
  load_txt("/data/hw3/A.txt", h_A, RANK * IN_DIM);
  load_txt("/data/hw3/B.txt", h_B, OUT_DIM * RANK);
  load_txt("/data/hw3/y.txt", h_ref, BATCH * OUT_DIM);

  // Allocate GPU memory
  float *d_x, *d_W, *d_A, *d_B, *d_y;
  CHECK_CUDA(cudaMalloc(&d_x, BATCH * IN_DIM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_W, OUT_DIM * IN_DIM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_A, RANK * IN_DIM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, OUT_DIM * RANK * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, BATCH * OUT_DIM * sizeof(float)));

  // Copy to device
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), BATCH * IN_DIM * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), OUT_DIM * IN_DIM * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), RANK * IN_DIM * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), OUT_DIM * RANK * sizeof(float),
                        cudaMemcpyHostToDevice));

  // CUDA timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Launch
  CHECK_CUDA(cudaEventRecord(start));
  lora(d_x, d_W, d_A, d_B, d_y, BATCH, IN_DIM, OUT_DIM, RANK, scale);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  printf("LoRA kernel time: %.3f ms\n", elapsed_ms);

  // Copy result
  std::vector<float> h_out(BATCH * OUT_DIM);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_y, BATCH * OUT_DIM * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Compare
  float err = max_abs_error(h_ref, h_out);
  printf("Max abs error vs y.txt: %.6f\n", err);

  // Cleanup
  cudaFree(d_x);
  cudaFree(d_W);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_y);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
