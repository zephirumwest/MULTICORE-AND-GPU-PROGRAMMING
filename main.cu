#include "conv.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static bool load_binary(const std::string &path, float *dst, size_t count) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    printf("[ERR] cannot open %s\n", path.c_str());
    return false;
  }
  ifs.read(reinterpret_cast<char *>(dst), count * sizeof(float));
  if (!ifs) {
    printf("[ERR] read failed %s\n", path.c_str());
    return false;
  }
  return true;
}

int main() {
  int gpu_count = 0;
  cudaGetDeviceCount(&gpu_count);

  if (gpu_count == 0) {
    std::cerr << "No CUDA devices found.\n";
    return 1;
  }

  // 시드 설정 및 무작위 인덱스 선택
  std::srand(std::time(nullptr));
  int selected = std::rand() % gpu_count;

  // GPU 설정
  cudaSetDevice(selected);

  // 확인
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, selected);
  std::cout << "Using GPU " << selected << ": " << prop.name << std::endl;

  ///////////////////////////////////////////////////////////////////////////////////
  const int N = 1, C = 1, H = 2048, W = 2048, K = 1;
  const int stride = 1, pad = 0, dilation = 1;
  const char *img_file = "/data/hw4/mat_2048x2048.txt";
  const char *filt_files[2] = {"/data/hw4/mat_3x3.txt", "/data/hw4/mat_7x7.txt"};
  const int ks[2] = {3, 7};
  const float eps = 1e-8f;

  size_t img_elems = (size_t)N * C * H * W;
  float *h_img = (float *)malloc(img_elems * sizeof(float));
  if (!load_binary(img_file, h_img, img_elems))
    return 1;
  float *d_img;
  CUDA_CHECK(cudaMalloc(&d_img, img_elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_img, h_img, img_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  for (int t = 0; t < 2; ++t) {
    printf("===================================\n");
    int kH = ks[t], kW = ks[t];
    size_t filt_elems = (size_t)K * C * kH * kW;
    float *h_filt = (float *)malloc(filt_elems * sizeof(float));
    if (!load_binary(filt_files[t], h_filt, filt_elems))
      return 1;
    float *d_filt;
    CUDA_CHECK(cudaMalloc(&d_filt, filt_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filt, h_filt, filt_elems * sizeof(float),
                          cudaMemcpyHostToDevice));

    int outH = (H - kH) / stride + 1, outW = (W - kW) / stride + 1;
    size_t out_elems = (size_t)N * K * outH * outW;
    size_t col_rows = C * kH * kW, col_cols = outH * outW;

    // allocate
    float *d_col, *d_conv;
    CUDA_CHECK(cudaMalloc(&d_col, col_rows * col_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv, out_elems * sizeof(float)));

    //////////////////////////////////////////////////////////////////////////////////////

    // measure direct conv2d time
    cudaEvent_t conv_start, conv_end;
    CUDA_CHECK(cudaEventCreate(&conv_start));
    CUDA_CHECK(cudaEventCreate(&conv_end));
    CUDA_CHECK(cudaEventRecord(conv_start));
    // run direct conv2d
    launch_conv2d_direct(d_img, d_filt, d_conv, N, C, H, W, K, kH, kW, pad, pad,
                         stride, stride, dilation, dilation);
    CUDA_CHECK(cudaEventRecord(conv_end));
    CUDA_CHECK(cudaEventSynchronize(conv_end));
    float time_conv;
    CUDA_CHECK(cudaEventElapsedTime(&time_conv, conv_start, conv_end));
    // printf("Filter %dx%d direct conv time: %.3f ms\n", kH, kW, time_conv);
    CUDA_CHECK(cudaEventDestroy(conv_start));
    CUDA_CHECK(cudaEventDestroy(conv_end));
    float *h_conv = (float *)malloc(out_elems * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_conv, d_conv, out_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // compare conv
    std::string conv_file = "/data/hw4/ans_conv_" + std::to_string(kH) + ".bin";
    float *h_conv_exp = (float *)malloc(out_elems * sizeof(float));
    if (!load_binary(conv_file, h_conv_exp, out_elems))
      return 1;
    bool ok_conv = true;
    for (size_t i = 0; i < out_elems; ++i) {
      if (fabs(h_conv[i] - h_conv_exp[i]) > eps) {
        ok_conv = false;
        break;
      }
    }
    printf("Filter %dx%d conv2d check: %s\n", kH, kW,
           ok_conv ? "PASS" : "FAIL");

    //////////////////////////////////////////////////////////////////////////////////////

    // measure im2col time
    cudaEvent_t im2col_start, im2col_end;
    CUDA_CHECK(cudaEventCreate(&im2col_start));
    CUDA_CHECK(cudaEventCreate(&im2col_end));
    CUDA_CHECK(cudaEventRecord(im2col_start));
    // run im2col
    launch_im2col(d_img, N, C, H, W, kH, kW, pad, pad, stride, stride, dilation,
                  dilation, d_col);
    CUDA_CHECK(cudaEventRecord(im2col_end));
    CUDA_CHECK(cudaEventSynchronize(im2col_end));
    float time_im2col;
    CUDA_CHECK(cudaEventElapsedTime(&time_im2col, im2col_start, im2col_end));
    // printf("Filter %dx%d im2col time: %.3f ms\n", kH, kW, time_im2col);
    CUDA_CHECK(cudaEventDestroy(im2col_start));
    CUDA_CHECK(cudaEventDestroy(im2col_end));
    // copy to host
    float *h_col = (float *)malloc(col_rows * col_cols * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_col, d_col, col_rows * col_cols * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // compare with expected
    std::string col_file = "/data/hw4/ans_im2col_" + std::to_string(kH) + ".bin";
    float *h_col_exp = (float *)malloc(col_rows * col_cols * sizeof(float));
    if (!load_binary(col_file, h_col_exp, col_rows * col_cols))
      return 1;
    bool ok_col = true;
    for (size_t i = 0; i < col_rows * col_cols; ++i) {
      if (fabs(h_col[i] - h_col_exp[i]) > eps) {
        ok_col = false;
        break;
      }
    }
    printf("Filter %dx%d im2col check: %s\n", kH, kW, ok_col ? "PASS" : "FAIL");

    //////////////////////////////////////////////////////////////////////////////////////

    // =============== im2col + GEMM conv check ===============
    // allocate GEMM output and filter-col buffer
    float *d_gemm;
    CUDA_CHECK(cudaMalloc(&d_gemm, out_elems * sizeof(float)));
    float *d_filt_col;
    CUDA_CHECK(cudaMalloc(&d_filt_col, filt_elems * sizeof(float)));
    // copy filter to GEMM weight buffer
    CUDA_CHECK(cudaMemcpy(d_filt_col, d_filt, filt_elems * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    // measure GEMM time
    cudaEvent_t gemm_start, gemm_end;
    CUDA_CHECK(cudaEventCreate(&gemm_start));
    CUDA_CHECK(cudaEventCreate(&gemm_end));
    CUDA_CHECK(cudaEventRecord(gemm_start));
    // run GEMM: W_col (K x rows) * im2col matrix (rows x cols) = (K x cols)
    launch_matmul(d_filt_col, d_col, d_gemm, K, col_cols, col_rows);
    CUDA_CHECK(cudaEventRecord(gemm_end));
    CUDA_CHECK(cudaEventSynchronize(gemm_end));
    float time_gemm;
    CUDA_CHECK(cudaEventElapsedTime(&time_gemm, gemm_start, gemm_end));
    // printf("Filter %dx%d GEMM time: %.3f ms\n", kH, kW, time_gemm);
    CUDA_CHECK(cudaEventDestroy(gemm_start));
    CUDA_CHECK(cudaEventDestroy(gemm_end));
    // copy GEMM result to host
    float *h_gemm = (float *)malloc(out_elems * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_gemm, d_gemm, out_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // compare GEMM result to expected conv output
    bool ok_gemm = true;
    for (size_t i = 0; i < out_elems; ++i) {
      if (fabs(h_gemm[i] - h_conv_exp[i]) > eps) {
        ok_gemm = false;
        break;
      }
    }
    printf("Filter %dx%d GEMM conv check: %s\n", kH, kW,
           ok_gemm ? "PASS" : "FAIL");
    // print timing results
    printf("Filter %dx%d direct conv time: %.3f ms\n", kH, kW, time_conv);
    printf("Filter %dx%d im2col time: %.3f ms\n", kH, kW, time_im2col);
    printf("Filter %dx%d GEMM time: %.3f ms\n", kH, kW, time_gemm);
    // cleanup GEMM buffers
    free(h_gemm);
    CUDA_CHECK(cudaFree(d_gemm));
    CUDA_CHECK(cudaFree(d_filt_col));

    // cleanup per filter
    free(h_col);
    free(h_col_exp);
    free(h_conv);
    free(h_conv_exp);
    free(h_filt);
    CUDA_CHECK(cudaFree(d_filt));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_conv));
  }

  // final cleanup
  CUDA_CHECK(cudaFree(d_img));
  free(h_img);
  return 0;
  return 0;
}
