#include "cuda_runtime.h"

/*
  You can change the kernel name and the function signature.
*/
// __global__ void lora_kernel(const float *x, const float *W, const float *A,
//                             const float *B, float *y, int BATCH, int IN_DIM,
//                             int OUT_DIM, int RANK, float scale) {}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y, int B,
          int in_dim, int out_dim, int r, float scale) {
  /*
   Call the kernel here.
  */
}
