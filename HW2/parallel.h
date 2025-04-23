#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>
#include <random>
// You cannot use OpenMP <omp.h>
// Include header files if you need,
// but it must work without modifying the Makefile

/**
 * @brief Initializes a vector with random double values.
 *
 * This function fills the given array
 *
 * @param a Pointer to the array to be initialized.
 * @param N The number of elements in the array.
 */
 
const int MAX_THREADS = std::thread::hardware_concurrency();

inline void init_vec(double *a, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */
  int num_threads = MAX_THREADS;
  int chunk = N / num_threads;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([=]() {
          int start = t * chunk;
          int end = (t == num_threads - 1) ? N : start + chunk;
          for (int i = start; i < end; ++i) {
              a[i] = static_cast<double>((i % 13) + 1); // 단순한 값
          }
      });
  }
  for (auto &th : threads) th.join();

	/****************/
}

/**
 * @brief Performs a matrix-vector multiplication.
 *
 * This function computes the product of a matrix 'a' and a vector 'b', storing
 * the result in vector 'c'.
 *
 * @param a Pointer to the first element of the matrix 'a' (assumed to be in
 * row-major order).
 * @param b Pointer to the first element of the vector 'b'.
 * @param c Pointer to the first element of the result vector 'c'.
 * @param N The dimension of the matrix and vectors (assuming a square matrix
 * and compatible vector sizes).
 */
inline void gemv(double *a, double *b, double *c, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */
  int num_threads = MAX_THREADS;
  int chunk = N / num_threads;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([=]() {
          int start = t * chunk;
          int end = (t == num_threads - 1) ? N : start + chunk;

          for (int i = start; i < end; ++i) {
              double sum = 0;
              int j = 0;
              for (; j + 4 <= N; j += 4) {
                  sum += a[i * N + j] * b[j];
                  sum += a[i * N + j + 1] * b[j + 1];
                  sum += a[i * N + j + 2] * b[j + 2];
                  sum += a[i * N + j + 3] * b[j + 3];
              }
              for (; j < N; ++j) {
                  sum += a[i * N + j] * b[j];
              }
              c[i] = sum;
          }
      });
  }
  for (auto &th : threads) th.join();

	/****************/
}

/**
 * @brief Performs matrix multiplication of two NxN matrices.
 *
 * This function computes the product of two square matrices a and b,
 * and stores the result in matrix c. All matrices are represented as
 * 1-dimensional arrays in row-major order.
 *
 * @param a Pointer to the first input matrix (NxN).
 * @param b Pointer to the second input matrix (NxN).
 * @param c Pointer to the output matrix (NxN) where the result will be stored.
 * @param N The dimension of the matrices (number of rows and columns).
 */
inline void gemm(double *a, double *b, double *c, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */
  int num_threads = MAX_THREADS;
    int chunk = N / num_threads;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([=]() {
            int row_start = t * chunk;
            int row_end = (t == num_threads - 1) ? N : row_start + chunk;

            for (int i = row_start; i < row_end; ++i) {
                for (int j = 0; j < N; ++j) {
                    c[i * N + j] = 0;
                }
                for (int k = 0; k < N; ++k) {
                    double aik = a[i * N + k];
                    for (int j = 0; j < N; ++j) {
                        c[i * N + j] += aik * b[k * N + j];
                    }
                }
            }
        });
    }
    for (auto &th : threads) th.join();
	/****************/
}
