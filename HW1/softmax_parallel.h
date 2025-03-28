#include <algorithm>
#include <cmath>
#include <thread>
// Do NOT add any other headers.

using namespace std; // You can remove this line if you want.

/**
 * @brief Computes the softmax function in parallel.
 *
 * This function takes an input array of floats and computes the softmax
 * function in parallel using the specified number of threads. The result
 * is stored in the output array.
 *
 * @param in Pointer to the input array of floats.
 * @param out Pointer to the output array where the softmax results will be
 * stored.
 * @param elems The number of elements in the input array.
 */
inline void softmax_parallel(float *in, float *out, int elems) {
    const int NTHREADS = std::thread::hardware_concurrency(); 
    int chunk_size = (elems + NTHREADS - 1) / NTHREADS;  // ceil(elems / NTHREADS)

    // Step 1: Find global max in parallel
    alignas(64) float local_max[NTHREADS];  
    std::thread threads[NTHREADS];

    for (int t = 0; t < NTHREADS; ++t) {
        threads[t] = std::thread([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, elems);
            local_max[t] = -std::numeric_limits<float>::infinity();
            for (int i = start; i < end; ++i) {
                if (in[i] > local_max[t]) local_max[t] = in[i];
            }
        });
    }
    for (int t = 0; t < NTHREADS; ++t) threads[t].join();

    float global_max = local_max[0];
    for (int t = 1; t < NTHREADS; ++t) {
        if (local_max[t] > global_max) global_max = local_max[t];
    }

    // Step 2: Compute exponentials and sum in parallel
    alignas(64) float local_sum[NTHREADS]{0.0f};

    for (int t = 0; t < NTHREADS; ++t) {
        threads[t] = std::thread([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, elems);
            float sum = 0.0f;
            for (int i = start; i < end; ++i) {
                out[i] = std::exp(in[i] - global_max);
                sum += out[i];
            }
            local_sum[t] = sum;
        });
    }
    for (int t = 0; t < NTHREADS; ++t) threads[t].join();

    float global_sum = 0.0f;
    for (int t = 0; t < NTHREADS; ++t) {
        global_sum += local_sum[t];
    }

    // Step 3: Normalize in parallel
    for (int t = 0; t < NTHREADS; ++t) {
        threads[t] = std::thread([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, elems);
            for (int i = start; i < end; ++i) {
                out[i] /= global_sum;
            }
        });
    }
    for (int t = 0; t < NTHREADS; ++t) threads[t].join();
}