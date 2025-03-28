#include "softmax_parallel.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
	srand(time(NULL));
	int N = 1 << 28;

	float *array_in = new float[N];
	float *array_out_serial = new float[N];
	float *array_out_parallel = new float[N];

	{
		// 0. Initialize
		std::chrono::duration<float> diff;
		auto start = std::chrono::steady_clock::now();
		for (int i = 0; i < N; i++) {
			array_in[i] = ((rand() % 200'000) - 100'000) / 1'000.0;
			array_out_serial[i] = -1000000.0;
			array_out_parallel[i] = -1000000.0;
			// printf("%f\n", array_in[i]);
		}
		auto end = std::chrono::steady_clock::now();
		diff = end - start;
		std::cout << "init took " << diff.count() << " sec" << std::endl;
	}

	{
		// 1. Serial
		std::chrono::duration<float> diff;
		auto start = std::chrono::steady_clock::now();
		{
			float max = *std::max_element(array_in, array_in + N);
			for (int i = 0; i < N; i++) {
				array_out_serial[i] = array_in[i] - max;
			}
			std::for_each(array_out_serial, array_out_serial + N,
						  [&](float &x) { x = std::exp(x); });
			float sum =
				std::accumulate(array_out_serial, array_out_serial + N, 0.0f);
			std::for_each(array_out_serial, array_out_serial + N,
						  [&](float &x) { x = x / sum; });
		}
		auto end = std::chrono::steady_clock::now();
		diff = end - start;
		std::cout << "serial softmax took " << diff.count() << " sec"
				  << std::endl;
	}

	{
		// 2. parallel softmax
		std::chrono::duration<float> diff;
		auto start = std::chrono::steady_clock::now();
		softmax_parallel(array_in, array_out_parallel, N);
		auto end = std::chrono::steady_clock::now();
		diff = end - start;
		std::cout << "parallel softmax took " << diff.count() << " sec"
				  << std::endl;
	}

	{
		int error_counts = 0;
		const float epsilon = 0.01;
		for (int i = 0; i < N; i++) {
			float err = std::abs(array_out_serial[i] - array_out_parallel[i]);
			if (err > epsilon) {
				error_counts++;
				if (error_counts < 5) {
					std::cout << "ERROR at " << i << ": Serial[" << i
							  << "] = " << array_out_serial[i] << " Parallel["
							  << i << "] = " << array_out_parallel[i]
							  << std::endl;
					std::cout << "err: " << err << std::endl;
				}
			}
		}

		if (error_counts == 0) {
			std::cout << "PASS" << std::endl;
		} else {
			std::cout << "There are " << error_counts << " errors" << std::endl;
		}
	}

	return 0;
}
