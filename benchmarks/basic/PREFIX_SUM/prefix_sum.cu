// origin: https://github.com/mark-poscablo/gpu-prefix-sum

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <ctime>

#include "utils.h"
#include "timer.h"
#include "scan.h"

void cpu_sum_scan(unsigned int* const h_out, const unsigned int* const h_in,
                  const size_t numElems) {
  unsigned int run_sum = 0;
  for (int i = 0; i < numElems; ++i) {
    h_out[i] = run_sum;
    run_sum = run_sum + h_in[i];
  }
}

int main() {
  // Set up clock for timing comparisons
  srand(time(NULL));
  std::clock_t start;
  double duration;

  unsigned int h_in_len = 0;
  int k = 10;
  //   for (int k = 1; k < 29; ++k) {
  h_in_len = (1 << k) + 3;
  std::cout << "h_in size: " << h_in_len << std::endl;

  // Generate input
  unsigned int* h_in = new unsigned int[h_in_len];
  for (int i = 0; i < h_in_len; ++i) {
    // h_in[i] = rand() % 2;
    h_in[i] = i;
  }

  // Set up host-side memory for output
  unsigned int* h_out_naive = new unsigned int[h_in_len];
  unsigned int* h_out_blelloch = new unsigned int[h_in_len];

  // Set up device-side memory for input
  unsigned int* d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * h_in_len));
  checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * h_in_len,
                             cudaMemcpyHostToDevice));

  // Set up device-side memory for output
  unsigned int* d_out_blelloch;
  checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(unsigned int) * h_in_len));

  // Do CPU scan for reference
  start = std::clock();
  cpu_sum_scan(h_out_naive, h_in, h_in_len);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  std::cout << "CPU time: " << duration << std::endl;

  // Do GPU scan
  start = std::clock();
  sum_scan_blelloch(d_out_blelloch, d_in, h_in_len);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  std::cout << "GPU time: " << duration << std::endl;

  // Copy device output array to host output array
  // And free device-side memory
  checkCudaErrors(cudaMemcpy(h_out_blelloch, d_out_blelloch,
                             sizeof(unsigned int) * h_in_len,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_out_blelloch));
  checkCudaErrors(cudaFree(d_in));

  // Check for any mismatches between outputs of CPU and GPU
  bool match = true;
  int index_diff = 0;
  for (int i = 0; i < h_in_len; ++i) {
    if (h_out_naive[i] != h_out_blelloch[i]) {
      match = false;
      index_diff = i;
      break;
    }
  }
  std::cout << "Match: " << match << std::endl;

  // Detail the mismatch if any
  if (!match) {
    std::cout << "Difference in index: " << index_diff << std::endl;
    std::cout << "CPU: " << h_out_naive[index_diff] << std::endl;
    std::cout << "Blelloch: " << h_out_blelloch[index_diff] << std::endl;
    int window_sz = 10;

    std::cout << "Contents: " << std::endl;
    std::cout << "CPU: ";
    for (int i = -(window_sz / 2); i < (window_sz / 2); ++i) {
      std::cout << h_out_naive[index_diff + i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Blelloch: ";
    for (int i = -(window_sz / 2); i < (window_sz / 2); ++i) {
      std::cout << h_out_blelloch[index_diff + i] << ", ";
    }
    std::cout << std::endl;
  }

  // Free host-side memory
  delete[] h_in;
  delete[] h_out_naive;
  delete[] h_out_blelloch;

  std::cout << std::endl;

  if (match) {
    return 0;
  } else {
    return 1;
  }
  //   }
}
