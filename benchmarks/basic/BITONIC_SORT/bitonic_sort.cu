// Copyright 2022, ACALab of SJTU

#include <stdio.h>
#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (1 << 6)
#endif

#define GRID_SIZE ((1 << 10) / BLOCK_SIZE)

__device__ inline void swap(int &a, int &b) {
  unsigned int tmp = a;
  a = b;
  b = tmp;
}

__global__ static void bitonicSort(int *values) {
  extern __shared__ int shared[];
  const unsigned int tid = threadIdx.x;
  int offset = blockIdx.x * BLOCK_SIZE;
  shared[tid] = values[offset + tid];

  __syncthreads();

  for (unsigned int k = 2; k <= BLOCK_SIZE; k *= 2) {
    for (unsigned int j = k / 2; j > 0; j /= 2) {
      unsigned int ixj = tid ^ j;
      if (ixj > tid) {
        if ((tid & k) == 0) {
          if (shared[ixj] < shared[tid]) swap(shared[tid], shared[ixj]);
        }

        else {
          if (shared[ixj] > shared[tid]) swap(shared[tid], shared[ixj]);
        }
      }
      __syncthreads();
    }
  }
  values[offset + tid] = shared[tid];
}

void initialize_data(int *data, int size) {
  for (int i = 0; i < size; i++) data[i] = BLOCK_SIZE - i;
}

int main(int argc, char **argv) {
  srand(0);
  int *values_h, *values_d;
  int size = BLOCK_SIZE * GRID_SIZE;

  values_h = (int *)malloc(sizeof(int) * size);
  cudaMalloc((void **)&values_d, sizeof(int) * size);

  initialize_data(values_h, size);

  cudaMemcpy(values_d, values_h, sizeof(int) * size, cudaMemcpyHostToDevice);
  bitonicSort<<<GRID_SIZE, BLOCK_SIZE, sizeof(int) * BLOCK_SIZE>>>(values_d);
  cudaMemcpy(values_h, values_d, sizeof(int) * size, cudaMemcpyDeviceToHost);

  bool pass = true;

  for (int blk = 0; blk < GRID_SIZE; ++blk) {
    int offset = blk * BLOCK_SIZE;
    for (int i = offset; i < offset + BLOCK_SIZE - 1; i++) {
      if (values_h[i] > values_h[i + 1]) {
        pass = false;
        break;
      }
    }
    if (!pass) {
      printf("Calculation Error!\n");
      break;
    }
  }

  cudaFree(values_d);
  free(values_h);

  if (pass) {
    printf("Calculation right!\n");
    return 0;
  }

  return 1;
}
