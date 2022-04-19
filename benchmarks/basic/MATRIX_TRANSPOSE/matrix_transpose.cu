// Copyright 2022, ACALab of SJTU

#include <stdio.h>
#include <stdlib.h>

#define BLOCK 16

__global__ void parallelTransposeMemCoalescing(int* A, int* B, int m, int n) {
  __shared__ int block[BLOCK][BLOCK];

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    block[threadIdx.y][threadIdx.x] = A[i * n + j];
    __syncthreads();
    B[j * m + i] = block[threadIdx.y][threadIdx.x];
  }
}

int main(int argc, char* argv[]) {
  int m = 1024;
  int n = 2048;

  int* A = (int*)malloc(m * n * sizeof(int));
  int* B = (int*)malloc(m * n * sizeof(int));

  int i;
  for (i = 0; i < m * n; ++i) A[i] = rand() % 100;

  int *d_A, *d_B;
  cudaMalloc(&d_A, n * m * sizeof(int));
  cudaMalloc(&d_B, n * m * sizeof(int));

  // dimensions
  dim3 threadblock(BLOCK, BLOCK);
  dim3 grid(1 + n / threadblock.x, 1 + m / threadblock.y);

  // copying A to the GPU
  cudaMemcpy(d_A, A, n * m * sizeof(int), cudaMemcpyHostToDevice);

  // calling function
  parallelTransposeMemCoalescing<<<grid, threadblock>>>(d_A, d_B, m, n);

  // once the function has been called I copy the result in matrix
  cudaMemcpy(B, d_B, n * m * sizeof(int), cudaMemcpyDeviceToHost);

  /////////////////////////// CHECKING RESULTS ///////////////////////////
  int error = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      if (B[j * m + i] != A[i * n + j]) {
        error = 1;
        break;
      }
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);

  free(A);
  free(B);

  if (error) {
    printf("Calculation Error!\n");
    return 1;
  }

  printf("Calculation right!\n");
  return 0;
}