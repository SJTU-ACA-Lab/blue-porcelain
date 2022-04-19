// Copyright 2022, ACALab of SJTU

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 2048

void host_div(float *a, float *b, float *c) {
  for (int idx = 0; idx < N; idx++) c[idx] = a[idx] + b[idx];
}

__global__ void vectorDiv(float *a, float *b, float *c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] / b[index];
}

// basically just fills the array with index.
void fill_array(float *data, float d) {
  for (int idx = 0; idx < N; idx++) data[idx] = (idx + d) / (d + 1);
}

int main(void) {
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;  // device copies of a, b, c
  int threads_per_block = 0, no_of_blocks = 0;

  int size = N * sizeof(float);

  // Alloc space for host copies of a, b, c and setup input values
  a = (float *)malloc(size);
  fill_array(a, 3.0);
  b = (float *)malloc(size);
  fill_array(b, 5.0);
  c = (float *)malloc(size);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  threads_per_block = 256;
  no_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  vectorDiv<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  int flag = 0;
  for (int i = 0; i < N; i++) {
    if (c[i] - (a[i] / b[i]) < -0.000001 || c[i] - (a[i] / b[i]) > 0.000001) {
      printf("C[%d]:%f != %f\n", i, c[i], a[i] / b[i]);
      flag = 1;
    }
  }

  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  if (flag == 0) {
    printf("Calculation right!\n");
    return 0;
  }

  return 1;
}