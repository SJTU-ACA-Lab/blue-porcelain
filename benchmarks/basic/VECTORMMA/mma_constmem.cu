// Copyright 2022, ACALab of SJTU

#include <stdio.h>
#include <stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c) {
  for (int idx = 0; idx < N; idx++) c[idx] = a[idx] + b[idx];
}
__constant__ int d[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
__global__ void device_add(int *a, int *b, int *c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    c[index] = a[index] * b[index] + d[index];
  }
}

// basically just fills the array with index.
void fill_array(int *data) {
  for (int i = 0; i < N; ++i) {
    data[i] = rand() % (int)1000;
  }
}

void print_output(int *a, int *b, int *c) {
  for (int idx = 0; idx < N; idx++)
    printf("\n %d + %d  = %d", a[idx], b[idx], c[idx]);
}
int main(void) {
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;  // device copies of a, b, c
  int threads_per_block = 0, no_of_blocks = 0;

  int size = N * sizeof(int);

  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(size);
  fill_array(a);
  b = (int *)malloc(size);
  fill_array(b);
  c = (int *)malloc(size);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  threads_per_block = 256;
  no_of_blocks = N / threads_per_block;
  device_add<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  int flag = 0;
  for (int i = 0; i < N; i++) {
    if (i < 10) {
      if (c[i] != a[i] * b[i] + i + 1) {
        printf("\nCalculation Error!\n");
        flag = 1;
        break;
      }
    } else {
      if (c[i] != a[i] * b[i]) {
        printf("\nCalculation Error!\n");
        flag = 1;
        break;
      }
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
