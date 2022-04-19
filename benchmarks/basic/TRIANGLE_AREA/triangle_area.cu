// Copyright 2022, ACALab of SJTU

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#define N 2048

void host_triangle_area(float *arr1, float *arr2, float *arr3, float *res) {
  float a, b, c;
  for (int idx = 0; idx < N; idx++) {
    a = arr1[idx];
    b = arr2[idx];
    c = arr2[idx];
    if (a + b > c && a + c > b && b + c > a) {
      float p = (a + b + c) * 0.5;  //计算半周长
      float area = sqrt(p * (p - a) * (p - b) * (p - c));
      res[idx] = area;

    } else {
      res[idx] = 0;
    }
  }
}

__global__ void gpu_triangle_area(float *arr1, float *arr2, float *arr3,
                                  float *res) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float a, b, c;
  a = arr1[idx];
  b = arr2[idx];
  c = arr2[idx];
  if (a + b > c && a + c > b && b + c > a) {
    float p = (a + b + c) * 0.5;  //计算半周长
    float area = sqrt(p * (p - a) * (p - b) * (p - c));
    res[idx] = area;

  } else {
    res[idx] = 0;
  }
}

int main(void) {
  float *a, *b, *c, *res, *cpu_res;
  float *d_a, *d_b, *d_c, *d_res;  // device copies of a, b, c
  int threads_per_block = 0, no_of_blocks = 0;

  int size = N * sizeof(float);

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<float> distr;

  // Alloc space for host copies of a, b, c and setup input values
  a = (float *)malloc(size);
  for (int i = 0; i < N; i++) {
    a[i] = distr(eng);
  }
  b = (float *)malloc(size);
  for (int i = 0; i < N; i++) {
    b[i] = distr(eng);
  }
  c = (float *)malloc(size);
  for (int i = 0; i < N; i++) {
    c[i] = distr(eng);
  }
  res = (float *)malloc(size);
  cpu_res = (float *)malloc(size);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);
  cudaMalloc((void **)&d_res, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

  threads_per_block = 256;
  no_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  gpu_triangle_area<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c, d_res);

  // Copy result back to host
  cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);

  host_triangle_area(a, b, c, cpu_res);

  int flag = 0;
  for (int i = 0; i < N; i++) {
    if (abs((cpu_res[i] - res[i]) / cpu_res[i]) > 1e-3) {
      printf("C[%d]:%f != %f\n", i, cpu_res[i], res[i]);
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