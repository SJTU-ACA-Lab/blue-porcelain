// Copyright 2022, ACALab of SJTU

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

__global__ void vecAddF(float *a, float *b, float *c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) c[index] = a[index] + b[index];
}

__global__ void vecAddI32(int *a, int *b, int *c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) c[index] = a[index] + b[index];
}

__global__ void vecAddI64(long long *a, long long *b, long long *c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

// basically just fills the array with index.
float getRandData(int min, int max) {
  float m1 = (double)(rand() % 101) / 101;
  min++;
  float m2 = (double)((rand() % (max - min + 1)) + min);
  m2 = m2 - 1;
  return m1 + m2;
}

void fill_array(float *data) {
  for (int i = 0; i < N; ++i) {
    data[i] = getRandData(0, 1000);
  }
}

void fill_array(int *data) {
  for (int idx = 0; idx < N; idx++) data[idx] = rand() % 200000;
}

void fill_array(long long *data) {
  for (int idx = 0; idx < N; idx++) data[idx] = INT_MAX + idx;
}

int main() {
  srand((unsigned)time(NULL));

  float *fa, *fb, *fc;
  float *d_fa, *d_fb, *d_fc;

  int *a, *b, *c;
  int *d_a, *d_b, *d_c;

  long long *la, *lb, *lc;
  long long *d_la, *d_lb, *d_lc;

  int threads_per_block = 0, no_of_blocks = 0;

  int fsize = N * sizeof(float);
  int size = N * sizeof(int);
  int lsize = N * sizeof(long long);

  // Alloc space for host copies of a, b, c and setup input values
  fa = (float *)malloc(fsize);
  fill_array(fa);
  fb = (float *)malloc(fsize);
  fill_array(fb);
  fc = (float *)malloc(fsize);

  a = (int *)malloc(size);
  fill_array(a);
  b = (int *)malloc(size);
  fill_array(b);
  c = (int *)malloc(size);

  la = (long long *)malloc(lsize);
  fill_array(la);
  lb = (long long *)malloc(lsize);
  fill_array(lb);
  lc = (long long *)malloc(lsize);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_fa, fsize);
  cudaMalloc((void **)&d_fb, fsize);
  cudaMalloc((void **)&d_fc, fsize);

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  cudaMalloc((void **)&d_la, lsize);
  cudaMalloc((void **)&d_lb, lsize);
  cudaMalloc((void **)&d_lc, lsize);

  // Copy inputs to device
  cudaMemcpy(d_fa, fa, fsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_fb, fb, fsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_la, la, lsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lb, lb, lsize, cudaMemcpyHostToDevice);

  threads_per_block = 256;
  no_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  vecAddF<<<no_of_blocks, threads_per_block>>>(d_fa, d_fb, d_fc);
  vecAddI32<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c);
  vecAddI64<<<no_of_blocks, threads_per_block>>>(d_la, d_lb, d_lc);

  // Copy result back to host
  cudaMemcpy(fc, d_fc, fsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(lc, d_lc, lsize, cudaMemcpyDeviceToHost);

  int flag = 0;
  for (int i = 0; i < N; i++) {
    if (fc[i] != fa[i] + fb[i]) {
      printf("FC[%d]:%f != %f + %f\n", i, fc[i], fa[i], fb[i]);
      flag = 1;
    }
    if (c[i] != a[i] + b[i]) {
      printf("C[%d]:%d != %d + %d\n", i, c[i], a[i], b[i]);
      flag = 1;
    }
    if (lc[i] != la[i] + lb[i]) {
      printf("LC[%d]:%lld != %lld + %lld\n", i, lc[i], la[i], lb[i]);
      flag = 1;
    }
  }

  free(fa);
  free(fb);
  free(fc);
  free(a);
  free(b);
  free(c);
  free(la);
  free(lb);
  free(lc);
  cudaFree(d_fa);
  cudaFree(d_fb);
  cudaFree(d_fc);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_la);
  cudaFree(d_lb);
  cudaFree(d_lc);

  if (flag == 0) {
    printf("Calculation right!\n");
    return 0;
  }

  return 1;
}
