// Copyright 2022, ACALab of SJTU

#include <stdio.h>
#include <iostream>

#define imin(a, b) (a < b ? a : b)

const int N = 16 * 1024;

const int threadsPerBlock = 256;

const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {
  __shared__ float cache[threadsPerBlock];  // 线程块共享
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  // 设置cache中相应位置上的值
  cache[cacheIndex] = temp;

  // 对线程块内线程进行同步
  __syncthreads();

  // 规约运算，求和
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }
  if (cacheIndex == 0) c[blockIdx.x] = cache[0];
}

int main() {
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  // 在CPU上分配内存
  a = (float *)malloc(N * sizeof(float));
  b = (float *)malloc(N * sizeof(float));
  partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

  // 在GPU上分配内存
  (cudaMalloc((void **)&dev_a, N * sizeof(float)));
  (cudaMalloc((void **)&dev_b, N * sizeof(float)));
  (cudaMalloc((void **)&dev_partial_c, blocksPerGrid * sizeof(float)));

  // 填充主机内存
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // 将数组a和b复制到GPU
  (cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
  (cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

  dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

  // 将数组c从GPU复制到CPU
  (cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float),
              cudaMemcpyDeviceToHost));

  // 在CPU上完成最终的求和运算
  c = 0;
  for (int i = 0; i < blocksPerGrid; i++) {
    c += partial_c[i];
  }

#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
  float cpu_value = 2 * sum_squares((float)(N - 1));
  printf("Does GPU value %.6g = %.6g?\n", c, cpu_value);

  // 释放GPU上的内存
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);

  // 释放CPU上的内存
  free(a);
  free(b);
  free(partial_c);

  if ((c - cpu_value) / cpu_value < 0.00001 &&
      (c - cpu_value) / cpu_value > -0.00001) {
    std::cout << "Calculation Right!!\n";
    return 0;
  } else {
    std::cout << "Calculation Error!\n";
  }

  return 1;
}