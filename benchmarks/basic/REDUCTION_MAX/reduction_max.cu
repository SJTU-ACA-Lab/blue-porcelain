// Copyright 2022, ACALab of SJTU

#include <iostream>
#include <random>

using namespace std;

const int threadsPerBlock = 512;
const int N = (1 << 11) - 3;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

__global__ void sum(int* arr, int* out, int N) {
  __shared__ int s_data[threadsPerBlock];
  unsigned int tid = threadIdx.x;
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    s_data[tid] = arr[i];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && i + s < N) {
      if (s_data[tid] < s_data[tid + s]) s_data[tid] = s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = s_data[0];
  }
}

int varifyOutput(int* predict, int* arr, int N) {
  int pred = -2000, result = -2000;
  for (int i = 0; i < blocksPerGrid; i++) {
    if (predict[i] > pred) pred = predict[i];
  }
  for (int i = 0; i < N; i++) {
    if (arr[i] > result) result = arr[i];
  }
  return pred != result;
}

int main() {
  int *a_host, *r_host;
  int *a_device, *r_device;

  cudaMallocHost(&a_host, N * sizeof(int));
  cudaMallocHost(&r_host, blocksPerGrid * sizeof(int));

  cudaMalloc(&a_device, N * sizeof(int));
  cudaMalloc(&r_device, blocksPerGrid * sizeof(int));

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_int_distribution<int> distr(-1000, 1000);

  for (int i = 0; i < N; i++) {
    a_host[i] = distr(eng);
  }
  for (int i = 0; i < blocksPerGrid; i++) {
    r_host[i] = 0.0;
  }

  cudaMemcpyAsync(a_device, a_host, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(r_device, r_host, blocksPerGrid * sizeof(int),
                  cudaMemcpyHostToDevice);

  sum<<<blocksPerGrid, threadsPerBlock, 0>>>(a_device, r_device, N);

  cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(int),
             cudaMemcpyDeviceToHost);

  int error = varifyOutput(r_host, a_host, N);

  cudaFree(r_device);
  cudaFree(a_device);
  cudaFreeHost(r_host);
  cudaFreeHost(a_host);

  if (error) {
    cout << "Calculation Error!\n";
    return 1;
  }
  cout << "Calculation right!\n";
  return 0;
}