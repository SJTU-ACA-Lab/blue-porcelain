// Copyright 2022, ACALab of SJTU

#include <iostream>
#include <random>

using namespace std;

const int threadsPerBlock = 512;
const int N = (1 << 11) - 3;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

__global__ void sum(float* arr, float* out, int N) {
  __shared__ float s_data[threadsPerBlock];
  unsigned int tid = threadIdx.x;
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    s_data[tid] = arr[i];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && i + s < N) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = s_data[0];
  }
}

int varifyOutput(float* predict, float* arr, int N) {
  float pred = 0.0;
  for (int i = 0; i < blocksPerGrid; i++) {
    pred += predict[i];
  }

  float result = 0.0;
  for (int i = 0; i < N; i++) {
    result += arr[i];
  }

  if (abs((pred - result) / result) > 1e-5) return 1;
  return 0;
}

int main() {
  float *a_host, *r_host;
  float *a_device, *r_device;

  cudaMallocHost(&a_host, N * sizeof(float));
  cudaMallocHost(&r_host, blocksPerGrid * sizeof(float));

  cudaMalloc(&a_device, N * sizeof(float));
  cudaMalloc(&r_device, blocksPerGrid * sizeof(float));

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::normal_distribution<float> distr(-8.8, 8.8);

  for (int i = 0; i < N; i++) {
    a_host[i] = distr(eng);
  }
  for (int i = 0; i < blocksPerGrid; i++) {
    r_host[i] = 0.0;
  }

  cudaMemcpyAsync(a_device, a_host, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(r_device, r_host, blocksPerGrid * sizeof(float),
                  cudaMemcpyHostToDevice);

  sum<<<blocksPerGrid, threadsPerBlock, 0>>>(a_device, r_device, N);

  cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(float),
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