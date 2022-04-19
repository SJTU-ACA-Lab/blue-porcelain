/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
             DATA_TYPE *tmp) {
  int i, j;

  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}

void init(DATA_TYPE *A, DATA_TYPE *x) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void compareResults(DATA_TYPE *y, DATA_TYPE *y_outputFromGpu, int &res) {
  int i, fail;
  fail = 0;

  for (i = 0; i < (N); i++) {
    if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
      printf("y[%d]=%f\ty_gpu[%d]=%f\n", i, y[i], i, y_outputFromGpu[i]);
    }
  }

  // Print results
  printf(
      "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
      "%d\n",
      PERCENT_DIFF_ERROR_THRESHOLD, fail);
  res = fail;
}

void GPU_argv_init() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  cudaSetDevice(GPU_DEVICE);
}

__global__ void gesummv_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *x,
                               DATA_TYPE *y, DATA_TYPE *tmp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    int j;
    for (j = 0; j < N; j++) {
      tmp[i] += a[i * N + j] * x[j];
      y[i] += b[i * N + j] * x[j];
    }
    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}

void gesummvCuda(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
                 DATA_TYPE *tmp, DATA_TYPE *y_outputFromGpu) {
  double t_start, t_end;

  DATA_TYPE *A_gpu;
  DATA_TYPE *B_gpu;
  DATA_TYPE *x_gpu;
  DATA_TYPE *y_gpu;
  DATA_TYPE *tmp_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
  cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
  cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
  cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
  cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * N);

  cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((unsigned int)ceil(((float)N) / ((float)block.x)), 1);

  t_start = rtclock();
  gesummv_kernel<<<grid, block>>>(A_gpu, B_gpu, x_gpu, y_gpu, tmp_gpu);
  cudaThreadSynchronize();
  t_end = rtclock();
  cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * N,
             cudaMemcpyDeviceToHost);

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}

int main(int argc, char *argv[]) {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  A = (DATA_TYPE *)calloc(N * N, sizeof(DATA_TYPE));
  B = (DATA_TYPE *)calloc(N * N, sizeof(DATA_TYPE));
  x = (DATA_TYPE *)calloc(N, sizeof(DATA_TYPE));
  y = (DATA_TYPE *)calloc(N, sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)calloc(N, sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)calloc(N, sizeof(DATA_TYPE));
  init(A, x);

  GPU_argv_init();
  gesummvCuda(A, B, x, y, tmp, y_outputFromGpu);

  t_start = rtclock();
  gesummv(A, B, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
  int res = 0;
  compareResults(y, y_outputFromGpu, res);

  free(A);
  free(B);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return res == 0 ? 0 : 1;
}
