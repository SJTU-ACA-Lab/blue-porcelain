/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../../common/polybenchUtilFuncts.h"

// Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#define NX 4096
#define NY 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* p, DATA_TYPE* r) {
  int i, j;

  for (i = 0; i < NX; i++) {
    r[i] = i * M_PI;

    for (j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
    }
  }

  for (i = 0; i < NY; i++) {
    p[i] = i * M_PI;
  }
}

void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q,
                    DATA_TYPE* q_outputFromGpu, int& res) {
  int i, fail;
  fail = 0;

  // Compare s with s_cuda
  for (i = 0; i < NX; i++) {
    if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  for (i = 0; i < NY; i++) {
    if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  // print results
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

// Distributed (split) from initial loop and permuted into reverse order to
// allow parallelism...
__global__ void bicg_kernel1(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < NY) {
    s[j] = 0.0f;

    int i;
    for (i = 0; i < NX; i++) {
      s[j] += A[i * NY + j] * r[i];
    }
  }
}

// Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(DATA_TYPE* A, DATA_TYPE* p, DATA_TYPE* q) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < NX) {
    q[i] = 0.0f;

    int j;
    for (j = 0; j < NY; j++) {
      q[i] += A[i * NY + j] * p[j];
    }
  }
}

void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p,
              DATA_TYPE* q) {
  int i, j;

  for (i = 0; i < NY; i++) {
    s[i] = 0.0;
  }

  for (i = 0; i < NX; i++) {
    q[i] = 0.0;
    for (j = 0; j < NY; j++) {
      s[j] = s[j] + r[i] * A[i * NY + j];
      q[i] = q[i] + A[i * NY + j] * p[j];
    }
  }
}

void bicgCuda(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p,
              DATA_TYPE* q, DATA_TYPE* s_outputFromGpu,
              DATA_TYPE* q_outputFromGpu) {
  double t_start, t_end;

  DATA_TYPE* A_gpu;
  DATA_TYPE* q_gpu;
  DATA_TYPE* p_gpu;
  DATA_TYPE* r_gpu;
  DATA_TYPE* s_gpu;

  cudaMalloc((void**)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
  cudaMalloc((void**)&r_gpu, sizeof(DATA_TYPE) * NX);
  cudaMalloc((void**)&s_gpu, sizeof(DATA_TYPE) * NY);
  cudaMalloc((void**)&p_gpu, sizeof(DATA_TYPE) * NY);
  cudaMalloc((void**)&q_gpu, sizeof(DATA_TYPE) * NX);
  cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
  cudaMemcpy(r_gpu, r, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);
  cudaMemcpy(s_gpu, s, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
  cudaMemcpy(p_gpu, p, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
  cudaMemcpy(q_gpu, q, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)(ceil(((float)NY) / ((float)block.x))), 1);
  dim3 grid2((size_t)(ceil(((float)NX) / ((float)block.x))), 1);

  t_start = rtclock();
  bicg_kernel1<<<grid1, block>>>(A_gpu, r_gpu, s_gpu);
  cudaThreadSynchronize();
  bicg_kernel2<<<grid2, block>>>(A_gpu, p_gpu, q_gpu);
  cudaThreadSynchronize();
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  cudaMemcpy(s_outputFromGpu, s_gpu, sizeof(DATA_TYPE) * NY,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(q_outputFromGpu, q_gpu, sizeof(DATA_TYPE) * NX,
             cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(r_gpu);
  cudaFree(s_gpu);
  cudaFree(p_gpu);
  cudaFree(q_gpu);
}

int main(int argc, char** argv) {
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* r;
  DATA_TYPE* s;
  DATA_TYPE* p;
  DATA_TYPE* q;
  DATA_TYPE* s_outputFromGpu;
  DATA_TYPE* q_outputFromGpu;

  A = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));
  r = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
  s = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  p = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  q = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
  s_outputFromGpu = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  q_outputFromGpu = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));

  init_array(A, p, r);

  GPU_argv_init();

  bicgCuda(A, r, s, p, q, s_outputFromGpu, q_outputFromGpu);

  t_start = rtclock();
  bicg_cpu(A, r, s, p, q);
  t_end = rtclock();

  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
  int res = 0;
  compareResults(s, s_outputFromGpu, q, q_outputFromGpu, res);

  free(A);
  free(r);
  free(s);
  free(p);
  free(q);
  free(s_outputFromGpu);
  free(q_outputFromGpu);

  return res == 0 ? 0 : 1;
}
