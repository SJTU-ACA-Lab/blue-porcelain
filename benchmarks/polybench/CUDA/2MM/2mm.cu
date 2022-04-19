/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <unistd.h>

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size. */
#define NI 160
#define NJ 160
#define NK 160
#define NL 160

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NL; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NL + j] = 0;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

void compareResults(DATA_TYPE* E, DATA_TYPE* E_outputFromGpu, int& res) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NL; i++) {
    for (j = 0; j < NI; j++) {
      if (percentDiff(E[i * NI + j], E_outputFromGpu[i * NI + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // print results
  printf(
      "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
      "%d\n",
      PERCENT_DIFF_ERROR_THRESHOLD, fail);
  res = fail;
}

__global__ void mm2_kernel1(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < NI) && (j < NJ)) {
    int k;
    for (k = 0; k < NK; k++) {
      C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
    }
  }
}

__global__ void mm2_kernel2(DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < NI) && (j < NL)) {
    int k;
    for (k = 0; k < NJ; k++) {
      E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
    }
  }
}

void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D,
             DATA_TYPE* E) {
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] = 0.0;
      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
      }
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      E[i * NL + j] = 0.0;
      for (k = 0; k < NJ; ++k) {
        E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
      }
    }
  }
}

void mm2Cuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D,
             DATA_TYPE* E, DATA_TYPE* E_outputFromGpu) {
  double t_start, t_end;

  DATA_TYPE* A_gpu;
  DATA_TYPE* B_gpu;
  DATA_TYPE* C_gpu;
  DATA_TYPE* D_gpu;
  DATA_TYPE* E_gpu;

  cudaMalloc((void**)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
  cudaMalloc((void**)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
  cudaMalloc((void**)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
  cudaMalloc((void**)&D_gpu, sizeof(DATA_TYPE) * NJ * NL);
  cudaMalloc((void**)&E_gpu, sizeof(DATA_TYPE) * NI * NL);

  cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
  cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
  cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)ceil(((float)NJ) / ((float)block.x)),
             (size_t)ceil(((float)NI) / ((float)block.y)));
  dim3 grid2((size_t)ceil(((float)NL) / ((float)block.x)),
             (size_t)ceil(((float)NI) / ((float)block.y)));
  t_start = rtclock();
  mm2_kernel1<<<grid1, block>>>(A_gpu, B_gpu, C_gpu);
  mm2_kernel2<<<grid2, block>>>(C_gpu, D_gpu, E_gpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  cudaMemcpy(E_outputFromGpu, E_gpu, sizeof(DATA_TYPE) * NI * NL,
             cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  cudaFree(D_gpu);
  cudaFree(E_gpu);
}

int main(int argc, char** argv) {
  double t_start, t_end;

  DATA_TYPE* C;
  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* D;
  DATA_TYPE* E;
  DATA_TYPE* E_outputFromGpu;

  C = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));
  A = (DATA_TYPE*)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NK * NJ * sizeof(DATA_TYPE));
  D = (DATA_TYPE*)malloc(NJ * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE*)malloc(NI * NL * sizeof(DATA_TYPE));
  E_outputFromGpu = (DATA_TYPE*)malloc(NI * NL * sizeof(DATA_TYPE));

  init_array(A, B, C, D);

  mm2Cuda(A, B, C, D, E, E_outputFromGpu);

  t_start = rtclock();
  mm2_cpu(A, B, C, D, E);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
  int res = 0;
  compareResults(E, E_outputFromGpu, res);

  free(C);
  free(A);
  free(B);
  free(D);
  free(E);
  free(E_outputFromGpu);
  if (res == 0)
    return 0;
  else
    return 1;
}
