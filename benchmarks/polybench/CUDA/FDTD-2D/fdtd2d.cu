/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include <iostream>

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
#define tmax 10
#define NX 2048
#define NY 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey,
                 DATA_TYPE* hz) {
  int i, j;

  for (i = 0; i < tmax; i++) {
    _fict_[i] = (DATA_TYPE)i;
  }
}

void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz) {
  int t, i, j;

  for (t = 0; t < tmax; t++) {
    for (j = 0; j < NY; j++) {
      ey[0 * NY + j] = _fict_[t];
    }

    for (i = 1; i < NX; i++) {
      for (j = 0; j < NY; j++) {
        ey[i * NY + j] =
            ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
      }
    }

    for (i = 0; i < NX; i++) {
      for (j = 1; j < NY; j++) {
        ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] -
                               0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
      }
    }

    for (i = 0; i < NX; i++) {
      for (j = 0; j < NY; j++) {
        hz[i * NY + j] =
            hz[i * NY + j] -
            0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] +
                   ey[(i + 1) * NY + j] - ey[i * NY + j]);
      }
    }
  }
}

void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2, int& res) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      if (percentDiff(hz1[i * NY + j], hz2[i * NY + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
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

__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE* ex,
                                  DATA_TYPE* ey, DATA_TYPE* hz, int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < NX) && (j < NY)) {
    if (i == 0) {
      ey[i * NY + j] = _fict_[t];
    } else {
      ey[i * NY + j] =
          ey[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
    }
  }
}

__global__ void fdtd_step2_kernel(DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz,
                                  int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < NX) && (j < NY) && (j > 0)) {
    ex[i * (NY + 1) + j] =
        ex[i * (NY + 1) + j] - 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
  }
}

__global__ void fdtd_step3_kernel(DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz,
                                  int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < NX) && (j < NY)) {
    hz[i * NY + j] = hz[i * NY + j] -
                     0.7f * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] +
                             ey[(i + 1) * NY + j] - ey[i * NY + j]);
  }
}

void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz,
              DATA_TYPE* hz_outputFromGpu) {
  double t_start, t_end;

  DATA_TYPE* _fict_gpu;
  DATA_TYPE* ex_gpu;
  DATA_TYPE* ey_gpu;
  DATA_TYPE* hz_gpu;

  cudaMalloc((void**)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
  cudaMalloc((void**)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
  cudaMalloc((void**)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
  cudaMalloc((void**)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);

  cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax,
             cudaMemcpyHostToDevice);
  cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY,
             cudaMemcpyHostToDevice);
  cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil(((float)NY) / ((float)block.x)),
            (size_t)ceil(((float)NX) / ((float)block.y)));

  t_start = rtclock();

  for (int t = 0; t < tmax; t++) {
    fdtd_step1_kernel<<<grid, block>>>(_fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
    cudaThreadSynchronize();
    fdtd_step2_kernel<<<grid, block>>>(ex_gpu, ey_gpu, hz_gpu, t);
    cudaThreadSynchronize();
    fdtd_step3_kernel<<<grid, block>>>(ex_gpu, ey_gpu, hz_gpu, t);
    cudaThreadSynchronize();
  }

  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY,
             cudaMemcpyDeviceToHost);

  cudaFree(_fict_gpu);
  cudaFree(ex_gpu);
  cudaFree(ey_gpu);
  cudaFree(hz_gpu);
}

int main() {
  double t_start, t_end;

  DATA_TYPE* _fict_;
  DATA_TYPE* ex;
  DATA_TYPE* ey;
  DATA_TYPE* hz;
  DATA_TYPE* hz_outputFromGpu;

  _fict_ = (DATA_TYPE*)malloc(tmax * sizeof(DATA_TYPE));
  ex = (DATA_TYPE*)malloc(NX * (NY + 1) * sizeof(DATA_TYPE));
  ey = (DATA_TYPE*)malloc((NX + 1) * NY * sizeof(DATA_TYPE));
  hz = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));
  hz_outputFromGpu = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));

  init_arrays(_fict_, ex, ey, hz);

  GPU_argv_init();
  fdtdCuda(_fict_, ex, ey, hz, hz_outputFromGpu);

  t_start = rtclock();
  runFdtd(_fict_, ex, ey, hz);
  t_end = rtclock();

  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
  int res = 0;
  compareResults(hz, hz_outputFromGpu, res);

  free(_fict_);
  free(ex);
  free(ey);
  free(hz);
  free(hz_outputFromGpu);

  return res == 0 ? 0 : 1;
}
