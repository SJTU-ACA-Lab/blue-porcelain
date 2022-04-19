/**
 * correlation.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

/* Problem size */
#define M 512
#define N 512

/* Thread block dimensions for kernel 1*/
#define DIM_THREAD_BLOCK_KERNEL_1_X 256
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_THREAD_BLOCK_KERNEL_2_X 256
#define DIM_THREAD_BLOCK_KERNEL_2_Y 1

/* Thread block dimensions for kernel 3*/
#define DIM_THREAD_BLOCK_KERNEL_3_X 32
#define DIM_THREAD_BLOCK_KERNEL_3_Y 8

/* Thread block dimensions for kernel 4*/
#define DIM_THREAD_BLOCK_KERNEL_4_X 256
#define DIM_THREAD_BLOCK_KERNEL_4_Y 1

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01f
#define EPS 0.005f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *data) {
  int i, j;

  for (i = 0; i < (M + 1); i++) {
    for (j = 0; j < (N + 1); j++) {
      data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / (M + 1);
    }
  }
}

void correlation(DATA_TYPE *data, DATA_TYPE *mean, DATA_TYPE *stddev,
                 DATA_TYPE *symmat) {
  int i, j, j1, j2;

  // Determine mean of column vectors of input data matrix
  for (j = 1; j < (M + 1); j++) {
    mean[j] = 0.0;

    for (i = 1; i < (N + 1); i++) {
      mean[j] += data[i * (M + 1) + j];
    }

    mean[j] /= (DATA_TYPE)FLOAT_N;
  }

  // Determine standard deviations of column vectors of data matrix.
  for (j = 1; j < (M + 1); j++) {
    stddev[j] = 0.0;

    for (i = 1; i < (N + 1); i++) {
      stddev[j] +=
          (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
    }

    stddev[j] /= FLOAT_N;
    stddev[j] = sqrt_of_array_cell(stddev, j);
    stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
  }

  // Center and reduce the column vectors.
  for (i = 1; i < (N + 1); i++) {
    for (j = 1; j < (M + 1); j++) {
      data[i * (M + 1) + j] -= mean[j];
      data[i * (M + 1) + j] /= (sqrt(FLOAT_N) * stddev[j]);
    }
  }

  // Calculate the m * m correlation matrix.
  for (j1 = 1; j1 < M; j1++) {
    symmat[j1 * (M + 1) + j1] = 1.0;

    for (j2 = j1 + 1; j2 < (M + 1); j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;

      for (i = 1; i < (N + 1); i++) {
        symmat[j1 * (M + 1) + j2] +=
            (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
      }

      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }

  symmat[M * (M + 1) + M] = 1.0;
}

void compareResults(DATA_TYPE *symmat, DATA_TYPE *symmat_outputFromGpu,
                    int &res) {
  int i, j, fail;
  fail = 0;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      if (percentDiff(symmat[i * (N + 1) + j],
                      symmat_outputFromGpu[i * (N + 1) + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
        printf("i: %d j: %d\n1: %f 2: %f\n", i, j, symmat[i * N + j],
               symmat_outputFromGpu[i * N + j]);
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

void GPU_argv_init() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  cudaSetDevice(GPU_DEVICE);
}

__global__ void mean_kernel(DATA_TYPE *mean, DATA_TYPE *data) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if ((j >= 1) && (j < (M + 1))) {
    mean[j] = 0.0;

    int i;
    for (i = 1; i < (N + 1); i++) {
      mean[j] += data[i * (M + 1) + j];
    }

    mean[j] /= (DATA_TYPE)FLOAT_N;
  }
}

__global__ void std_kernel(DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if ((j >= 1) && (j < (M + 1))) {
    std[j] = 0.0;

    int i;
    for (i = 1; i < (N + 1); i++) {
      std[j] +=
          (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
    }
    std[j] /= (FLOAT_N);
    std[j] = sqrt(std[j]);
    if (std[j] <= EPS) {
      std[j] = 1.0;
    }
  }
}

__global__ void reduce_kernel(DATA_TYPE *mean, DATA_TYPE *std,
                              DATA_TYPE *data) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if ((i >= 1) && (i < (N + 1)) && (j >= 1) && (j < (M + 1))) {
    data[i * (M + 1) + j] -= mean[j];
    data[i * (M + 1) + j] /= (sqrt(FLOAT_N) * std[j]);
  }
}

__global__ void corr_kernel(DATA_TYPE *symmat, DATA_TYPE *data) {
  int j1 = blockIdx.x * blockDim.x + threadIdx.x + 1;

  int i, j2;
  if ((j1 >= 1) && (j1 < M)) {
    symmat[j1 * (M + 1) + j1] = 1.0;

    for (j2 = (j1 + 1); j2 < (M + 1); j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;

      for (i = 1; i < (N + 1); i++) {
        symmat[j1 * (M + 1) + j2] +=
            data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
      }
      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }
}

void correlationCuda(DATA_TYPE *data, DATA_TYPE *mean, DATA_TYPE *stddev,
                     DATA_TYPE *symmat, DATA_TYPE *symmat_outputFromGpu) {
  double t_start, t_end;

  DATA_TYPE *data_gpu;
  DATA_TYPE *stddev_gpu;
  DATA_TYPE *mean_gpu;
  DATA_TYPE *symmat_gpu;

  cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * (M + 1) * (N + 1));
  cudaMalloc((void **)&symmat_gpu, sizeof(DATA_TYPE) * (M + 1) * (N + 1));
  cudaMalloc((void **)&stddev_gpu, sizeof(DATA_TYPE) * (M + 1));
  cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * (M + 1));
  cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * (M + 1) * (N + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(symmat_gpu, symmat, sizeof(DATA_TYPE) * (M + 1) * (N + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(stddev_gpu, stddev, sizeof(DATA_TYPE) * (M + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(mean_gpu, mean, sizeof(DATA_TYPE) * (M + 1),
             cudaMemcpyHostToDevice);

  dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
  dim3 grid1((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)),
             1);

  dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
  dim3 grid2((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)),
             1);

  dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
  dim3 grid3((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)),
             (size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_Y)));

  dim3 block4(DIM_THREAD_BLOCK_KERNEL_4_X, DIM_THREAD_BLOCK_KERNEL_4_Y);
  dim3 grid4((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_4_X)),
             1);

  t_start = rtclock();
  mean_kernel<<<grid1, block1>>>(mean_gpu, data_gpu);
  cudaThreadSynchronize();
  std_kernel<<<grid2, block2>>>(mean_gpu, stddev_gpu, data_gpu);
  cudaThreadSynchronize();
  reduce_kernel<<<grid3, block3>>>(mean_gpu, stddev_gpu, data_gpu);
  cudaThreadSynchronize();
  corr_kernel<<<grid4, block4>>>(symmat_gpu, data_gpu);
  cudaThreadSynchronize();
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  DATA_TYPE valueAtSymmatIndexMTimesMPlus1PlusMPoint = 1.0;
  cudaMemcpy(&(symmat_gpu[(M) * (M + 1) + (M)]),
             &valueAtSymmatIndexMTimesMPlus1PlusMPoint, sizeof(DATA_TYPE),
             cudaMemcpyHostToDevice);

  cudaMemcpy(symmat_outputFromGpu, symmat_gpu,
             sizeof(DATA_TYPE) * (M + 1) * (N + 1), cudaMemcpyDeviceToHost);

  cudaFree(data_gpu);
  cudaFree(symmat_gpu);
  cudaFree(stddev_gpu);
  cudaFree(mean_gpu);
}

int main() {
  double t_start, t_end;

  DATA_TYPE *data;
  DATA_TYPE *mean;
  DATA_TYPE *stddev;
  DATA_TYPE *symmat;
  DATA_TYPE *symmat_outputFromGpu;

  data = (DATA_TYPE *)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  mean = (DATA_TYPE *)malloc((M + 1) * sizeof(DATA_TYPE));
  stddev = (DATA_TYPE *)malloc((M + 1) * sizeof(DATA_TYPE));
  symmat = (DATA_TYPE *)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  symmat_outputFromGpu =
      (DATA_TYPE *)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));

  init_arrays(data);

  GPU_argv_init();

  correlationCuda(data, mean, stddev, symmat, symmat_outputFromGpu);

  t_start = rtclock();
  correlation(data, mean, stddev, symmat);
  t_end = rtclock();

  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
  int res = 0;
  compareResults(symmat, symmat_outputFromGpu, res);

  free(data);
  free(mean);
  free(stddev);
  free(symmat);
  free(symmat_outputFromGpu);

  return res == 0 ? 0 : 1;
}
