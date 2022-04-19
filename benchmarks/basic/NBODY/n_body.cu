// origin: https://github.com/9prady9/CUDA/blob/master/nBodyComputation.cu

#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define MAX_BLKS 65537
#define MAX_THRDS_P_BLK 512

struct Vector {
  float x;
  float y;
  float z;
  __device__ float d_influenceBy(Vector p) {
    return 1 / sqrt((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y) +
                    (z - p.z) * (z - p.z));
  }
  __host__ float h_influenceBy(Vector p) {
    return 1 / sqrt((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y) +
                    (z - p.z) * (z - p.z));
  }
};

__host__ int ciel(float value) {
  float mantissa = value - (int)value;
  return ((int)value + (mantissa == 0 ? 0 : 1));
}

/**
 * Kernel is launched with as many blocks as bodies. Not an optimal strategy but
 * first iteration. In each block, if number of threads is a power of two which
 * is computed based on number of bodies given Hence, the binary reduction in
 * kernel is much simpler since thread load is power of two. Each thread
 * computes forces of <bodiesPerThread> bodies on itself, followed by a
 * reduction to compute overall force.
 */
__global__ void forceComp(Vector *positions, int bodyCount,
                          float *resultantForce, int bodiesPerThread) {
  extern __shared__ float perBlockCache[];
  int tid = threadIdx.x * bodiesPerThread;
  int Limit = tid + bodiesPerThread;

  if (tid < bodyCount) {
    perBlockCache[threadIdx.x] = 0.0;
    while (tid < Limit) {
      if (blockIdx.x != tid)
        perBlockCache[threadIdx.x] +=
            positions[blockIdx.x].d_influenceBy(positions[tid]);
      tid++;
    }
    __syncthreads();

    /* now do reduction by addition for the resultant
     * force on body with Id = blockIdx.x */
    int reduceDim = blockDim.x / 2;
    while (reduceDim > 0) {
      if (threadIdx.x < reduceDim)
        perBlockCache[threadIdx.x] += perBlockCache[threadIdx.x + reduceDim];
      __syncthreads();
      reduceDim /= 2;
    }
    if (threadIdx.x == 0) resultantForce[blockIdx.x] = perBlockCache[0];
  }
}

/**
 * Program requires a numerical input i.e. the number
 * of bodies participating in the n-body simulation
 * This number should be less than MAX_BLKS = 65537
 */
int main(int argc, char *argv[]) {
  int host_bodyCount = 256;

  size_t res_size;
  int iter;
  float tempResult;
  Vector *host_positions;
  float *host_resultantForce;
  Vector *dev_positions;
  float *dev_resultantForce;

  size_t size = host_bodyCount * sizeof(Vector);
  int blocksPerGrid = host_bodyCount;
  int thrdCntHold = MAX_THRDS_P_BLK;
  for (int i = 1; host_bodyCount < thrdCntHold && thrdCntHold > 2; ++i)
    thrdCntHold >>= 1;
  int threadsPerBlock = thrdCntHold;
  int bodiesPerThread = ciel((float)host_bodyCount / threadsPerBlock);
  res_size = threadsPerBlock * sizeof(float);

  printf("Blocks per Grid: %d\nThreads per Block: %d\n", blocksPerGrid,
         threadsPerBlock);
  srand(time(NULL));

  /* Allocate host memory to prepare data */
  host_positions = (Vector *)malloc(size);
  host_resultantForce = (float *)malloc(res_size);
  for (iter = 0; iter < host_bodyCount; iter++) {
    host_positions[iter].x = iter + 1.0;
    host_positions[iter].y = iter + 1.0;
    host_positions[iter].z = iter + 1.0;
    // printf("Body %d position is (%f,%f,%f)\n", iter+1,
    // host_positions[iter].x, host_positions[iter].y, host_positions[iter].z
    // );
  }

  /* Allocate device memory, GPU memory */
  cudaMalloc((void **)&dev_positions, size);
  cudaMalloc((void **)&dev_resultantForce, res_size);

  /* Copy data from host to device */
  cudaMemcpy(dev_positions, host_positions, size, cudaMemcpyHostToDevice);
  forceComp<<<blocksPerGrid, threadsPerBlock, res_size>>>(
      dev_positions, host_bodyCount, dev_resultantForce, bodiesPerThread);

  /* Copy result from device to host */
  cudaMemcpy(host_resultantForce, dev_resultantForce, res_size,
             cudaMemcpyDeviceToHost);

  /* Compute on host for comparison */
  int error = 0;
  for (int i = 0; i < host_bodyCount; i++) {
    tempResult = 0.0;
    for (iter = 0; iter < host_bodyCount; iter++) {
      if (iter != i)
        tempResult =
            tempResult + host_positions[i].h_influenceBy(host_positions[iter]);
    }
    printf("Force on me(%d) is %.3f; Device result is %.3f\n", i + 1,
           tempResult, host_resultantForce[i]);
    if (abs((tempResult - host_resultantForce[i]) / host_resultantForce[i]) >
        1e-3)
      error = 1;
  }

  /* clear all memory  */
  cudaFree(dev_positions);
  cudaFree(dev_resultantForce);
  free(host_positions);
  free(host_resultantForce);

  if (error == 1) {
    printf("Calculation Error!\n");
    return 1;
  }

  printf("Calculation right!\n");
  return 0;
}