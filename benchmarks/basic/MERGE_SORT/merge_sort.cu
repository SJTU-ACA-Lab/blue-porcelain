// Copyright 2022, ACALab of SJTU

#include <stdio.h>
#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// data[], size, threads, blocks,
bool mergesort(long *, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(long *, long *, long, long, long, dim3 *, dim3 *);
__device__ void gpu_bottomUpMerge(long *, long *, long, long, long);

#define min(a, b) (a < b ? a : b)

void init_data(long *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = static_cast<long>(rand() % 1024);
  }
}

int main(int argc, char **argv) {
  dim3 threadsPerBlock(BLOCK_SIZE);
  dim3 blocksPerGrid(1024);

  int size = 1 << 11;

  long *data = (long *)malloc(sizeof(long) * size);

  init_data(data, size);

  // merge-sort the data
  bool pass = true;
  pass = mergesort(data, size, threadsPerBlock, blocksPerGrid);
  free(data);

  if (!pass) {
    printf("Calculation Error!\n");
    return 1;
  }

  printf("Calculation right!\n");
  return 0;
}

bool mergesort(long *data, long size, dim3 threadsPerBlock,
               dim3 blocksPerGrid) {
  //
  // Allocate two arrays on the GPU
  // we switch back and forth between them during the sort
  //
  long *D_data;
  long *D_swp;
  dim3 *D_threads;
  dim3 *D_blocks;

  // Actually allocate the two arrays
  cudaMalloc((void **)&D_data, size * sizeof(long));
  cudaMalloc((void **)&D_swp, size * sizeof(long));

  // Copy from our input list into the first array
  cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice);

  // Copy the thread / block info to the GPU as well
  cudaMalloc((void **)&D_threads, sizeof(dim3));
  cudaMalloc((void **)&D_blocks, sizeof(dim3));

  cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
  cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

  long *A = D_data;
  long *B = D_swp;

  long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                  blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

  //
  // Slice up the list and give pieces of it to each thread, letting the pieces
  // grow bigger and bigger until the whole list is sorted
  //
  for (int width = 2; width < (size << 1); width <<= 1) {
    long slices = size / ((nThreads)*width) + 1;

    gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices,
                                                      D_threads, D_blocks);

    // Switch the input / output arrays instead of copying them around
    A = A == D_data ? D_swp : D_data;
    B = B == D_data ? D_swp : D_data;
  }

  cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost);

  bool pass = true;
  for (int i = 0; i < size - 1; i++) {
    if (data[i] > data[i + 1]) {
      pass = false;
      break;
    }
  }
  // Free the GPU memory
  cudaFree(A);
  cudaFree(B);

  return pass;
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3 *threads, dim3 *blocks) {
  int x;
  return threadIdx.x + threadIdx.y * (x = threads->x) +
         threadIdx.z * (x *= threads->y) + blockIdx.x * (x *= threads->z) +
         blockIdx.y * (x *= blocks->z) + blockIdx.z * (x *= blocks->y);
}

__global__ void gpu_mergesort(long *source, long *dest, long size, long width,
                              long slices, dim3 *threads, dim3 *blocks) {
  unsigned int idx = getIdx(threads, blocks);
  long start = width * idx * slices, middle, end;

  for (long slice = 0; slice < slices; slice++) {
    if (start >= size) break;

    middle = min(start + (width >> 1), size);
    end = min(start + width, size);
    // gpu_bottomUpMerge(source, dest, start, middle, end);
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
      bool cond = i < middle && (j >= end || source[i] < source[j]);
      if (cond) {
        dest[k] = source[i];
        i++;
      } else {
        dest[k] = source[j];
        j++;
      }
    }

    start += width;
  }
}