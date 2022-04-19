// Copyright 2022, ACALab of SJTU

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

using namespace std;

#define SIZEOFBLOCK 256

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

void printToFile(int* arr, int n) {
  ofstream fstream;
  fstream.open("./output.txt");
  for (int i = 0; i < n; i++) {
    fstream << arr[i] << endl;
  }
}

void printArr(int* arr, int n) {
  for (int i = 0; i < n; i++) {
    cout << arr[i] << endl;
  }
}

void rng(int* arr, int n) {
  int seed = 13516154;
  srand(seed);
  for (long i = 0; i < n; i++) {
    arr[i] = (int)rand();
  }
}

// parallel radix sort
// get specific bit at index = idx
__global__ void generateFlag(int* flag, int* arr, int n, int idx) {
  // parallel
  for (int i = 0; i < n; i++) {
    if ((arr[i] >> idx) & 1 == 1) {
      flag[i] = 1;
    } else {
      flag[i] = 0;
    }
  }
}

// create I-down array
int* generateIDown(int* flag, int n) {
  int* iDown = (int*)malloc(n * sizeof(int));
  int val = 0;

  iDown[0] = val;
  for (int i = 1; i < n; i++) {
    if (flag[i - 1] == 0) {
      val++;
    }
    iDown[i] = val;
  }
  return iDown;
}

// create I-up array
int* generateIUp(int* flag, int n) {
  int* iUp = (int*)malloc(n * sizeof(int));
  int val = n - 1;

  iUp[n - 1] = val;
  for (int i = n - 2; i >= 0; i--) {
    if (flag[i + 1] == 1) {
      val--;
    }
    iUp[i] = val;
  }
  return iUp;
}

__global__ void generateShouldIndex(int* shouldIndex, int* flag, int* iDown,
                                    int* iUp, int n) {
  // parallel
  for (int i = 0; i < n; i++) {
    if (flag[i] == 0) {
      shouldIndex[i] = iDown[i];
    } else {
      shouldIndex[i] = iUp[i];
    }
  }
}

void permute(int* arr, int* flag, int* iDown, int* iUp, int n) {
  int* shouldArr = (int*)malloc(n * sizeof(int));
  int numBlocks = (n + SIZEOFBLOCK - 1) / SIZEOFBLOCK;

  int* d_flag;

  int* h_shouldIndex = (int*)malloc(n * sizeof(int));
  int* d_shouldIndex;
  int* d_iDown;
  int* d_iUp;

  cudaMalloc(&d_shouldIndex, n * sizeof(int));
  cudaMalloc(&d_flag, n * sizeof(int));
  cudaMalloc(&d_iDown, n * sizeof(int));
  cudaMalloc(&d_iUp, n * sizeof(int));

  cudaMemcpy(d_flag, flag, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_iDown, iDown, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_iUp, iUp, n * sizeof(int), cudaMemcpyHostToDevice);
  generateShouldIndex<<<numBlocks, SIZEOFBLOCK>>>(d_shouldIndex, d_flag,
                                                  d_iDown, d_iUp, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_shouldIndex, d_shouldIndex, n * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaFree(d_flag);
  cudaFree(d_iDown);
  cudaFree(d_iUp);
  cudaFree(d_shouldIndex);

  // parallel
  for (int i = 0; i < n; i++) {
    shouldArr[h_shouldIndex[i]] = arr[i];
  }

  // parallel
  for (int i = 0; i < n; i++) {
    arr[i] = shouldArr[i];
  }
}

void split(int* arr, int n, int idx) {
  int numBlocks = (n + SIZEOFBLOCK - 1) / SIZEOFBLOCK;

  int* h_flag = (int*)malloc(n * sizeof(int));
  int* d_flag;

  int* d_arr;

  cudaMalloc(&d_flag, n * sizeof(int));
  cudaMalloc(&d_arr, n * sizeof(int));

  cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

  generateFlag<<<numBlocks, SIZEOFBLOCK>>>(d_flag, d_arr, n, idx);
  cudaDeviceSynchronize();

  cudaMemcpy(h_flag, d_flag, n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_flag);
  cudaFree(d_arr);

  int* iDown = generateIDown(h_flag, n);
  int* iUp = generateIUp(h_flag, n);

  permute(arr, h_flag, iDown, iUp, n);
}

void radixSort(int* arr, int n) {
  int idx = 0;

  for (int i = 0; i < 32; i++) {
    split(arr, n, i);
  }
}

int main(int argc, char** argv) {
  int n = 256;

  int* arr = (int*)malloc(n * sizeof(int));

  rng(arr, n);

  clock_t beginTime = clock();
  radixSort(arr, n);
  clock_t endTime = clock();

  double elapsedTime = (double)endTime - beginTime / CLOCKS_PER_SEC;
  int error = 0;
  bool all_zero = true;
  for (int i = 1; i < n; ++i) {
    if (arr[i] < arr[i - 1]) error += 1;
    if (arr[i] != 0) all_zero = false;
  }
  printToFile(arr, n);

  cout << "Parallel Radix Sort Time: " << elapsedTime << endl;
  cout << endl;

  if (error || all_zero) {
    printf("Calculation Error!\n");
    return 1;
  }

  printf("Calculation right!\n");
  return 0;
}
