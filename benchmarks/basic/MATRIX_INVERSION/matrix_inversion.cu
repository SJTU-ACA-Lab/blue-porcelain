// Copyright 2022, ACALab of SJTU

#include <math.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

#define blocksize 8

/*storing matrix*/
void matrix_read(float *L, int dimension) {
  FILE *fp;
  int row, col;

  fp = fopen("randomMatrix_100.input", "r");  // open output file
  if (fp == NULL)                             // open failed
    return;

  for (row = 0; row < dimension; row++) {
    for (col = 0; col < dimension; col++)
      if (fscanf(fp, "%f,", &L[row * dimension + col]) == EOF)
        break;  // read data

    if (feof(fp)) break;  // if the file is over
  }

  fclose(fp);  // close file
}

__global__ void nodiag_normalize(float *A, float *I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n)
    if (x == i && x != y) {
      I[x * n + y] /= A[i * n + i];
      A[x * n + y] /= A[i * n + i];
    }
}

__global__ void diag_normalize(float *A, float *I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n)
    if (x == y && x == i) {
      I[x * n + y] /= A[i * n + i];
      A[x * n + y] /= A[i * n + i];
    }
}

__global__ void gaussjordan(float *A, float *I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n) {
    if (x != i) {
      I[x * n + y] -= I[i * n + y] * A[x * n + i];
      if (y != i) {
        A[x * n + y] -= A[i * n + y] * A[x * n + i];
      }
    }
  }
}

__global__ void set_zero(float *A, float *I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n) {
    if (x != i) {
      if (y == i) {
        A[x * n + y] = 0;
      }
    }
  }
}

void savetofile(float *A, string s, int n, int h) {
  std::ofstream plik;
  plik.open(s);

  for (int j = 0; j < h; j++) {
    for (int i = 0; i < h; i++) {
      plik << A[j * n + i] << "\t";
    }
    plik << endl;
  }
  plik.close();
}

int main() {
  const int n = 100;
  // creating input
  float *iL = new float[n * n];
  float *L = new float[n * n];
  matrix_read(L, n);

  cout << "inv\n";
  float *d_A, *d_L, *I, *dI;
  int ddsize = n * n * sizeof(float);

  dim3 threadsPerBlock(blocksize, blocksize);
  dim3 numBlocks((n + blocksize - 1) / blocksize,
                 (n + blocksize - 1) / blocksize);
  // memory allocation
  cudaMalloc((void **)&d_A, ddsize);
  cudaMalloc((void **)&dI, ddsize);
  I = new float[n * n];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j)
        I[i * n + i] = 1.0;
      else
        I[i * n + j] = 0.0;
    }
  }

  // copy data from CPU to GPU
  cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
  cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);

  // L^(-1)
  for (int i = 0; i < n; i++) {
    nodiag_normalize<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
    diag_normalize<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
    gaussjordan<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
    set_zero<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
  }

  // copy data from GPU to CPU
  cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);

  savetofile(iL, "inv.txt", n, n);
  // savetofile(I, "I.txt", n, n);
  cudaFree(d_A);
  cudaFree(dI);

  float *c = new float[n * n];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      c[i * n + j] = 0;  // put the initial value to zero
      for (int x = 0; x < n; x++)
        c[i * n + j] = c[i * n + j] +
                       L[i * n + x] * iL[x * n + j];  // matrix multiplication
    }
  savetofile(c, "c.txt", n, n);

  delete[] I;
  delete[] L;
  delete[] iL;

  return 0;
}