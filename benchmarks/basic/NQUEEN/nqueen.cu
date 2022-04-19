// N-queen for CUDA
// origin:
// https://github.com/charitha22/cgo22ae-darm-benchmarks/tree/main/benchmarks/NQU
// Copyright(c) 2008 Ping-Che Chen

#include <stdio.h>
#include <iostream>

#define THREAD_NUM 96

/* -------------------------------------------------------------------
 * This is a non-recursive version of n-queen backtracking solver.
 * This provides the basis for the CUDA version.
 * -------------------------------------------------------------------
 */

long long solve_nqueen(int n) {
  unsigned int mask[32];
  unsigned int l_mask[32];
  unsigned int r_mask[32];
  unsigned int m[32];

  if (n <= 0 || n > 32) {
    return 0;
  }

  const unsigned int t_mask = (1 << n) - 1;
  long long total = 0;
  long long upper_total = 0;
  int i = 0, j;
  unsigned int index;

  mask[0] = 0;
  l_mask[0] = 0;
  r_mask[0] = 0;
  m[0] = 0;

  for (j = 0; j < (n + 1) / 2; j++) {
    index = (1 << j);
    m[0] |= index;

    mask[1] = index;
    l_mask[1] = index << 1;
    r_mask[1] = index >> 1;
    m[1] = (mask[1] | l_mask[1] | r_mask[1]);
    i = 1;

    if (n % 2 == 1 && j == (n + 1) / 2 - 1) {
      upper_total = total;
      total = 0;
    }

    while (i > 0) {
      if ((m[i] & t_mask) == t_mask) {
        i--;
      } else {
        index = ((m[i] + 1) ^ m[i]) & ~m[i];
        m[i] |= index;
        if ((index & t_mask) != 0) {
          if (i + 1 == n) {
            total++;
            i--;
          } else {
            mask[i + 1] = mask[i] | index;
            l_mask[i + 1] = (l_mask[i] | index) << 1;
            r_mask[i + 1] = (r_mask[i] | index) >> 1;
            m[i + 1] = (mask[i + 1] | l_mask[i + 1] | r_mask[i + 1]);
            i++;
          }
        } else {
          i--;
        }
      }
    }
  }

  if (n % 2 == 0) {
    return total * 2;
  } else {
    return upper_total * 2 + total;
  }
}

/* --------------------------------------------------------------------------
 * This is a non-recursive version of n-queen backtracking solver for CUDA.
 * It receives multiple initial conditions from a CPU iterator, and count
 * each conditions.
 * --------------------------------------------------------------------------
 */

__global__ void solve_nqueen_cuda_kernel(
    int n, int mark, unsigned int* total_masks, unsigned int* total_l_masks,
    unsigned int* total_r_masks, unsigned int* results, int total_conditions) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid * blockDim.x + tid;

  __shared__ unsigned int mask[THREAD_NUM][10];
  __shared__ unsigned int l_mask[THREAD_NUM][10];
  __shared__ unsigned int r_mask[THREAD_NUM][10];
  __shared__ unsigned int m[THREAD_NUM][10];

  __shared__ unsigned int sum[THREAD_NUM];

  const unsigned int t_mask = (1 << n) - 1;
  int total = 0;
  int i = 0;
  unsigned int index;

  if (idx < total_conditions) {
    mask[tid][i] = total_masks[idx];
    l_mask[tid][i] = total_l_masks[idx];
    r_mask[tid][i] = total_r_masks[idx];
    m[tid][i] = mask[tid][i] | l_mask[tid][i] | r_mask[tid][i];

    while (i >= 0) {
      if ((m[tid][i] & t_mask) == t_mask) {
        i--;
      } else {
        index = (m[tid][i] + 1) & ~m[tid][i];
        m[tid][i] |= index;
        if ((index & t_mask) != 0) {
          if (i + 1 == mark) {
            total++;
            i--;
          } else {
            mask[tid][i + 1] = mask[tid][i] | index;
            l_mask[tid][i + 1] = (l_mask[tid][i] | index) << 1;
            r_mask[tid][i + 1] = (r_mask[tid][i] | index) >> 1;
            m[tid][i + 1] =
                (mask[tid][i + 1] | l_mask[tid][i + 1] | r_mask[tid][i + 1]);
            i++;
          }
        } else {
          i--;
        }
      }
    }

    sum[tid] = total;
  } else {
    sum[tid] = 0;
  }

  __syncthreads();

  // reduction
  if (tid < 64 && tid + 64 < THREAD_NUM) {
    sum[tid] += sum[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    sum[tid] += sum[tid + 32];
  }
  __syncthreads();
  if (tid < 16) {
    sum[tid] += sum[tid + 16];
  }
  __syncthreads();
  if (tid < 8) {
    sum[tid] += sum[tid + 8];
  }
  __syncthreads();
  if (tid < 4) {
    sum[tid] += sum[tid + 4];
  }
  __syncthreads();
  if (tid < 2) {
    sum[tid] += sum[tid + 2];
  }
  __syncthreads();
  if (tid < 1) {
    sum[tid] += sum[tid + 1];
  }
  __syncthreads();

  if (tid == 0) {
    results[bid] = sum[0];
  }
}

long long solve_nqueen_cuda(int n, int steps) {
  // generating start conditions
  unsigned int mask[32];
  unsigned int l_mask[32];
  unsigned int r_mask[32];
  unsigned int m[32];
  unsigned int index;

  if (n <= 0 || n > 32) {
    return 0;
  }

  unsigned int* total_masks = new unsigned int[steps];
  unsigned int* total_l_masks = new unsigned int[steps];
  unsigned int* total_r_masks = new unsigned int[steps];
  unsigned int* results = new unsigned int[steps];

  unsigned int* masks_cuda;
  unsigned int* l_masks_cuda;
  unsigned int* r_masks_cuda;
  unsigned int* results_cuda;

  cudaMalloc((void**)&masks_cuda, sizeof(int) * steps);
  cudaMalloc((void**)&l_masks_cuda, sizeof(int) * steps);
  cudaMalloc((void**)&r_masks_cuda, sizeof(int) * steps);
  cudaMalloc((void**)&results_cuda, sizeof(int) * steps / THREAD_NUM);

  const unsigned int t_mask = (1 << n) - 1;
  const unsigned int mark = n > 11 ? n - 10 : 2;
  long long total = 0;
  int total_conditions = 0;
  int i = 0, j;

  mask[0] = 0;
  l_mask[0] = 0;
  r_mask[0] = 0;
  m[0] = 0;

  bool computed = false;

  for (j = 0; j < n / 2; j++) {
    index = (1 << j);
    m[0] |= index;

    mask[1] = index;
    l_mask[1] = index << 1;
    r_mask[1] = index >> 1;
    m[1] = (mask[1] | l_mask[1] | r_mask[1]);
    i = 1;

    while (i > 0) {
      if ((m[i] & t_mask) == t_mask) {
        i--;
      } else {
        index = (m[i] + 1) & ~m[i];
        m[i] |= index;
        if ((index & t_mask) != 0) {
          mask[i + 1] = mask[i] | index;
          l_mask[i + 1] = (l_mask[i] | index) << 1;
          r_mask[i + 1] = (r_mask[i] | index) >> 1;
          m[i + 1] = (mask[i + 1] | l_mask[i + 1] | r_mask[i + 1]);
          i++;
          if (i == mark) {
            total_masks[total_conditions] = mask[i];
            total_l_masks[total_conditions] = l_mask[i];
            total_r_masks[total_conditions] = r_mask[i];
            total_conditions++;
            if (total_conditions == steps) {
              if (computed) {
                cudaMemcpy(results, results_cuda,
                           sizeof(int) * steps / THREAD_NUM,
                           cudaMemcpyDeviceToHost);

                for (int j = 0; j < steps / THREAD_NUM; j++) {
                  total += results[j];
                }

                computed = false;
              }

              // start computation
              cudaMemcpy(masks_cuda, total_masks,
                         sizeof(int) * total_conditions,
                         cudaMemcpyHostToDevice);
              cudaMemcpy(l_masks_cuda, total_l_masks,
                         sizeof(int) * total_conditions,
                         cudaMemcpyHostToDevice);
              cudaMemcpy(r_masks_cuda, total_r_masks,
                         sizeof(int) * total_conditions,
                         cudaMemcpyHostToDevice);

              solve_nqueen_cuda_kernel<<<steps / THREAD_NUM, THREAD_NUM>>>(
                  n, n - mark, masks_cuda, l_masks_cuda, r_masks_cuda,
                  results_cuda, total_conditions);

              computed = true;

              total_conditions = 0;
            }
            i--;
          }
        } else {
          i--;
        }
      }
    }
  }

  if (computed) {
    cudaMemcpy(results, results_cuda, sizeof(int) * steps / THREAD_NUM,
               cudaMemcpyDeviceToHost);

    for (int j = 0; j < steps / THREAD_NUM; j++) {
      total += results[j];
    }

    computed = false;
  }

  cudaMemcpy(masks_cuda, total_masks, sizeof(int) * total_conditions,
             cudaMemcpyHostToDevice);
  cudaMemcpy(l_masks_cuda, total_l_masks, sizeof(int) * total_conditions,
             cudaMemcpyHostToDevice);
  cudaMemcpy(r_masks_cuda, total_r_masks, sizeof(int) * total_conditions,
             cudaMemcpyHostToDevice);

  solve_nqueen_cuda_kernel<<<steps / THREAD_NUM, THREAD_NUM>>>(
      n, n - mark, masks_cuda, l_masks_cuda, r_masks_cuda, results_cuda,
      total_conditions);

  cudaMemcpy(results, results_cuda, sizeof(int) * steps / THREAD_NUM,
             cudaMemcpyDeviceToHost);

  for (int j = 0; j < steps / THREAD_NUM; j++) {
    total += results[j];
  }

  total *= 2;

  if (n % 2 == 1) {
    computed = false;
    total_conditions = 0;

    index = (1 << (n - 1) / 2);
    m[0] |= index;

    mask[1] = index;
    l_mask[1] = index << 1;
    r_mask[1] = index >> 1;
    m[1] = (mask[1] | l_mask[1] | r_mask[1]);
    i = 1;

    while (i > 0) {
      if ((m[i] & t_mask) == t_mask) {
        i--;
      } else {
        index = (m[i] + 1) & ~m[i];
        m[i] |= index;
        if ((index & t_mask) != 0) {
          mask[i + 1] = mask[i] | index;
          l_mask[i + 1] = (l_mask[i] | index) << 1;
          r_mask[i + 1] = (r_mask[i] | index) >> 1;
          m[i + 1] = (mask[i + 1] | l_mask[i + 1] | r_mask[i + 1]);
          i++;
          if (i == mark) {
            total_masks[total_conditions] = mask[i];
            total_l_masks[total_conditions] = l_mask[i];
            total_r_masks[total_conditions] = r_mask[i];
            total_conditions++;
            if (total_conditions == steps) {
              if (computed) {
                cudaMemcpy(results, results_cuda,
                           sizeof(int) * steps / THREAD_NUM,
                           cudaMemcpyDeviceToHost);

                for (int j = 0; j < steps / THREAD_NUM; j++) {
                  total += results[j];
                }

                computed = false;
              }

              // start computation
              cudaMemcpy(masks_cuda, total_masks,
                         sizeof(int) * total_conditions,
                         cudaMemcpyHostToDevice);
              cudaMemcpy(l_masks_cuda, total_l_masks,
                         sizeof(int) * total_conditions,
                         cudaMemcpyHostToDevice);
              cudaMemcpy(r_masks_cuda, total_r_masks,
                         sizeof(int) * total_conditions,
                         cudaMemcpyHostToDevice);

              solve_nqueen_cuda_kernel<<<steps / THREAD_NUM, THREAD_NUM>>>(
                  n, n - mark, masks_cuda, l_masks_cuda, r_masks_cuda,
                  results_cuda, total_conditions);

              computed = true;

              total_conditions = 0;
            }
            i--;
          }
        } else {
          i--;
        }
      }
    }

    if (computed) {
      cudaMemcpy(results, results_cuda, sizeof(int) * steps / THREAD_NUM,
                 cudaMemcpyDeviceToHost);

      for (int j = 0; j < steps / THREAD_NUM; j++) {
        total += results[j];
      }

      computed = false;
    }

    cudaMemcpy(masks_cuda, total_masks, sizeof(int) * total_conditions,
               cudaMemcpyHostToDevice);
    cudaMemcpy(l_masks_cuda, total_l_masks, sizeof(int) * total_conditions,
               cudaMemcpyHostToDevice);
    cudaMemcpy(r_masks_cuda, total_r_masks, sizeof(int) * total_conditions,
               cudaMemcpyHostToDevice);

    solve_nqueen_cuda_kernel<<<steps / THREAD_NUM, THREAD_NUM>>>(
        n, n - mark, masks_cuda, l_masks_cuda, r_masks_cuda, results_cuda,
        total_conditions);

    cudaMemcpy(results, results_cuda, sizeof(int) * steps / THREAD_NUM,
               cudaMemcpyDeviceToHost);

    for (int j = 0; j < steps / THREAD_NUM; j++) {
      total += results[j];
    }
  }

  cudaFree(masks_cuda);
  cudaFree(l_masks_cuda);
  cudaFree(r_masks_cuda);
  cudaFree(results_cuda);

  delete[] total_masks;
  delete[] total_l_masks;
  delete[] total_r_masks;
  delete[] results;

  return total;
}

int main(int argc, char** argv) {
  int n = 8;
  long long solution_cpu, solution_gpu;
  int argstart = 1, steps = 24576;

  if (argc < argstart + 1) {
    printf("Usage: %s n steps\n", argv[0]);
    printf("  n: n-queen\n");
    printf("  steps: step number for GPU\n");
    printf("Default to 8 queen\n");
  } else {
    n = atoi(argv[argstart]);
    if (n <= 1 || n > 32) {
      printf("Invalid n, n should be > 1 and <= 32\n");
      printf("Note: n > 18 will require a very very long time to compute!\n");
      return 0;
    }

    if (argc >= argstart + 2) {
      steps = atoi(argv[argstart + 1]);
      if (steps <= THREAD_NUM || steps % THREAD_NUM != 0) {
        printf("Invalid step, step should be multiple of %d\n", THREAD_NUM);
        return 0;
      }
    }
  }
  solution_cpu = solve_nqueen(n);
  solution_gpu = solve_nqueen_cuda(n, steps);

  if (solution_cpu != solution_gpu) {
    printf("Calculation Error!\n");
    return 1;
  }

  printf("Calculation right!\n");
  return 0;
}