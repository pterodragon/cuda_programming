#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NDEBUG
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
#else
#define gpuErrchk(ans)                                                         \
  {}
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void matrix_mult_kernel(int *a, int *b, int *c, int m, int n,
                                   int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if (col < n && row < m) {
    for (int i = 0; i < k; i++) {
      sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
  }
}

void matrix_mult(int *a, int *b, int *c, int m, int n, int k) {
  constexpr int BLOCK_WIDTH = 16;
  int n_blocks = ceil(m / (float)BLOCK_WIDTH);
  dim3 dimGrid(n_blocks, n_blocks);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

  cudaError_t err = cudaSuccess;
  printf("[matrix_mult] (%d, %d) * (%d, %d)\n", m, k, k, n);
  int size_A = m * k * sizeof(int);
  int size_B = k * n * sizeof(int);
  int size_C = m * n * sizeof(int);

  int *d_A, *d_B, *d_C;
  err = cudaMalloc((void **)&d_A, size_A);
  gpuErrchk(err);
  err = cudaMalloc((void **)&d_B, size_B);
  gpuErrchk(err);
  err = cudaMalloc((void **)&d_C, size_C);
  gpuErrchk(err);
  err = cudaMemcpy(d_A, a, size_A, cudaMemcpyHostToDevice);
  gpuErrchk(err);
  err = cudaMemcpy(d_B, b, size_B, cudaMemcpyHostToDevice);
  gpuErrchk(err);

  matrix_mult_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
  err = cudaMemcpy(c, d_C, size_C, cudaMemcpyDeviceToHost);
  gpuErrchk(err);
  err = cudaFree(d_A);
  gpuErrchk(err);
  err = cudaFree(d_B);
  gpuErrchk(err);
  err = cudaFree(d_C);
  gpuErrchk(err);
  err = cudaDeviceReset();
  gpuErrchk(err);
}
