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

__global__ void matrix_mult_kernel_tiled(int *d_m, int *d_n, int *d_p, int m,
                                         int n, int k) {
  /*
   * [m][k] @ [k][n] = [m][n]
   */
  constexpr int TILE_WIDTH = 16;
  __shared__ int ds_m[TILE_WIDTH][TILE_WIDTH]; // ds: device shared memory
  __shared__ int ds_n[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  int pvalue = 0;

  for (int i = 0; i < ceil(k / (float)TILE_WIDTH); ++i) {
    // thread collaborative loading into shared memory
    if (row < m && (i * TILE_WIDTH + tx) < k)
      ds_m[ty][tx] = d_m[row * k + i * TILE_WIDTH + tx];
    else
      ds_m[ty][tx] = 0;
    if (col < n && (i * TILE_WIDTH + ty) < k)
      ds_n[ty][tx] = d_n[(i * TILE_WIDTH + ty) * n + col];
    else
      ds_n[ty][tx] = 0;

    __syncthreads();

    for (int j = 0; j < TILE_WIDTH; j++)
      pvalue += ds_m[ty][j] * ds_n[j][tx];
    __syncthreads();
  }

  if (row < m && col < n)
    d_p[row * n + col] = pvalue;
}

void matrix_mult_tiled(int *a, int *b, int *c, int m, int n, int k) {
  /*
   * [m][k] @ [k][n] = [m][n]
   */
  constexpr int BLOCK_WIDTH = 16;
  int n_blocks = ceil(m / (float)BLOCK_WIDTH);
  dim3 dimGrid(n_blocks, n_blocks);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

  cudaError_t err = cudaSuccess;
  printf("[matrix_mult_tiled] (%d, %d) * (%d, %d)\n", m, k, k, n);
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

  matrix_mult_kernel_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
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

void matrix_mult(int *a, int *b, int *c, int m, int n, int k) {
  /*
   * [m][k] @ [k][n] = [m][n]
   */
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

