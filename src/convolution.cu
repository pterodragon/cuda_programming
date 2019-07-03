#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "util.hpp"
#include "gpu.hpp"

#define MAX_MASK_WIDTH 5
#define MAX_MASK_SIZE (MAX_MASK_WIDTH * MAX_MASK_WIDTH)
__constant__ float c_M[MAX_MASK_SIZE];
constexpr const int TILE_SIZE = 4;

__global__ void convolution_2D_tiled_kernel(float *P, float *N, int height,
                                            int width, int pitch,
                                            int mask_width) {
  const int halo_w = mask_width / 2; // halo cells width (left or right)
  // TILE_SIZE = blockDim.x = blockDim.y
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int row_o = by * TILE_SIZE + ty;
  const int col_o = bx * TILE_SIZE + tx;

  // for simplicity, halo cells are accessed through global memory directly
  // halo cells to the left and to the right of the current block
  // are cached and trigger caching for the next block respectively.
  // Halo cells up above and down below still need shared memory optimization
  // (not implemented).
  __shared__ float N_ds[TILE_SIZE][TILE_SIZE];
  if (row_o < height && col_o < width) {
    N_ds[ty][tx] = N[row_o * pitch + col_o];
  } else {
    N_ds[ty][tx] = 0.0f;
  }

  __syncthreads();

  const int x_start = col_o - halo_w; // bx * TILE_SIZE + tx - m_w / 2
  const int y_start = row_o - halo_w;

  const int blk_x_a = bx * TILE_SIZE;
  const int blk_x_b = (bx + 1) * TILE_SIZE;
  const int blk_y_a = by * TILE_SIZE;
  const int blk_y_b = (by + 1) * TILE_SIZE;

  float output = 0.0f;
  for (int i = 0; i < mask_width; ++i) {
    for (int j = 0; j < mask_width; ++j) {
      int x_idx = x_start + i;
      int y_idx = y_start + j;
      if (x_idx >= 0 && x_idx < width && //
          y_idx >= 0 && y_idx < height) {
        if (x_idx >= blk_x_a && x_idx < blk_x_b && //
            y_idx >= blk_y_a && y_idx < blk_y_b) {
          output +=
              c_M[j * mask_width + i] * N_ds[ty + j - halo_w][tx + i - halo_w];
        } else {
          output += c_M[j * mask_width + i] * N[y_idx * pitch + x_idx];
        }
      }
    }
  }
  if (row_o < height && col_o < width) {
    P[row_o * pitch + col_o] = output;
  }
}

void print_mat2(float *arr, int m, int n) {
  for (int j = 0; j < m; ++j) {
    std::cout << '[';
    for (int i = 0; i < n - 1; ++i) {
      std::cout << arr[j * n + i] << ", ";
    }
    if (n > 0) {
      std::cout << arr[j * n + n - 1];
    }
    std::cout << "]" << std::endl;
  }
}

void convolution_2D_tiled(float *P, float *N, int height, int width, int pitch,
                          int mask_width, float *M) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  cudaError_t err = cudaSuccess;

  float *d_P, *d_N;
  int size_P = height * pitch * sizeof(float);
  int size_N = height * pitch * sizeof(float);
  err = cudaMalloc((void **)&d_P, size_P);
  gpuErrchk(err);
  err = cudaMalloc((void **)&d_N, size_N);
  gpuErrchk(err);
  err = cudaMemcpy(d_P, P, size_P, cudaMemcpyHostToDevice);
  gpuErrchk(err);
  err = cudaMemcpy(d_N, N, size_N, cudaMemcpyHostToDevice);
  gpuErrchk(err);
  err = cudaMemcpyToSymbol(c_M, M, mask_width * mask_width * sizeof(float));
  gpuErrchk(err);

  int m_blocks = (height + TILE_SIZE - 1) / TILE_SIZE;
  int n_blocks = (width + TILE_SIZE - 1) / TILE_SIZE;
  dim3 dimGrid(m_blocks, n_blocks);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  // call kernel
  printf("m_blocks = %d\n", m_blocks);
  printf("n_blocks = %d\n", n_blocks);
  convolution_2D_tiled_kernel<<<dimGrid, dimBlock>>>(d_P, d_N, height, width,
                                                     pitch, mask_width);
  if (cudaSuccess != cudaGetLastError())
    printf("Error!\n");
  err = cudaMemcpy(P, d_P, size_N, cudaMemcpyDeviceToHost);
  gpuErrchk(err);

  err = cudaFree(d_P);
  gpuErrchk(err);
  err = cudaFree(d_N);
  gpuErrchk(err);
}
