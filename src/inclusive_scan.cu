#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "util.hpp"
#include "inclusive_scan.hpp"

constexpr const int SECTION_SIZE = 2048;
constexpr const int MAX_SECTIONS = 1024;

__device__ void brent_kung_scan_(float *X, float *Y, int InputSize) {
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int bdx = blockDim.x;

  __shared__ float XY[SECTION_SIZE];
  int i = 2 * bx * bdx + tx;
  if (i < InputSize)
    XY[tx] = X[i];
  if (i + bdx < InputSize)
    XY[tx + bdx] = X[i + bdx];
  for (unsigned int stride = 1; stride <= bdx; stride *= 2) {
    __syncthreads();
    int index = (tx + 1) * 2 * stride - 1;
    if (index < SECTION_SIZE) {
      XY[index] += XY[index - stride];
    }
  }
  for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < SECTION_SIZE) {
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();
  if (i < InputSize)
    Y[i] = XY[tx];
  if (i + bdx < InputSize)
    Y[i + bdx] = XY[tx + bdx];
}

/*
   max InputSize = max blockIdx.x * 2 (set in SECTION_SIZE)
 */
__global__ void brent_kung_scan_kernel(float *X, float *Y, int InputSize) {
  brent_kung_scan_(X, Y, InputSize);
}

__device__ int DCounter;
__device__ int flags[MAX_SECTIONS];
__device__ volatile float scan_value[MAX_SECTIONS];

__global__ void single_pass_inclusive_scan_kernel(float *X, float *Y, int N) {
  const int tx = threadIdx.x;
  const int bdx = blockDim.x;
  brent_kung_scan_(X, Y, N);

  __shared__ int sbid;
  if (tx == 0) {
    // dynamic block index assignment
    sbid = atomicAdd(&DCounter, 1);
  }
  __syncthreads();

  const int bid = sbid;
  float local_sum = Y[(bid + 1) * bdx * 2 - 1];

  __shared__ float previous_sum;
  if (tx == 0) {
    while (atomicAdd(&flags[bid], 0) == 0) {}
    previous_sum = scan_value[bid];
    scan_value[bid + 1] = previous_sum + local_sum;
    __threadfence();
    atomicAdd(&flags[bid + 1], 1);
  }
  __syncthreads(); // make sure previous sum is written by tx == 0
  Y[2 * bdx * bid + tx * 2] += previous_sum;
  Y[2 * bdx * bid + tx * 2 + 1] += previous_sum;
  __syncthreads();
}

void inclusive_scan(float *X, float *Y, int N) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  cudaError_t err = cudaSuccess;

  dim3 dimBlock = SECTION_SIZE / 2;
  dim3 dimGrid = (N + SECTION_SIZE - 1) / SECTION_SIZE;
  int size = N * sizeof(float);

  float *d_X, *d_Y;
  err = cudaMalloc((void **)&d_X, size);
  gpuErrchk(err);
  err = cudaMalloc((void **)&d_Y, size);
  gpuErrchk(err);
  err = cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
  gpuErrchk(err);
  int flag0 = 1;
  err = cudaMemcpyToSymbol(flags, &flag0, sizeof(int));
  gpuErrchk(err);

  printf("dimGrid = %d\n", (N + SECTION_SIZE - 1) / SECTION_SIZE);
  printf("dimBlock = %d\n", SECTION_SIZE / 2);

  single_pass_inclusive_scan_kernel<<<dimGrid, dimBlock>>>(d_X, d_Y, N);

  err = cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);
  gpuErrchk(err);
  err = cudaFree(d_X);
  gpuErrchk(err);
  err = cudaFree(d_Y);
  gpuErrchk(err);
}
