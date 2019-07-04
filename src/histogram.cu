#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "histogram.hpp"
#include "util.hpp"

__global__ void histogram_privatized_kernel(unsigned char *input,
                                            unsigned int *bins,
                                            unsigned int num_elements,
                                            unsigned int num_bins) {
  const int bx = blockIdx.x;
  const int bdx = blockDim.x;
  const int tx = threadIdx.x;
  const int gdx = gridDim.x;
  unsigned int tid = bx * bdx + tx;

  extern __shared__ unsigned int histo_s[]; // size is 3rd arg in <<< >>> of kernel
  for (unsigned int bin_idx = tx; bin_idx < num_bins; bin_idx += bdx) {
    histo_s[bin_idx] = 0u;
  }
  __syncthreads();

  const int bin_size = (num_elements - 1) / num_bins + 1;
  for (unsigned int i = tid; i < num_elements; i += bdx * gdx) {
    int c = input[i] - 'a';
    if (c >= 0 && c < 26)
      atomicAdd(&(histo_s[c / bin_size]), 1);
  }
  __syncthreads();

  for (unsigned int bin_idx = tx; bin_idx < num_bins; bin_idx += bdx) {
    atomicAdd(&(bins[bin_idx]), histo_s[bin_idx]);
  }
}

void histogram(unsigned char *input, unsigned int *histo, unsigned int N,
               unsigned int N_bins) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  printf("input = %s, N = %d, N_bins = %d\n", input, N, N_bins);
  cudaError_t err = cudaSuccess;

  unsigned char *d_in;
  unsigned int *d_out;
  int size_N = N * sizeof(unsigned char);
  int size_out = N_bins * sizeof(unsigned int);
  err = cudaMalloc((void **)&d_in, size_N);
  gpuErrchk(err);
  err = cudaMalloc((void **)&d_out, size_out);
  gpuErrchk(err);
  err = cudaMemcpy(d_in, input, size_N, cudaMemcpyHostToDevice);
  gpuErrchk(err);
  err = cudaMemset(d_out, 0, size_out);
  gpuErrchk(err);

  dim3 dimBlock = 8;
  dim3 dimGrid = 2;

  histogram_privatized_kernel<<<dimGrid, dimBlock, N_bins>>>(d_in, d_out, N,
                                                             N_bins);

  err = cudaMemcpy(histo, d_out, size_out, cudaMemcpyDeviceToHost);
  gpuErrchk(err);
  err = cudaFree(d_in);
  gpuErrchk(err);
  err = cudaFree(d_out);
  gpuErrchk(err);
}
