#include <iostream>
#include "gpu.hpp"

#ifndef NDEBUG
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#else
#define gpuErrchk(ans) { }
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printCudaVersion() {
    std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}

__global__ void vectorAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n) C[i] = A[i] + B[i];
}

void vecAdd(float *h_A, float *h_B, float *h_C, int numElements) {
    cudaError_t err = cudaSuccess;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float *d_A, *d_B, *d_C;
    err = cudaMalloc((void **)&d_A, size); gpuErrchk(err);
    err = cudaMalloc((void **)&d_B, size); gpuErrchk(err);
    err = cudaMalloc((void **)&d_C, size); gpuErrchk(err);

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); gpuErrchk(err);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); gpuErrchk(err);

    int threadsPerBlock = 256;
    int blocksPerGrid = ceil(numElements / (float)threadsPerBlock);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements); gpuErrchk(cudaGetLastError());

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); gpuErrchk(err);

    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    err = cudaFree(d_A); gpuErrchk(err);
    err = cudaFree(d_B); gpuErrchk(err);
    err = cudaFree(d_C); gpuErrchk(err);
    err = cudaDeviceReset(); gpuErrchk(err);

    printf("Done\n");
}


