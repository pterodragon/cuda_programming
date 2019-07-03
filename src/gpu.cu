#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "util.hpp"
#include "gpu.hpp"

void printCudaVersion() {
  std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;

  int runtime_ver;
  cudaRuntimeGetVersion(&runtime_ver);
  std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

  int driver_ver;
  cudaDriverGetVersion(&driver_ver);
  std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}

void print_device_properties() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  int dev_count;
  cudaError_t err = cudaSuccess;
  err = cudaGetDeviceCount(&dev_count);
  gpuErrchk(err);

  for (int dev = 0; dev < dev_count; ++dev) {
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, dev);
    gpuErrchk(err);

    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        printf("No CUDA GPU has been detected\n");
        return;
      } else if (dev_count == 1) {
        printf("There is 1 device supporting CUDA\n");
      } else {
        printf("There are %d devices supporting CUDA\n", dev_count);
      }
    }
    printf("For device #%d\n", dev);
    printf("Device name:                %s\n", deviceProp.name);
    printf("Major revision number:      %d\n", deviceProp.major);
    printf("Minor revision Number:      %d\n", deviceProp.minor);
    printf("Total Global Memory:        %lu\n", deviceProp.totalGlobalMem);
    printf("Total shared mem per block: %lu\n", deviceProp.sharedMemPerBlock);
    printf("Total const mem size:       %lu\n", deviceProp.totalConstMem);
    printf("Warp size:                  %d\n", deviceProp.warpSize);
    printf("Maximum block dimensions:   %d x %d x %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions:    %d x %d x %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("Clock Rate:                 %d\n", deviceProp.clockRate);
    printf("Number of multiprocessors:   %d\n", deviceProp.multiProcessorCount);
  }

  return;
}
