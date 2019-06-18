#ifndef GPU_HPP
#define GPU_HPP 

void printCudaVersion();
void print_device_properties();

void vecAdd(float* A, float* B, float* C, int n);

#endif /* GPU_HPP */
