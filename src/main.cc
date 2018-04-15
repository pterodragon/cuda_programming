#include <iostream>
#include <cstring>

#ifdef USE_CUDA
#include "gpu.hpp"
#endif

template<typename T>
void fill_array(T *arr, int n, T v) {
    for (int i = 0; i < n; ++i) {
        arr[i] = v;
    }
}


template<typename T>
void print_array(T *arr, int n) {
    std::cout << '[';
    for (int i = 0; i < n - 1; ++i) {
        std::cout << arr[i] << ", ";
    }
    if (n > 0) {
        std::cout << arr[n - 1];
    }
    std::cout << "]" << std::endl;
}

void my_add() {
    const int n = 25600;
    float a[n], b[n], c[n];
    fill_array(a, n, 1.0f);
    fill_array(b, n, 2.0f);
    vecAdd(a, b, c, n);
    print_array(a, 3);
    print_array(b, 3);
    print_array(c, 3);
}

int main() {
    std::cout << "Hello, world!" << std::endl;

#ifdef USE_CUDA
    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();
    my_add();
#else
    std::cout << "CUDA: Off" << std::endl;
#endif

    return 0;
}
