#include <iostream>
#include <cstring>

#ifdef USE_CUDA
#include "gpu.hpp"
#include "matrix.hpp"
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

template<typename T>
void print_mat(T *arr, int m, int n) {
  for (int j = 0; j < m; ++j) {
    std::cout << '[';
    for (int i = 0; i < n - 1; ++i) {
        std::cout << arr[i] << ", ";
    }
    if (n > 0) {
        std::cout << arr[n - 1];
    }
    std::cout << "]" << std::endl;
  }
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

void test_matmul() {
    /*
     * [m][k] @ [k][n] = [m][n]
     */
    const int m_d = 4;
    const int n_d = 4;
    const int k_d = 5;
    const int A = m_d * k_d;
    const int B = k_d * n_d;
    const int C = m_d * n_d;
    int a[A], b[B], c[C];
    fill_array(a, A, 3);
    fill_array(b, B, 2);
    matrix_mult(a, b, c, m_d, n_d, k_d);
    print_mat(a, m_d, k_d);
    print_mat(b, k_d, n_d);
    print_mat(c, m_d, n_d);
}

void test_matmul_tiled() {
    /*
     * [m][k] @ [k][n] = [m][n]
     */
    const int m_d = 3;
    const int n_d = 4;
    const int k_d = 5;
    const int A = m_d * k_d;
    const int B = k_d * n_d;
    const int C = m_d * n_d;
    int a[A], b[B], c[C];
    fill_array(a, A, 4);
    fill_array(b, B, 5);
    matrix_mult_tiled(a, b, c, m_d, n_d, k_d);
    print_mat(a, m_d, k_d);
    print_mat(b, k_d, n_d);
    print_mat(c, m_d, n_d);
}
void ps() {
  printf("---\n");
}

int main() {
    std::cout << "Hello, world!" << std::endl;

#ifdef USE_CUDA
    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();
    my_add();
    ps();
    test_matmul();
    ps();
    test_matmul_tiled();
    ps();
#else
    std::cout << "CUDA: Off" << std::endl;
#endif

    return 0;
}
