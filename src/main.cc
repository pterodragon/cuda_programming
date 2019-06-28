#include <iostream>
#include <cstring>

#ifdef USE_CUDA
#include "gpu.hpp"
#include "matrix.hpp"
#include "convolution.hpp"
#include "inclusive_scan.hpp"
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
        std::cout << arr[j * n + i] << ", ";
    }
    if (n > 0) {
        std::cout << arr[j * n + n - 1];
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

void test_convolution_2d_tiled() {
    const int height = 5;
    const int width = 7;
    const int pitch = 8;
    const int mask_width = 3;
    const int N_size = height * pitch;
    const int M_size = mask_width * mask_width;
    float N[N_size], M[M_size], P[N_size];
    fill_array(N, N_size, 1.0f);
    fill_array(M, M_size, 2.0f);
    M[4] = 10.0f;
    for (int q = 0; q < height; ++q) {
      for (int w = width; w < pitch; ++w) {
        N[pitch * q + w] = 0.0f;
      }
    }
    for (int q = 0; q < height; ++q) {
      for (int w = 0; w < pitch; ++w) {
        P[pitch * q + w] = -1.0f;
      }
    }
    print_mat(N, height, pitch);
    print_mat(M, mask_width, mask_width);

    std::cout << "height = " << height << '\n';
    std::cout << "width = " << width << '\n';
    std::cout << "pitch = " << pitch << '\n';
    std::cout << "mask_width = " << mask_width << '\n';
    convolution_2D_tiled(P, N, height, width, pitch, mask_width, M);

    print_mat(P, height, pitch);
}

void test_inclusive_scan() {
  const int N = 1 << 16;
  float nums[N], out[N];
  for (int q = 0; q < N; ++q) nums[q] = (int)(q / 2048 + 1);
  print_array(nums, N);
  inclusive_scan(nums, out, N);
  print_array(out, N);
  // answer should be (32 * 33) / 2 * 2048 = 1081344
}

void ps() {
  printf("---\n");
}

int main() {
    std::cout << "Hello, world!" << std::endl;

#ifdef USE_CUDA
    std::cout << "CUDA: On" << std::endl;
    ps();
    printCudaVersion();
    ps();
    print_device_properties();
    ps();
//    my_add();
//    ps();
//    test_matmul();
//    ps();
//    test_matmul_tiled();
//    ps();
//    test_convolution_2d_tiled();
//    ps();
    test_inclusive_scan();
    ps();
#else
    std::cout << "CUDA: Off" << std::endl;
#endif

    return 0;
}
