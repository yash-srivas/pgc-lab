/**
 * @file matrix_cuda.cpp
 * @brief Massively Parallel Matrix Multiplication in C++ with NVIDIA CUDA
 * @details Implements dense matrix multiplication on the GPU using modern C++ wrappers.
 */

#include <iostream>
#include <vector>
#include <iomanip>
#if defined(__has_include)
  #if __has_include(<cuda_runtime.h>)
    #include <cuda_runtime.h>
  #else
    #include "cuda_runtime_compat.h"
  #endif
#elif defined(__CUDACC__)
  #include <cuda_runtime.h>
#else
  #include "cuda_runtime_compat.h"
#endif

constexpr int DEFAULT_N = 4000;
constexpr int BLOCK_SIZE = 16;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void matrix_multiply_kernel(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    int n = DEFAULT_N;
    if (argc > 1) {
        int arg_n = std::atoi(argv[1]);
        if (arg_n > 0) n = arg_n;
    }

    size_t total_elements = static_cast<size_t>(n) * n;
    size_t bytes = total_elements * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "=================================================================\n";
    std::cout << "   CUDA C++ Massively Parallel GPU Benchmark                     \n";
    std::cout << "=================================================================\n";
    std::cout << "GPU Device               : " << prop.name << "\n";
    std::cout << "Matrix Dimension (N x N) : " << n << " x " << n << "\n";
    std::cout << "Memory Allocation        : " << std::fixed << std::setprecision(2)
              << (bytes / (1024.0 * 1024.0)) << " MB per matrix\n";

    std::vector<float> h_A(total_elements, 1.0f);
    std::vector<float> h_B(total_elements, 1.0f);
    std::vector<float> h_C(total_elements, 0.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    cudaEvent_t total_start, total_stop, kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&total_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    CUDA_CHECK(cudaEventRecord(total_start));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventRecord(kernel_start));
#if defined(__CUDACC__) || defined(__CUDA__)
    matrix_multiply_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
#else
    (void)grid;
    (void)block;
    matrix_multiply_kernel(d_A, d_B, d_C, n);
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(total_stop));
    CUDA_CHECK(cudaEventSynchronize(total_stop));

    float kernel_ms = 0.0f;
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));

    double kernel_sec = kernel_ms / 1000.0;
    double total_sec = total_ms / 1000.0;

    double kernel_gflops = (2.0 * n * n * n) / (kernel_sec * 1e9);
    double total_gflops = (2.0 * n * n * n) / (total_sec * 1e9);

    bool verified = (h_C[0] == static_cast<float>(n) && h_C[total_elements - 1] == static_cast<float>(n));

    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Status                   : Completed\n";
    std::cout << "Kernel Execution Time    : " << std::fixed << std::setprecision(6) 
              << kernel_sec << " seconds\n";
    std::cout << "Kernel Throughput        : " << std::fixed << std::setprecision(2) 
              << kernel_gflops << " GFLOPS\n";
    std::cout << "Total Phase Time         : " << std::fixed << std::setprecision(6) 
              << total_sec << " seconds\n";
    std::cout << "Total Phase Throughput   : " << std::fixed << std::setprecision(2) 
              << total_gflops << " GFLOPS\n";
    std::cout << "Verification (C[0][0])   : " << std::fixed << std::setprecision(2) 
              << h_C[0] << " (Expected: " << static_cast<float>(n) << ") -> "
              << (verified ? "PASSED" : "FAILED") << "\n";
    std::cout << "=================================================================\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(total_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
