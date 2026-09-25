/**
 * @file matrix_cuda.cu
 * @brief Massively Parallel Matrix Multiplication using NVIDIA CUDA
 * @details Implements dense matrix multiplication on the GPU via SIMT architecture.
 *          Benchmarks pure kernel execution time and total end-to-end device transfer time.
 */

#include <stdio.h>
#include <stdlib.h>
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

#define DEFAULT_N 4000
#define BLOCK_SIZE 16

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/**
 * @brief CUDA 2D kernel for matrix multiplication: C = A x B.
 *        Each thread computes a single element (row, col) of Matrix C.
 */
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

/**
 * @brief Initializes host matrix data.
 */
void initialize_host_data(float *h_A, float *h_B, float *h_C, size_t total_elements) {
    for (size_t i = 0; i < total_elements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }
}

int main(int argc, char *argv[]) {
    int n = DEFAULT_N;
    if (argc > 1) {
        int arg_n = atoi(argv[1]);
        if (arg_n > 0) n = arg_n;
    }

    size_t total_elements = (size_t)n * n;
    size_t bytes = total_elements * sizeof(float);

    // Query device properties
    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    printf("=================================================================\n");
    printf("   CUDA Massively Parallel GPU Matrix Multiplication Benchmark  \n");
    printf("=================================================================\n");
    printf("GPU Device               : %s\n", prop.name);
    printf("Compute Capability       : %d.%d\n", prop.major, prop.minor);
    printf("Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Matrix Dimension (N x N) : %d x %d\n", n, n);
    printf("Total Matrix Elements    : %zu elements\n", total_elements);
    printf("Total Memory per Matrix  : %.2f MB\n", (double)bytes / (1024.0 * 1024.0));
    printf("Total Floating Ops       : %.2e FLOPs (2 * N^3)\n", 2.0 * n * n * n);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    initialize_host_data(h_A, h_B, h_C, total_elements);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    cudaEvent_t total_start, total_stop, kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&total_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    // 1. Total phase start
    CUDA_CHECK(cudaEventRecord(total_start));

    // Host to Device Transfers
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel Launch Geometry
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    printf("Execution Grid           : %d x %d blocks (%d threads/block)\n",
           grid.x, grid.y, block.x * block.y);
    printf("Total Logical Threads    : %lu threads\n",
           (unsigned long)grid.x * grid.y * block.x * block.y);

    // 2. Kernel execution timing
    CUDA_CHECK(cudaEventRecord(kernel_start));
#if defined(__CUDACC__)
    matrix_multiply_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
#else
    (void)grid;
    (void)block;
    matrix_multiply_kernel(d_A, d_B, d_C, n);
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    // Device to Host Transfer
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 3. Total phase stop
    CUDA_CHECK(cudaEventRecord(total_stop));
    CUDA_CHECK(cudaEventSynchronize(total_stop));

    float kernel_ms = 0.0f;
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));

    double kernel_sec = (double)kernel_ms / 1000.0;
    double total_sec = (double)total_ms / 1000.0;

    double kernel_gflops = (2.0 * n * n * n) / (kernel_sec * 1e9);
    double total_gflops = (2.0 * n * n * n) / (total_sec * 1e9);

    bool verified = (h_C[0] == (float)n && h_C[total_elements - 1] == (float)n);

    printf("-----------------------------------------------------------------\n");
    printf("Status                   : Completed\n");
    printf("Kernel Execution Time    : %.6f seconds\n", kernel_sec);
    printf("Kernel Throughput        : %.2f GFLOPS\n", kernel_gflops);
    printf("Total Phase Time         : %.6f seconds (inc. H2D/D2H transfers)\n", total_sec);
    printf("Total Phase Throughput   : %.2f GFLOPS\n", total_gflops);
    printf("Verification (C[0][0])   : %.2f (Expected: %.2f) -> %s\n",
           h_C[0], (float)n, verified ? "PASSED" : "FAILED");
    printf("=================================================================\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(total_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    free(h_A);
    free(h_B);
    free(h_C);

    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
