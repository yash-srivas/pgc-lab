/**
 * @file matrix_openmp.c
 * @brief Shared-Memory Parallel Matrix Multiplication using OpenMP
 * @details Implements multi-threaded dense matrix multiplication C = A x B via OpenMP work-sharing.
 *          Evaluates thread scaling, shared-memory bus contention, and parallel efficiency.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#define DEFAULT_N 4000
#define DEFAULT_THREADS 8

/**
 * @brief Allocates an N x N matrix in contiguous memory.
 */
double* allocate_matrix(size_t n) {
    return (double *)malloc(n * n * sizeof(double));
}

/**
 * @brief Initializes matrix values in parallel.
 */
void initialize_matrices(double *A, double *B, double *C, size_t n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A[i * n + j] = 1.0;
            B[i * n + j] = 1.0;
            C[i * n + j] = 0.0;
        }
    }
}

/**
 * @brief Multi-threaded matrix multiplication with OpenMP loop scheduling.
 * @param A Matrix A (Input)
 * @param B Matrix B (Input)
 * @param C Matrix C (Output)
 * @param n Matrix dimension
 */
void multiply_matrices_openmp(const double *A, const double *B, double *C, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * @brief Verifies correctness of computed matrix.
 */
bool verify_result(const double *C, size_t n, double expected) {
    if (C[0] != expected) return false;
    if (C[(n - 1) * n + (n - 1)] != expected) return false;
    return true;
}

/**
 * @brief Frees allocated matrix buffers.
 */
void cleanup_matrices(double *A, double *B, double *C) {
    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
}

int main(int argc, char *argv[]) {
    size_t n = DEFAULT_N;
    int num_threads = DEFAULT_THREADS;

    if (argc > 1) {
        long arg_n = atol(argv[1]);
        if (arg_n > 0) n = (size_t)arg_n;
    }
    if (argc > 2) {
        int arg_t = atoi(argv[2]);
        if (arg_t > 0) num_threads = arg_t;
    }

    omp_set_num_threads(num_threads);

    printf("=================================================================\n");
    printf("   OpenMP Shared-Memory Parallel Matrix Multiplication Benchmark  \n");
    printf("=================================================================\n");
    printf("Matrix Dimension (N x N) : %zu x %zu\n", n, n);
    printf("Configured Thread Count  : %d threads\n", num_threads);
    printf("Max OpenMP Threads Avail : %d\n", omp_get_max_threads());
    printf("Total Floating Ops       : %.2e FLOPs (2 * N^3)\n", 2.0 * n * n * n);

    double *A = allocate_matrix(n);
    double *B = allocate_matrix(n);
    double *C = allocate_matrix(n);

    if (!A || !B || !C) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        cleanup_matrices(A, B, C);
        return EXIT_FAILURE;
    }

    printf("Initializing matrices in parallel...\n");
    initialize_matrices(A, B, C, n);

    printf("Executing OpenMP parallel kernel across %d threads...\n", num_threads);
    double start_time = omp_get_wtime();
    multiply_matrices_openmp(A, B, C, n);
    double end_time = omp_get_wtime();

    double elapsed_sec = end_time - start_time;
    double gflops = (2.0 * n * n * n) / (elapsed_sec * 1e9);
    bool verified = verify_result(C, n, (double)n);

    printf("-----------------------------------------------------------------\n");
    printf("Status                   : Completed\n");
    printf("Active Threads Used      : %d\n", num_threads);
    printf("Wall-Clock Time          : %.6f seconds\n", elapsed_sec);
    printf("Throughput               : %.2f GFLOPS\n", gflops);
    printf("Verification (C[0][0])   : %.2f (Expected: %.2f) -> %s\n", 
           C[0], (double)n, verified ? "PASSED" : "FAILED");
    printf("=================================================================\n");

    cleanup_matrices(A, B, C);
    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
