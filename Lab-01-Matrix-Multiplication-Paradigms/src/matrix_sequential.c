/**
 * @file matrix_sequential.c
 * @brief High-Performance Sequential Matrix Multiplication (Baseline)
 * @details Implements dense matrix multiplication C = A x B using single-threaded CPU execution.
 *          Serves as the computational baseline for speedup (S = T1 / Tp) and parallel efficiency evaluations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define DEFAULT_N 4000

/**
 * @brief Allocates a contiguous 1D array representing an N x N matrix.
 * @param n Dimension of the matrix.
 * @return Pointer to allocated buffer, or NULL on failure.
 */
double* allocate_matrix(size_t n) {
    double *mat = (double *)malloc(n * n * sizeof(double));
    return mat;
}

/**
 * @brief Initializes input matrices A and B with unit values and clears matrix C.
 * @param A Pointer to Matrix A.
 * @param B Pointer to Matrix B.
 * @param C Pointer to Matrix C.
 * @param n Dimension of the matrices.
 */
void initialize_matrices(double *A, double *B, double *C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A[i * n + j] = 1.0;
            B[i * n + j] = 1.0;
            C[i * n + j] = 0.0;
        }
    }
}

/**
 * @brief Performs sequential dense matrix multiplication: C = A x B.
 * @param A Pointer to Matrix A (Row-major order).
 * @param B Pointer to Matrix B (Row-major order).
 * @param C Pointer to Matrix C (Row-major order, output).
 * @param n Dimension of the matrices.
 */
void multiply_matrices_sequential(const double *A, const double *B, double *C, size_t n) {
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
 * @brief Verifies numerical consistency of the result matrix.
 * @param C Pointer to Result Matrix C.
 * @param n Dimension of the matrix.
 * @param expected Expected value of each cell.
 * @return True if verification passes, false otherwise.
 */
bool verify_result(const double *C, size_t n, double expected) {
    // Check first element and diagonal elements
    if (C[0] != expected) return false;
    if (C[(n - 1) * n + (n - 1)] != expected) return false;
    return true;
}

/**
 * @brief Releases dynamically allocated memory buffers.
 */
void cleanup_matrices(double *A, double *B, double *C) {
    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
}

int main(int argc, char *argv[]) {
    size_t n = DEFAULT_N;
    if (argc > 1) {
        long arg_n = atol(argv[1]);
        if (arg_n > 0) {
            n = (size_t)arg_n;
        }
    }

    printf("=================================================================\n");
    printf("   Sequential Dense Matrix Multiplication Benchmark (Baseline)   \n");
    printf("=================================================================\n");
    printf("Matrix Dimension (N x N) : %zu x %zu\n", n, n);
    printf("Total Floating Ops       : %.2e FLOPs (2 * N^3)\n", 2.0 * n * n * n);

    double *A = allocate_matrix(n);
    double *B = allocate_matrix(n);
    double *C = allocate_matrix(n);

    if (!A || !B || !C) {
        fprintf(stderr, "Error: Memory allocation failed for %zu x %zu double matrices.\n", n, n);
        cleanup_matrices(A, B, C);
        return EXIT_FAILURE;
    }

    printf("Initializing matrices (A[i][j]=1.0, B[i][j]=1.0)...\n");
    initialize_matrices(A, B, C, n);

    printf("Executing sequential kernel...\n");
    clock_t start_clk = clock();
    multiply_matrices_sequential(A, B, C, n);
    clock_t end_clk = clock();

    double elapsed_sec = (double)(end_clk - start_clk) / CLOCKS_PER_SEC;
    double gflops = (2.0 * n * n * n) / (elapsed_sec * 1e9);
    bool verified = verify_result(C, n, (double)n);

    printf("-----------------------------------------------------------------\n");
    printf("Status                   : Completed\n");
    printf("Wall-Clock Time          : %.6f seconds\n", elapsed_sec);
    printf("Throughput               : %.2f GFLOPS\n", gflops);
    printf("Verification (C[0][0])   : %.2f (Expected: %.2f) -> %s\n", 
           C[0], (double)n, verified ? "PASSED" : "FAILED");
    printf("=================================================================\n");

    cleanup_matrices(A, B, C);
    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
