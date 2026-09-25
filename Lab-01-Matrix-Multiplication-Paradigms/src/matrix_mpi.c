/**
 * @file matrix_mpi.c
 * @brief Distributed-Memory Parallel Matrix Multiplication using MPI
 * @details Implements block-row decomposition of Matrix A with broadcast of Matrix B
 *          across distributed cluster nodes using MPI_Scatter and MPI_Gather.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

#define DEFAULT_N 4000

/**
 * @brief Multiplies a horizontal slice of A by the full Matrix B to produce local C slice.
 */
void compute_local_block(const double *local_A, const double *B, double *local_C, 
                         int rows_per_proc, int n) {
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }
}

/**
 * @brief Validates result matrix on root process.
 */
bool verify_result(const double *C, int n, double expected) {
    if (C[0] != expected) return false;
    if (C[(n - 1) * n + (n - 1)] != expected) return false;
    return true;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = DEFAULT_N;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (argc > 1) {
        int arg_n = atoi(argv[1]);
        if (arg_n > 0) n = arg_n;
    }

    if (n % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Matrix dimension N (%d) must be divisible by process count (%d).\n", n, size);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rows_per_proc = n / size;

    double *A = NULL;
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = NULL;

    double *local_A = (double *)malloc(rows_per_proc * n * sizeof(double));
    double *local_C = (double *)malloc(rows_per_proc * n * sizeof(double));

    if (!B || !local_A || !local_C) {
        fprintf(stderr, "[Rank %d] Error: Local memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        A = (double *)malloc(n * n * sizeof(double));
        C = (double *)malloc(n * n * sizeof(double));

        if (!A || !C) {
            fprintf(stderr, "Error: Root allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        printf("=================================================================\n");
        printf("   MPI Distributed-Memory Matrix Multiplication Benchmark        \n");
        printf("=================================================================\n");
        printf("Matrix Dimension (N x N) : %d x %d\n", n, n);
        printf("Total MPI Processes      : %d ranks\n", size);
        printf("Rows per Process         : %d rows\n", rows_per_proc);
        printf("Total Floating Ops       : %.2e FLOPs\n", 2.0 * n * n * n);
        printf("Initializing matrices on root rank...\n");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = 1.0;
                B[i * n + j] = 1.0;
                C[i * n + j] = 0.0;
            }
        }
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Distribute row blocks of A across all ranks
    MPI_Scatter(A, rows_per_proc * n, MPI_DOUBLE,
                local_A, rows_per_proc * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Broadcast full matrix B to all ranks
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local block
    compute_local_block(local_A, B, local_C, rows_per_proc, n);

    // Gather partial result blocks into root matrix C
    MPI_Gather(local_C, rows_per_proc * n, MPI_DOUBLE,
               C, rows_per_proc * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double elapsed_sec = end_time - start_time;
        double gflops = (2.0 * n * n * n) / (elapsed_sec * 1e9);
        bool verified = verify_result(C, n, (double)n);

        printf("-----------------------------------------------------------------\n");
        printf("Status                   : Completed\n");
        printf("MPI Cluster Processes    : %d\n", size);
        printf("Wall-Clock Time          : %.6f seconds\n", elapsed_sec);
        printf("Throughput               : %.2f GFLOPS\n", gflops);
        printf("Verification (C[0][0])   : %.2f (Expected: %.2f) -> %s\n", 
               C[0], (double)n, verified ? "PASSED" : "FAILED");
        printf("=================================================================\n");

        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
