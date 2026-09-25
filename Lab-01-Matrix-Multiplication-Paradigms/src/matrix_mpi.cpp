/**
 * @file matrix_mpi.cpp
 * @brief Distributed-Memory Parallel Matrix Multiplication in C++ with MPI
 * @details Implements block-row decomposition using C++ standard library buffers
 *          and MPI collective communication routines.
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>

constexpr int DEFAULT_N = 4000;

void compute_local_block(const double* local_A, const double* B, double* local_C,
                         int rows_per_proc, int n) {
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += local_A[i * n + k] * B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = DEFAULT_N;
    if (argc > 1) {
        int arg_n = std::atoi(argv[1]);
        if (arg_n > 0) n = arg_n;
    }

    if (n % size != 0) {
        if (rank == 0) {
            std::cerr << "Error: Matrix dimension (" << n << ") must be divisible by process count (" << size << ").\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rows_per_proc = n / size;

    std::vector<double> B(n * n, 1.0);
    std::vector<double> local_A(rows_per_proc * n);
    std::vector<double> local_C(rows_per_proc * n, 0.0);

    std::vector<double> A;
    std::vector<double> C;

    if (rank == 0) {
        A.assign(n * n, 1.0);
        C.assign(n * n, 0.0);

        std::cout << "=================================================================\n";
        std::cout << "   MPI C++ Distributed-Memory Matrix Multiplication Benchmark    \n";
        std::cout << "=================================================================\n";
        std::cout << "Matrix Dimension (N x N) : " << n << " x " << n << "\n";
        std::cout << "Total MPI Processes      : " << size << " ranks\n";
        std::cout << "Rows per Process         : " << rows_per_proc << " rows\n";
        std::cout << "Initializing buffers on root rank...\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    MPI_Scatter(A.data(), rows_per_proc * n, MPI_DOUBLE,
                local_A.data(), rows_per_proc * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    compute_local_block(local_A.data(), B.data(), local_C.data(), rows_per_proc, n);

    MPI_Gather(local_C.data(), rows_per_proc * n, MPI_DOUBLE,
               C.data(), rows_per_proc * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double elapsed_sec = end_time - start_time;
        double gflops = (2.0 * n * n * n) / (elapsed_sec * 1e9);
        bool verified = (C[0] == static_cast<double>(n) && C[(n - 1) * n + (n - 1)] == static_cast<double>(n));

        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "Status                   : Completed\n";
        std::cout << "MPI Processes            : " << size << "\n";
        std::cout << "Wall-Clock Time          : " << std::fixed << std::setprecision(6) 
                  << elapsed_sec << " seconds\n";
        std::cout << "Throughput               : " << std::fixed << std::setprecision(2) 
                  << gflops << " GFLOPS\n";
        std::cout << "Verification (C[0][0])   : " << std::fixed << std::setprecision(2) 
                  << C[0] << " (Expected: " << static_cast<double>(n) << ") -> "
                  << (verified ? "PASSED" : "FAILED") << "\n";
        std::cout << "=================================================================\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
