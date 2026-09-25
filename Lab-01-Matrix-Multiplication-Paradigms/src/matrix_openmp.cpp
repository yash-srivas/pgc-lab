/**
 * @file matrix_openmp.cpp
 * @brief Shared-Memory Parallel Matrix Multiplication in C++ with OpenMP
 * @details Implements multi-threaded matrix multiplication using modern C++ abstractions
 *          and OpenMP compiler pragmas for work-sharing.
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

constexpr size_t DEFAULT_N = 4000;
constexpr int DEFAULT_THREADS = 8;

class MatrixMultiplicationOpenMP {
private:
    size_t n_;
    int num_threads_;
    std::vector<double> A_;
    std::vector<double> B_;
    std::vector<double> C_;

public:
    MatrixMultiplicationOpenMP(size_t n, int num_threads)
        : n_(n), num_threads_(num_threads), A_(n * n, 1.0), B_(n * n, 1.0), C_(n * n, 0.0) {
        omp_set_num_threads(num_threads_);
    }

    /**
     * @brief Computes dense matrix product C = A x B in parallel across threads.
     */
    void compute() {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            for (size_t j = 0; j < n_; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n_; ++k) {
                    sum += A_[i * n_ + k] * B_[k * n_ + j];
                }
                C_[i * n_ + j] = sum;
            }
        }
    }

    bool verify(double expected) const {
        return (C_[0] == expected && C_[(n_ - 1) * n_ + (n_ - 1)] == expected);
    }

    double get_first_element() const { return C_[0]; }
    int get_thread_count() const { return num_threads_; }
    size_t get_dimension() const { return n_; }
};

int main(int argc, char* argv[]) {
    size_t n = DEFAULT_N;
    int num_threads = DEFAULT_THREADS;

    if (argc > 1) {
        long arg_n = std::atol(argv[1]);
        if (arg_n > 0) n = static_cast<size_t>(arg_n);
    }
    if (argc > 2) {
        int arg_t = std::atoi(argv[2]);
        if (arg_t > 0) num_threads = arg_t;
    }

    std::cout << "=================================================================\n";
    std::cout << "   OpenMP C++ Shared-Memory Matrix Multiplication Benchmark      \n";
    std::cout << "=================================================================\n";
    std::cout << "Matrix Dimension (N x N) : " << n << " x " << n << "\n";
    std::cout << "Thread Count Configured  : " << num_threads << " threads\n";

    MatrixMultiplicationOpenMP matmul(n, num_threads);

    std::cout << "Executing OpenMP parallel kernel...\n";
    double start_time = omp_get_wtime();
    matmul.compute();
    double end_time = omp_get_wtime();

    double elapsed_sec = end_time - start_time;
    double gflops = (2.0 * n * n * n) / (elapsed_sec * 1e9);
    bool verified = matmul.verify(static_cast<double>(n));

    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Status                   : Completed\n";
    std::cout << "Threads Utilized         : " << num_threads << "\n";
    std::cout << "Wall-Clock Time          : " << std::fixed << std::setprecision(6) 
              << elapsed_sec << " seconds\n";
    std::cout << "Throughput               : " << std::fixed << std::setprecision(2) 
              << gflops << " GFLOPS\n";
    std::cout << "Verification (C[0][0])   : " << std::fixed << std::setprecision(2) 
              << matmul.get_first_element() << " (Expected: " << static_cast<double>(n) << ") -> "
              << (verified ? "PASSED" : "FAILED") << "\n";
    std::cout << "=================================================================\n";

    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
