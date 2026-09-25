/**
 * @file matrix_sequential.cpp
 * @brief High-Performance Sequential Matrix Multiplication in C++ (Baseline)
 * @details Implements dense matrix multiplication C = A x B using modern C++ abstractions.
 *          Serves as the single-threaded benchmark for parallel scaling evaluations.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>

constexpr size_t DEFAULT_N = 4000;

class MatrixMultiplicationSequential {
private:
    size_t n_;
    std::vector<double> A_;
    std::vector<double> B_;
    std::vector<double> C_;

public:
    explicit MatrixMultiplicationSequential(size_t n)
        : n_(n), A_(n * n, 1.0), B_(n * n, 1.0), C_(n * n, 0.0) {}

    /**
     * @brief Computes dense matrix product C = A x B sequentially.
     */
    void compute() {
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

    /**
     * @brief Validates result matrix elements against expected value.
     */
    bool verify(double expected) const {
        return (C_[0] == expected && C_[(n_ - 1) * n_ + (n_ - 1)] == expected);
    }

    double get_first_element() const { return C_[0]; }
    size_t get_dimension() const { return n_; }
};

int main(int argc, char* argv[]) {
    size_t n = DEFAULT_N;
    if (argc > 1) {
        long arg_n = std::atol(argv[1]);
        if (arg_n > 0) {
            n = static_cast<size_t>(arg_n);
        }
    }

    std::cout << "=================================================================\n";
    std::cout << "  Sequential C++ Dense Matrix Multiplication Benchmark (Baseline) \n";
    std::cout << "=================================================================\n";
    std::cout << "Matrix Dimension (N x N) : " << n << " x " << n << "\n";
    std::cout << "Total Floating Ops       : " << std::scientific << std::setprecision(2) 
              << (2.0 * n * n * n) << " FLOPs\n";
    std::cout << "Initializing matrix buffers...\n";

    MatrixMultiplicationSequential matmul(n);

    std::cout << "Executing sequential matrix multiplication...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    matmul.compute();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    double elapsed_sec = elapsed.count();
    double gflops = (2.0 * n * n * n) / (elapsed_sec * 1e9);
    bool verified = matmul.verify(static_cast<double>(n));

    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Status                   : Completed\n";
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
