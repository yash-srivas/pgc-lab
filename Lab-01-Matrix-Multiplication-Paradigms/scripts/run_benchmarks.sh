#!/usr/bin/env bash
# ==============================================================================
# Parallel & GPU Computing (PGC) Coursework Automation Suite
# Laboratory 01: Dense Matrix Multiplication Across Four Parallel Paradigms
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
BIN_DIR="${ROOT_DIR}/bin"
IMAGES_DIR="${ROOT_DIR}/images"

mkdir -p "${BIN_DIR}"
mkdir -p "${IMAGES_DIR}"

COLOR_RESET="\033[0m"
COLOR_BOLD="\033[1m"
COLOR_GREEN="\033[32m"
COLOR_BLUE="\033[34m"
COLOR_YELLOW="\033[33m"
COLOR_RED="\033[31m"

print_header() {
    echo -e "${COLOR_BLUE}${COLOR_BOLD}===================================================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}${COLOR_BOLD}   Parallel & GPU Computing (PGC) - Experiment 01 Automation Suite   ${COLOR_RESET}"
    echo -e "${COLOR_BLUE}${COLOR_BOLD}===================================================================${COLOR_RESET}"
}

check_toolchains() {
    echo -e "\n${COLOR_BOLD}[*] Checking Toolchain Availability:${COLOR_RESET}"
    
    if command -v gcc >/dev/null 2>&1; then
        echo -e "  [+] GCC C Compiler          : $(gcc --version | head -n1)"
    else
        echo -e "  [!] GCC C Compiler          : ${COLOR_YELLOW}Not Found${COLOR_RESET}"
    fi

    if command -v g++ >/dev/null 2>&1; then
        echo -e "  [+] G++ C++ Compiler        : $(g++ --version | head -n1)"
    else
        echo -e "  [!] G++ C++ Compiler        : ${COLOR_YELLOW}Not Found${COLOR_RESET}"
    fi

    if command -v mpicc >/dev/null 2>&1; then
        echo -e "  [+] Open MPI C Compiler     : $(mpicc --version | head -n1)"
    else
        echo -e "  [-] Open MPI C Compiler     : ${COLOR_YELLOW}Skipped (Not Installed)${COLOR_RESET}"
    fi

    if command -v nvcc >/dev/null 2>&1; then
        echo -e "  [+] NVIDIA NVCC Compiler    : $(nvcc --version | grep release | head -n1)"
    else
        echo -e "  [-] NVIDIA NVCC Compiler    : ${COLOR_YELLOW}Skipped (No CUDA Device / Toolkit)${COLOR_RESET}"
    fi

    if command -v python3 >/dev/null 2>&1; then
        echo -e "  [+] Python 3 Runtime        : $(python3 --version)"
    elif command -v python >/dev/null 2>&1; then
        echo -e "  [+] Python Runtime          : $(python --version)"
    fi
}

compile_all() {
    echo -e "\n${COLOR_BOLD}[*] Compiling Parallel Implementations (Optimization -O2):${COLOR_RESET}"
    
    # 1. Sequential C & C++
    if command -v gcc >/dev/null 2>&1; then
        echo -e "  [>] Compiling Sequential C (matrix_sequential.c)..."
        gcc -O2 "${SRC_DIR}/matrix_sequential.c" -o "${BIN_DIR}/matrix_sequential_c"
    fi
    if command -v g++ >/dev/null 2>&1; then
        echo -e "  [>] Compiling Sequential C++ (matrix_sequential.cpp)..."
        g++ -O2 "${SRC_DIR}/matrix_sequential.cpp" -o "${BIN_DIR}/matrix_sequential_cpp"
    fi

    # 2. OpenMP C & C++
    if command -v gcc >/dev/null 2>&1; then
        echo -e "  [>] Compiling OpenMP C (matrix_openmp.c)..."
        gcc -O2 -fopenmp "${SRC_DIR}/matrix_openmp.c" -o "${BIN_DIR}/matrix_openmp_c"
    fi
    if command -v g++ >/dev/null 2>&1; then
        echo -e "  [>] Compiling OpenMP C++ (matrix_openmp.cpp)..."
        g++ -O2 -fopenmp "${SRC_DIR}/matrix_openmp.cpp" -o "${BIN_DIR}/matrix_openmp_cpp"
    fi

    # 3. MPI C & C++
    if command -v mpicc >/dev/null 2>&1; then
        echo -e "  [>] Compiling Open MPI C (matrix_mpi.c)..."
        mpicc -O2 "${SRC_DIR}/matrix_mpi.c" -o "${BIN_DIR}/matrix_mpi_c"
    fi
    if command -v mpicxx >/dev/null 2>&1; then
        echo -e "  [>] Compiling Open MPI C++ (matrix_mpi.cpp)..."
        mpicxx -O2 "${SRC_DIR}/matrix_mpi.cpp" -o "${BIN_DIR}/matrix_mpi_cpp"
    fi

    # 4. CUDA C & C++
    if command -v nvcc >/dev/null 2>&1; then
        echo -e "  [>] Compiling CUDA Kernel (matrix_cuda.cu)..."
        nvcc -O2 "${SRC_DIR}/matrix_cuda.cu" -o "${BIN_DIR}/matrix_cuda_cu"
        echo -e "  [>] Compiling CUDA C++ (matrix_cuda.cpp)..."
        nvcc -O2 "${SRC_DIR}/matrix_cuda.cpp" -o "${BIN_DIR}/matrix_cuda_cpp"
    fi

    echo -e "${COLOR_GREEN}[+] Compilation Complete! Binaries placed in: ${BIN_DIR}${COLOR_RESET}"
}

run_quick_tests() {
    local test_size="${1:-500}"
    echo -e "\n${COLOR_BOLD}[*] Executing Sanity Verification Test (Matrix Size N=${test_size}):${COLOR_RESET}"

    if [ -f "${BIN_DIR}/matrix_sequential_c" ]; then
        echo -e "\n--- Running Sequential C ---"
        "${BIN_DIR}/matrix_sequential_c" "${test_size}"
    fi

    if [ -f "${BIN_DIR}/matrix_openmp_c" ]; then
        echo -e "\n--- Running OpenMP C (4 Threads) ---"
        "${BIN_DIR}/matrix_openmp_c" "${test_size}" 4
    fi

    if [ -f "${BIN_DIR}/matrix_mpi_c" ] && command -v mpirun >/dev/null 2>&1; then
        echo -e "\n--- Running MPI C (2 Processes) ---"
        mpirun -np 2 "${BIN_DIR}/matrix_mpi_c" "${test_size}"
    fi

    if [ -f "${BIN_DIR}/matrix_cuda_cu" ]; then
        echo -e "\n--- Running CUDA GPU ---"
        "${BIN_DIR}/matrix_cuda_cu" "${test_size}"
    fi
}

run_analysis_and_plots() {
    echo -e "\n${COLOR_BOLD}[*] Running Metric Parsing & Visualizations:${COLOR_RESET}"
    local py_cmd="python3"
    if ! command -v python3 >/dev/null 2>&1; then
        py_cmd="python"
    fi

    ${py_cmd} "${SCRIPT_DIR}/parse_results.py"
    ${py_cmd} "${SCRIPT_DIR}/generate_plots.py" --output-dir "${IMAGES_DIR}"
}

clean_build() {
    echo -e "\n${COLOR_BOLD}[*] Cleaning Build Binaries:${COLOR_RESET}"
    rm -rf "${BIN_DIR}"
    echo -e "${COLOR_GREEN}[+] Clean complete.${COLOR_RESET}"
}

show_help() {
    echo "Usage: ./run_benchmarks.sh [OPTION]"
    echo "Options:"
    echo "  --all        Build binaries, run quick verification, and regenerate plots (Default)"
    echo "  --build      Compile all source codes"
    echo "  --test [N]   Execute verification run with matrix size N (default: 500)"
    echo "  --plots      Regenerate publication-grade performance charts"
    echo "  --clean      Remove compiled binaries"
    echo "  --help       Show this help message"
}

# Main Execution Routing
print_header
ACTION="${1:---all}"

case "${ACTION}" in
    --all)
        check_toolchains
        compile_all
        run_quick_tests 500
        run_analysis_and_plots
        ;;
    --build)
        check_toolchains
        compile_all
        ;;
    --test)
        run_quick_tests "${2:-500}"
        ;;
    --plots)
        run_analysis_and_plots
        ;;
    --clean)
        clean_build
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo -e "${COLOR_RED}Unknown option: ${ACTION}${COLOR_RESET}"
        show_help
        exit 1
        ;;
esac

echo -e "\n${COLOR_GREEN}${COLOR_BOLD}[*] Done!${COLOR_RESET}\n"
