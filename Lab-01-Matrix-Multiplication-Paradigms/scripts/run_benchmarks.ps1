# ==============================================================================
# Parallel & GPU Computing (PGC) Coursework Automation Suite (PowerShell)
# Laboratory 01: Dense Matrix Multiplication Across Four Parallel Paradigms
# ==============================================================================

param (
    [ValidateSet("all", "build", "test", "plots", "clean")]
    [string]$Task = "all",
    [int]$Size = 500
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$SrcDir = Join-Path $RootDir "src"
$BinDir = Join-Path $RootDir "bin"
$ImagesDir = Join-Path $RootDir "images"

if (!(Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }
if (!(Test-Path $ImagesDir)) { New-Item -ItemType Directory -Path $ImagesDir -Force | Out-Null }

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "   Parallel & GPU Computing (PGC) - Experiment 01 Automation Suite   " -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

function Build-Binaries {
    Write-Host "`n[*] Compiling Parallel Implementations (Optimization -O2):" -ForegroundColor Yellow

    if (Get-Command gcc -ErrorAction SilentlyContinue) {
        Write-Host "  [>] Compiling Sequential C (matrix_sequential.c)..."
        gcc -O2 (Join-Path $SrcDir "matrix_sequential.c") -o (Join-Path $BinDir "matrix_sequential_c.exe")
        Write-Host "  [>] Compiling OpenMP C (matrix_openmp.c)..."
        gcc -O2 -fopenmp (Join-Path $SrcDir "matrix_openmp.c") -o (Join-Path $BinDir "matrix_openmp_c.exe")
    }

    if (Get-Command g++ -ErrorAction SilentlyContinue) {
        Write-Host "  [>] Compiling Sequential C++ (matrix_sequential.cpp)..."
        g++ -O2 (Join-Path $SrcDir "matrix_sequential.cpp") -o (Join-Path $BinDir "matrix_sequential_cpp.exe")
        Write-Host "  [>] Compiling OpenMP C++ (matrix_openmp.cpp)..."
        g++ -O2 -fopenmp (Join-Path $SrcDir "matrix_openmp.cpp") -o (Join-Path $BinDir "matrix_openmp_cpp.exe")
    }

    Write-Host "[+] Build complete. Binaries stored in $BinDir" -ForegroundColor Green
}

function Run-Tests {
    param([int]$n)
    Write-Host "`n[*] Running Verification Test with Matrix Size N=${n}:" -ForegroundColor Yellow

    $seqExe = Join-Path $BinDir "matrix_sequential_c.exe"
    if (Test-Path $seqExe) {
        Write-Host "`n--- Running Sequential C ---" -ForegroundColor Cyan
        & $seqExe $n
    }

    $ompExe = Join-Path $BinDir "matrix_openmp_c.exe"
    if (Test-Path $ompExe) {
        Write-Host "`n--- Running OpenMP C (4 Threads) ---" -ForegroundColor Cyan
        & $ompExe $n 4
    }
}

function Run-Plots {
    Write-Host "`n[*] Generating Performance Visualizations & Tables:" -ForegroundColor Yellow
    python (Join-Path $ScriptDir "parse_results.py")
    python (Join-Path $ScriptDir "generate_plots.py") --output-dir $ImagesDir
}

function Clean-Binaries {
    Write-Host "`n[*] Removing compiled binaries..." -ForegroundColor Yellow
    if (Test-Path $BinDir) { Remove-Item -Path $BinDir -Recurse -Force }
    Write-Host "[+] Cleanup complete." -ForegroundColor Green
}

switch ($Task) {
    "all" {
        Build-Binaries
        Run-Tests -n $Size
        Run-Plots
    }
    "build" {
        Build-Binaries
    }
    "test" {
        Run-Tests -n $Size
    }
    "plots" {
        Run-Plots
    }
    "clean" {
        Clean-Binaries
    }
}

Write-Host "`n[*] Automation Task '$Task' Completed Successfully!`n" -ForegroundColor Green
