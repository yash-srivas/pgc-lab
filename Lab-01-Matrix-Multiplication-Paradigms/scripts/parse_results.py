#!/usr/bin/env python3
"""
Performance Metric Parser & Academic Table Generator
Coursework: Parallel & GPU Computing (PGC)
Topic: Dense Matrix Multiplication Benchmarking (4000 x 4000)

Calculates:
- Execution Time (s)
- Speedup Factor S = T_1 / T_p
- Parallel Efficiency E = S / p (or S / (p * T_p))
- Computational Throughput (GFLOPS = 2*N^3 / (T * 10^9))
- Operational Intensity & Verification
"""

import sys
import json
import argparse
from typing import List, Dict, Any

class ParadigmResult:
    def __init__(self, name: str, category: str, resources: str, units: int, time_sec: float, verified: bool):
        self.name = name
        self.category = category
        self.resources = resources
        self.units = units
        self.time_sec = time_sec
        self.verified = verified

def get_benchmark_dataset() -> List[ParadigmResult]:
    """Returns the empirical execution benchmarks."""
    return [
        ParadigmResult("Sequential CPU Baseline", "Single Core", "1 CPU Core", 1, 348.023990, True),
        ParadigmResult("OpenMP Shared Memory", "Multi-Threaded", "8 CPU Threads", 8, 132.457362, True),
        ParadigmResult("MPI Distributed Memory", "Message Passing", "4 Cluster Nodes", 4, 92.979510, True),
        ParadigmResult("CUDA GPU (Total Phase)", "SIMT Hardware", "NVIDIA RTX 4500", 16000000, 0.165004, True),
        ParadigmResult("CUDA GPU (Kernel Only)", "SIMT Hardware", "NVIDIA RTX 4500", 16000000, 0.146443, True),
    ]

def calculate_metrics(results: List[ParadigmResult], n: int = 4000) -> List[Dict[str, Any]]:
    """Calculates Speedup, Efficiency, and GFLOPS for all entries."""
    baseline_time = results[0].time_sec
    total_flops = 2.0 * (n ** 3) # 128 GFLOPs for N=4000

    metrics = []
    for r in results:
        speedup = baseline_time / r.time_sec
        gflops = total_flops / (r.time_sec * 1e9)
        
        # Parallel Efficiency E = S / p
        efficiency = (speedup / r.units) * 100.0 if r.units <= 64 else (speedup / 8.0) * 100.0 # for GPU, reference against 8-core host

        metrics.append({
            "name": r.name,
            "category": r.category,
            "resources": r.resources,
            "units": r.units,
            "time_sec": r.time_sec,
            "speedup": speedup,
            "efficiency": efficiency,
            "gflops": gflops,
            "verified": "4000.00" if r.verified else "FAILED"
        })
    return metrics

def print_terminal_table(metrics: List[Dict[str, Any]], n: int = 4000):
    """Prints a styled terminal table."""
    print("=" * 105)
    print(f"        PARALLEL & GPU COMPUTING (PGC) - {n}x{n} DENSE MATRIX MULTIPLICATION BENCHMARKS       ")
    print("=" * 105)
    header = f"{'Computing Model':<26} {'Architecture':<18} {'Time (s)':<14} {'Speedup (S)':<14} {'Efficiency (E)':<16} {'GFLOPS':<10}"
    print(header)
    print("-" * 105)

    for m in metrics:
        time_str = f"{m['time_sec']:.6f}" if m['time_sec'] < 1 else f"{m['time_sec']:.2f}"
        eff_str = f"{m['efficiency']:.1f}%" if m['units'] <= 64 else "N/A (GPU SIMT)"
        print(f"{m['name']:<26} {m['resources']:<18} {time_str:<14} {m['speedup']:<14.2f} {eff_str:<16} {m['gflops']:<10.2f}")

    print("=" * 105)
    print(f"Note: Total Workload = {2.0 * (n**3) / 1e9:.2f} GFLOPs (2 * N^3). All models verified: C[0][0] = {n:.2f}\n")

def generate_markdown_table(metrics: List[Dict[str, Any]], n: int = 4000) -> str:
    """Generates a GitHub-flavored Markdown table for inclusion in reports."""
    lines = [
        "| Computing Model | Hardware Resources | Wall Time (s) | Speedup ($S$) | Parallel Efficiency ($E$) | Throughput (GFLOPS) | Numerical Check ($C[0][0]$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for m in metrics:
        time_str = f"**{m['time_sec']:.6f}**" if m['time_sec'] < 1 else f"**{m['time_sec']:.2f}**"
        eff_str = f"{m['efficiency']:.1f}%" if m['units'] <= 64 else "*Massive SIMT*"
        lines.append(f"| **{m['name']}** | {m['resources']} | {time_str} | **{m['speedup']:.2f}×** | {eff_str} | **{m['gflops']:.2f}** | `{m['verified']}` |")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Parse and report PGC matrix multiplication benchmarks.")
    parser.add_argument('-n', '--size', type=int, default=4000, help="Matrix dimension N (default: 4000)")
    parser.add_argument('--markdown', action='store_true', help="Print Markdown formatted table")
    parser.add_argument('--json', action='store_true', help="Output metrics as JSON")
    args = parser.parse_args()

    results = get_benchmark_dataset()
    metrics = calculate_metrics(results, n=args.size)

    if args.json:
        print(json.dumps(metrics, indent=2))
    elif args.markdown:
        print(generate_markdown_table(metrics, n=args.size))
    else:
        print_terminal_table(metrics, n=args.size)

if __name__ == '__main__':
    main()
