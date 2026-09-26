#!/usr/bin/env python3
"""
Performance Visualization Engine for Parallel Computing (PGC)
Coursework: Dense Matrix Multiplication Benchmark Analysis (4000 x 4000)

Generates publication-quality performance graphs for:
- Sequential Baseline (1 CPU Core)
- OpenMP (8 CPU Threads, Shared Memory)
- MPI Cluster (4 Nodes / VMs, Distributed Memory)
"""

import os
import sys
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

# Publication-grade aesthetic color palette (Modern Academic / Nature-inspired)
PALETTE = {
    'sequential': '#4A5568',    # Slate Gray
    'openmp': '#E06D53',        # Warm Coral / Terracotta
    'mpi': '#2B6CB0',           # Royal Cobalt Blue
    'ideal': '#A0AEC0',         # Neutral reference
    'accent': '#805AD5'         # Purple accent
}

def set_academic_plot_style():
    """Configures matplotlib rcParams for publication-grade scientific graphics."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'figure.titleweight': 'bold',
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
    })

def get_benchmark_dataset() -> Dict[str, Any]:
    """Provides empirical benchmark data measured across the computational models."""
    return {
        'matrix_dim': 4000,
        'total_flops': 2.0 * (4000 ** 3), # 128 GFLOPs
        'models': [
            'Sequential\n(1 Core Baseline)',
            'OpenMP\n(8 CPU Threads)',
            'MPI Cluster\n(4 Nodes / VMs)'
        ],
        'short_names': ['Sequential', 'OpenMP (8T)', 'MPI (4 Nodes)'],
        'execution_times': [348.023990, 132.457362, 92.979510],
        'speedup_factors': [1.00, 2.63, 3.74],
        'resources': [1, 8, 4],
        'colors': [
            PALETTE['sequential'],
            PALETTE['openmp'],
            PALETTE['mpi']
        ]
    }

def plot_execution_time(data: Dict[str, Any], output_path: str, dpi: int = 300):
    """Plots execution time comparison across paradigms."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)
    
    bars = ax.bar(
        data['models'],
        data['execution_times'],
        color=data['colors'],
        width=0.45,
        edgecolor='#1A202C',
        linewidth=1.1,
        zorder=3
    )

    ax.set_ylim(0, 400)
    ax.set_ylabel('Execution Time in Seconds', labelpad=10)
    ax.set_title('Dense Matrix Multiplication (4000 x 4000) Runtime Comparison', pad=15)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f'{h:.2f} s',
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='#1A202C'
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Generated: {output_path}")

def plot_speedup_comparison(data: Dict[str, Any], output_path: str, dpi: int = 300):
    """Plots empirical speedup factors relative to sequential baseline."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)

    bars = ax.bar(
        data['models'],
        data['speedup_factors'],
        color=data['colors'],
        width=0.45,
        edgecolor='#1A202C',
        linewidth=1.1,
        zorder=3
    )

    ax.set_ylim(0, 4.5)
    ax.set_ylabel('Speedup Factor S = T_1 / T_p', labelpad=10)
    ax.set_title('Parallel Speedup over Sequential Baseline (348.02s)', pad=15)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f'{h:.2f}x',
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='#1A202C'
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Generated: {output_path}")

def plot_parallel_efficiency(data: Dict[str, Any], output_path: str, dpi: int = 300):
    """Evaluates multi-core CPU and cluster parallel efficiency: E = S / p."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    cpu_paradigms = ['OpenMP (8 Threads)', 'MPI (4 Cluster Nodes)']
    observed_speedup = [2.63, 3.74]
    processing_units = [8, 4]
    
    # Efficiency E = S / p * 100%
    efficiencies = [(s / p) * 100.0 for s, p in zip(observed_speedup, processing_units)]
    bar_colors = [PALETTE['openmp'], PALETTE['mpi']]

    bars = ax.bar(
        cpu_paradigms,
        efficiencies,
        color=bar_colors,
        width=0.42,
        edgecolor='#1A202C',
        linewidth=1.1,
        zorder=3
    )

    ax.axhline(100.0, color='#718096', linestyle='--', linewidth=1.4, label='Ideal Linear Scaling (100%)', zorder=2)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Parallel Efficiency E = S / p (%)', labelpad=10)
    ax.set_title('Parallel Efficiency Breakdown: Shared Memory vs. Distributed Memory', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)

    for bar, speedup, units in zip(bars, observed_speedup, processing_units):
        h = bar.get_height()
        ax.annotate(
            f'{h:.1f}%\n(Speedup: {speedup:.2f}x / {units} units)',
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='#1A202C'
        )

    # Contextual annotation explaining memory contention
    ax.text(
        0, 15,
        "RAM Bus Contention\n(Shared Memory Bottleneck)",
        ha='center', fontsize=9, style='italic', color='#742A2A',
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF5F5", ec="#FEB2B2")
    )
    ax.text(
        1, 15,
        "Dedicated VM Memory Controllers\n(Near-Linear Weak/Strong Scaling)",
        ha='center', fontsize=9, style='italic', color='#22543D',
        bbox=dict(boxstyle="round,pad=0.4", fc="#F0FFF4", ec="#9AE6B4")
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Generated: {output_path}")

def plot_comprehensive_dashboard(data: Dict[str, Any], output_path: str, dpi: int = 300):
    """Generates a master 4-panel publication-grade performance dashboard."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), dpi=dpi)
    fig.suptitle('Parallel Computing (PGC) Empirical Benchmark Analysis\nDense Matrix Multiplication (4000 x 4000)',
                 fontsize=15, fontweight='bold', y=0.98)

    # Panel 1: Execution Runtimes
    b1 = axs[0, 0].bar(data['models'], data['execution_times'], color=data['colors'], width=0.45, edgecolor='#1A202C', linewidth=1.1, zorder=3)
    axs[0, 0].set_title('A. Execution Time Comparison (Seconds)', pad=10)
    axs[0, 0].set_ylabel('Execution Time (s)')
    axs[0, 0].set_ylim(0, 400)
    for b in b1:
        axs[0, 0].annotate(f'{b.get_height():.2f} s', (b.get_x() + b.get_width()/2, b.get_height()),
                           xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Panel 2: Execution Time Reduction (%)
    baseline = data['execution_times'][0]
    time_reductions = [((baseline - t) / baseline) * 100.0 for t in data['execution_times']]
    b2 = axs[0, 1].bar(data['models'], time_reductions, color=data['colors'], width=0.45, edgecolor='#1A202C', linewidth=1.1, zorder=3)
    axs[0, 1].set_title('B. Execution Time Reduction vs Baseline (%)', pad=10)
    axs[0, 1].set_ylabel('Time Reduction (%)')
    axs[0, 1].set_ylim(0, 100)
    for b in b2:
        axs[0, 1].annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width()/2, b.get_height()),
                           xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Panel 3: Speedup Factors
    b3 = axs[1, 0].bar(data['short_names'], data['speedup_factors'], color=data['colors'], width=0.45, edgecolor='#1A202C', linewidth=1.1, zorder=3)
    axs[1, 0].set_title('C. Relative Speedup Factor S = T_1 / T_p', pad=10)
    axs[1, 0].set_ylabel('Speedup Factor (x Baseline)')
    axs[1, 0].set_ylim(0, 4.5)
    for b in b3:
        axs[1, 0].annotate(f'{b.get_height():.2f}x', (b.get_x() + b.get_width()/2, b.get_height()),
                           xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Panel 4: Computational Throughput (GFLOPS)
    gflops_list = [(data['total_flops'] / (t * 1e9)) for t in data['execution_times']]
    b4 = axs[1, 1].bar(data['short_names'], gflops_list, color=data['colors'], width=0.45, edgecolor='#1A202C', linewidth=1.1, zorder=3)
    axs[1, 1].set_title('D. Computational Throughput (GFLOPS)', pad=10)
    axs[1, 1].set_ylabel('GFLOPS (Higher is Better)')
    axs[1, 1].set_ylim(0, 2.0)
    for b in b4:
        axs[1, 1].annotate(f'{b.get_height():.2f}', (b.get_x() + b.get_width()/2, b.get_height()),
                           xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(output_path)
    plt.close()
    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate PGC Coursework Visualizations")
    parser.add_argument('--output-dir', type=str, default='images', help="Output directory for generated plots")
    parser.add_argument('--dpi', type=int, default=300, help="Image resolution DPI")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_academic_plot_style()
    data = get_benchmark_dataset()

    plot_execution_time(data, os.path.join(args.output_dir, 'execution_time_comparison.png'), dpi=args.dpi)
    plot_speedup_comparison(data, os.path.join(args.output_dir, 'speedup_comparison.png'), dpi=args.dpi)
    plot_parallel_efficiency(data, os.path.join(args.output_dir, 'parallel_efficiency.png'), dpi=args.dpi)
    plot_comprehensive_dashboard(data, os.path.join(args.output_dir, 'overall_performance_dashboard.png'), dpi=args.dpi)

    print("\nAll publication-grade performance charts generated successfully.")

if __name__ == '__main__':
    main()
