#!/usr/bin/env python3
"""Plot MPI scaling from a processors-vs-total-seconds CSV file."""

import argparse
import csv
from pathlib import Path


def read_csv(path):
    rows = []
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            proc = int(row['Processors'])
            total = float(row['TotalSeconds'])
            rows.append({'Processors': proc, 'TotalSeconds': total})
    return sorted(rows, key=lambda item: item['Processors'])


def plot_scaling(data, outpath):
    import matplotlib.pyplot as plt

    procs = [item['Processors'] for item in data]
    totals = [item['TotalSeconds'] for item in data]

    plt.figure(figsize=(8, 5))
    plt.plot(procs, totals, marker='o', linewidth=2, color='#C44E52')
    plt.xticks(procs)
    plt.xlabel('MPI processes')
    plt.ylabel('Total simulation time (seconds)')
    plt.title('MPI strong scaling: processors vs total seconds')
    plt.grid(True, linestyle='--', alpha=0.4)

    for x, y in zip(procs, totals):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot MPI scaling results.')
    parser.add_argument('--csv', default='results/mpi_scaling.csv', help='Input CSV file')
    parser.add_argument('--outdir', default='results/performance', help='Output directory for PNG')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f'CSV file not found: {csv_path}')

    data = read_csv(csv_path)
    if not data:
        raise SystemExit(f'No data found in {csv_path}')

    outpath = Path(args.outdir) / 'mpi_scaling.png'
    plot_scaling(data, outpath)
    print(f'Wrote: {outpath}')


if __name__ == '__main__':
    main()
