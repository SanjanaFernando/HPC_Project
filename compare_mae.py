#!/usr/bin/env python3
"""Compare final particle outputs against the sequential baseline using Mean Absolute Error.

The script reads the shared particle output format:
  mass x y z vx vy vz

It compares each backend with the sequential result and writes:
    - results/average_error_summary.csv
    - results/average_error_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


COMPONENTS = ["mass", "x", "y", "z", "vx", "vy", "vz"]
DEFAULT_TARGETS = [
    ("OpenMP", Path("results/final_particles_openmp.txt")),
    ("MPI", Path("results/final_particles_mpi.txt")),
    ("CUDA_MPI", Path("results/final_particles_cuda_mpi.txt")),
]


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: install with `pip install matplotlib`\n" + str(exc)
        )


def read_particles(path: Path):
    rows = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 7:
            raise SystemExit(f"Invalid particle line in {path} at line {line_no}: expected 7 floats, found {len(parts)}")
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise SystemExit(f"Invalid float in {path} at line {line_no}: {exc}") from exc
    if not rows:
        raise SystemExit(f"No particle rows found in {path}")
    return rows


def mean_absolute_error(values_a, values_b):
    """Calculate overall Mean Absolute Error (MAE)"""
    if len(values_a) != len(values_b):
        raise SystemExit(f"Mismatched particle counts: {len(values_a)} vs {len(values_b)}")

    total_error = 0.0
    count = 0
    for row_a, row_b in zip(values_a, values_b):
        for val_a, val_b in zip(row_a, row_b):
            diff = abs(val_a - val_b)
            total_error += diff
            count += 1

    return total_error / count if count else float("nan")


def mean_absolute_error_by_group(values_a, values_b):
    """Calculate MAE by group: mass, position, velocity"""
    indices = {
        "mass": [0],
        "position": [1, 2, 3],
        "velocity": [4, 5, 6],
    }
    result = {}
    for name, cols in indices.items():
        total_error = 0.0
        count = 0
        for row_a, row_b in zip(values_a, values_b):
            for idx in cols:
                diff = abs(row_a[idx] - row_b[idx])
                total_error += diff
                count += 1
        result[name] = total_error / count if count else float("nan")
    result["overall"] = mean_absolute_error(values_a, values_b)
    return result


def write_csv(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Program", "MassAvgError", "PositionAvgError", "VelocityAvgError", "OverallAvgError"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def plot_comparison(records, outpath: Path):
    ensure_matplotlib()
    import matplotlib.pyplot as plt

    programs = [item["Program"] for item in records]
    position_vals = [item["PositionAvgError"] for item in records]
    velocity_vals = [item["VelocityAvgError"] for item in records]

    x_positions = list(range(len(programs)))
    width = 0.36

    plt.figure(figsize=(9, 5))
    bars_pos = plt.bar([x - width / 2 for x in x_positions], position_vals, width=width,
                      label="Position Avg Error", color="#4C72B0")
    bars_vel = plt.bar([x + width / 2 for x in x_positions], velocity_vals, width=width,
                      label="Velocity Avg Error", color="#C44E52")

    plt.xticks(x_positions, programs)
    has_zero_or_negative = any(value <= 0.0 for value in position_vals + velocity_vals)
    if has_zero_or_negative:
        plt.yscale("linear")
        plt.ylabel("Mean Absolute Error vs sequential baseline")
    else:
        plt.yscale("log")
        plt.ylabel("Mean Absolute Error vs sequential baseline (log scale)")
    plt.title("Particle output Average Error by backend")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()

    for bars in (bars_pos, bars_vel):
        for bar in bars:
            value = bar.get_height()
            if math.isnan(value) or value <= 0.0:
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()


def parse_target_spec(spec: str):
    if ":" not in spec:
        raise SystemExit("Target format must be Program:Path")
    program, raw_path = spec.split(":", 1)
    program = program.strip()
    path = Path(raw_path.strip())
    if not program:
        raise SystemExit("Target program name cannot be empty")
    return program, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="results/final_particles_sequential.txt",
        help="Sequential baseline file",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional targets in the form Program:Path",
    )
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise SystemExit(f"Baseline file not found: {baseline_path}")

    if args.targets:
        target_specs = [parse_target_spec(spec) for spec in args.targets]
    else:
        target_specs = DEFAULT_TARGETS

    baseline = read_particles(baseline_path)

    records = []
    for program, path in target_specs:
        if not path.exists():
            raise SystemExit(f"Target file not found for {program}: {path}")
        
        target = read_particles(path)
        metrics = mean_absolute_error_by_group(baseline, target)
        
        records.append(
            {
                "Program": program,
                "MassAvgError": f"{metrics['mass']:.10e}",
                "PositionAvgError": float(metrics["position"]),
                "VelocityAvgError": float(metrics["velocity"]),
                "OverallAvgError": float(metrics["overall"]),
            }
        )

    outdir = Path(args.outdir)
    csv_path = outdir / "average_error_summary.csv"
    png_path = outdir / "average_error_comparison.png"
    
    write_csv(csv_path, records)
    plot_comparison(records, png_path)

    print(f"Baseline: {baseline_path}")
    for record in records:
        print(
            f"{record['Program']}: position={record['PositionAvgError']:.6e}, "
            f"velocity={record['VelocityAvgError']:.6e}, overall={record['OverallAvgError']:.6e}"
        )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
