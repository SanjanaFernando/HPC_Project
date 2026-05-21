#!/usr/bin/env python3
"""Generate PNG performance charts from results/performance_summary.csv.

Usage:
  python3 plot_performance.py --csv results/performance_summary.csv --outdir results

Outputs:
  - results/performance/total.png
  - results/erformance/average.png
  - results/performance/speedup.png
    - results/performance/steps.png
"""
import argparse
import csv
import math
import re
from pathlib import Path


def ensure_deps():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: install with `pip3 install matplotlib`\n" + str(exc)
        )


def read_csv(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_raw = (row.get("TotalSeconds") or "").strip()
            average_raw = (row.get("AverageSeconds") or "").strip()
            total_val = float(total_raw) if total_raw else float("nan")
            average_val = float(average_raw) if average_raw else float("nan")
            rows.append(
                {
                    "Program": (row.get("Program") or "").strip() or "Unknown",
                    "TotalSeconds": total_raw,
                    "AverageSeconds": average_raw,
                    "TotalSecondsVal": total_val,
                    "AverageSecondsVal": average_val,
                }
            )
    return rows


def read_step_logs(log_dir):
    patterns = {
        "Sequential": [
            re.compile(r"^Step\s+(\d+):\s+([0-9]*\.?[0-9]+)\s+seconds\s*$"),
        ],
        "OpenMP": [
            re.compile(r"^Step\s+(\d+)\s+time:\s+([0-9]*\.?[0-9]+)\s+s\s*$"),
        ],
        "MPI": [
            re.compile(r"^Step\s+(\d+)\s+time:\s+([0-9]*\.?[0-9]+)\s+s\s*$"),
        ],
        "CUDA_MPI": [
            re.compile(r"^Step\s+(\d+):\s+([0-9]*\.?[0-9]+)\s+seconds\s*$"),
            re.compile(r"^Step\s+(\d+)\s+time:\s+([0-9]*\.?[0-9]+)\s+s\s*$"),
        ],
    }

    log_files = {
        "Sequential": log_dir / "sequential.log",
        "OpenMP": log_dir / "openmp.log",
        "MPI": log_dir / "mpi.log",
        "CUDA_MPI": log_dir / "cuda_mpi.log",
    }

    series = {}
    for program, log_path in log_files.items():
        if not log_path.exists():
            continue

        points = []
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            for pattern in patterns[program]:
                match = pattern.match(line.strip())
                if match:
                    step = int(match.group(1))
                    elapsed = float(match.group(2))
                    points.append((step, elapsed))
                    break

        if points:
            series[program] = sorted(points, key=lambda item: item[0])

    return series


def plot_total(df, outpath):
    import matplotlib.pyplot as plt

    df_plot = sorted(df, key=lambda item: (math.inf if math.isnan(item["TotalSecondsVal"]) else item["TotalSecondsVal"]))
    programs = [item["Program"] for item in df_plot]
    values = [item["TotalSecondsVal"] for item in df_plot]

    plt.figure(figsize=(8, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = plt.bar(programs, values, color=colors[: len(programs)])
    plt.ylabel("Total seconds")
    plt.title("Total simulation time by program")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values):
        if not math.isnan(val):
            plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_average(df, outpath):
    import matplotlib.pyplot as plt

    df_plot = sorted(df, key=lambda item: (math.inf if math.isnan(item["AverageSecondsVal"]) else item["AverageSecondsVal"]))
    programs = [item["Program"] for item in df_plot]
    values = [item["AverageSecondsVal"] for item in df_plot]

    plt.figure(figsize=(8, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = plt.bar(programs, values, color=colors[: len(programs)])
    plt.ylabel("Average seconds per step")
    plt.title("Average time per step by program")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values):
        if not math.isnan(val):
            plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.6f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_speedup(df, outpath):
    import matplotlib.pyplot as plt

    # Use Sequential as baseline if available and numeric
    baseline_row = next((item for item in df if item["Program"] == "Sequential" and not math.isnan(item["TotalSecondsVal"])), None)
    if baseline_row is not None:
        baseline = float(baseline_row["TotalSecondsVal"])
    else:
        # fallback: fastest program (smallest total) as baseline
        numeric_rows = [item for item in df if not math.isnan(item["TotalSecondsVal"])]
        baseline = min((item["TotalSecondsVal"] for item in numeric_rows), default=None)

    programs = [item["Program"] for item in df]
    vals = []
    for item in df:
        v = item["TotalSecondsVal"]
        try:
            if baseline is None or math.isnan(v) or v == 0:
                vals.append(float('nan'))
            else:
                vals.append(baseline / float(v))
        except Exception:
            vals.append(float('nan'))

    plt.figure(figsize=(8, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = plt.bar(programs, vals, color=colors[: len(programs)])
    plt.ylabel("Speedup (vs baseline)")
    plt.title("Speedup relative to baseline")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, vals):
        if not math.isnan(val):
            plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}x", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_steps(step_series, outpath):
    import matplotlib.pyplot as plt

    if not step_series:
        return

    plt.figure(figsize=(9, 4.8))
    palette = {
        "Sequential": "#4C72B0",
        "OpenMP": "#55A868",
        "MPI": "#C44E52",
        "CUDA_MPI": "#8172B2",
    }

    for program, points in step_series.items():
        steps = [step for step, _ in points]
        values = [elapsed for _, elapsed in points]
        plt.plot(
            steps,
            values,
            marker="o",
            linewidth=2,
            label=program,
            color=palette.get(program),
        )

    plt.xlabel("Simulation step")
    plt.ylabel("Time per step (seconds)")
    plt.title("Per-step performance at steps 0, 20, 40, ...")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ensure_deps()
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results/performance_summary.csv')
    parser.add_argument('--outdir', default='results')
    parser.add_argument('--logsdir', default='results/logs')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_csv(csv_path)
    step_series = read_step_logs(Path(args.logsdir))

    total_png = outdir / 'performance_total.png'
    avg_png = outdir / 'performance_average.png'
    speed_png = outdir / 'performance_speedup.png'
    steps_png = outdir / 'performance_steps.png'

    plot_total(df, total_png)
    plot_average(df, avg_png)
    plot_speedup(df, speed_png)
    plot_steps(step_series, steps_png)

    outputs = [total_png, avg_png, speed_png]
    if step_series:
        outputs.append(steps_png)
    print("\n".join(f"Wrote: {path}" for path in outputs))


if __name__ == '__main__':
    main()
