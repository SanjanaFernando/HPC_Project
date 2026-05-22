#!/usr/bin/env python3
"""Generate PNG performance charts from results/performance_summary.csv.

Usage:
  python3 plot_performance.py --csv results/performance_summary.csv --outdir results

Outputs:
    - results/performance_total.png
    - results/performance_average.png
    - results/performance_speedup.png
    - results/performance_cuda_block_sizes.png
    - results/performance_steps.png
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
            block_raw = (row.get("BlockSize") or "").strip()
            total_raw = (row.get("TotalSeconds") or "").strip()
            average_raw = (row.get("AverageSeconds") or "").strip()
            block_val = float(block_raw) if block_raw else float("nan")
            total_val = float(total_raw) if total_raw else float("nan")
            average_val = float(average_raw) if average_raw else float("nan")
            rows.append(
                {
                    "Program": (row.get("Program") or "").strip() or "Unknown",
                    "BlockSize": block_raw,
                    "BlockSizeVal": block_val,
                    "TotalSeconds": total_raw,
                    "AverageSeconds": average_raw,
                    "TotalSecondsVal": total_val,
                    "AverageSecondsVal": average_val,
                }
            )
    return rows


def read_step_logs(log_dir):
    patterns = [
        re.compile(r"^Step\s+(\d+):\s+([0-9]*\.?[0-9]+)\s+seconds\s*$"),
        re.compile(r"^Step\s+(\d+)\s+time:\s+([0-9]*\.?[0-9]+)\s+s\s*$"),
    ]

    log_files = {
        "Sequential": log_dir / "sequential.log",
        "OpenMP": log_dir / "openmp.log",
        "MPI": log_dir / "mpi.log",
    }

    for log_path in sorted(log_dir.glob("cuda_mpi*.log")):
        stem = log_path.stem
        if stem.startswith("cuda_mpi_bs"):
            log_files[f"CUDA_MPI_{stem[len('cuda_mpi_'):]}"] = log_path
        elif stem == "cuda_mpi":
            log_files["CUDA_MPI"] = log_path

    series = {}
    for program, log_path in log_files.items():
        if not log_path.exists():
            continue

        points = []
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            for pattern in patterns:
                match = pattern.match(line.strip())
                if match:
                    step = int(match.group(1))
                    elapsed = float(match.group(2))
                    points.append((step, elapsed))
                    break

        if points:
            series[program] = sorted(points, key=lambda item: item[0])

    return series


def plot_cuda_block_sizes(df, outpath):
    import matplotlib.pyplot as plt

    rows = [item for item in df if item["Program"].startswith("CUDA_MPI") and not math.isnan(item["BlockSizeVal"]) and not math.isnan(item["TotalSecondsVal"])]
    if not rows:
        return

    rows = sorted(rows, key=lambda item: item["BlockSizeVal"])
    block_sizes = [int(item["BlockSizeVal"]) for item in rows]
    total_values = [item["TotalSecondsVal"] for item in rows]
    average_values = [item["AverageSecondsVal"] for item in rows]

    fig, (ax_total, ax_avg) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_total.plot(block_sizes, total_values, marker="o", linewidth=2, color="#8172B2")
    ax_total.set_ylabel("Total seconds")
    ax_total.set_title("CUDA + MPI performance by block size")
    ax_total.grid(True, linestyle="--", alpha=0.35)
    for block_size, value in zip(block_sizes, total_values):
        ax_total.text(block_size, value, f"{value:.3f}", ha="center", va="bottom")

    ax_avg.plot(block_sizes, average_values, marker="o", linewidth=2, color="#55A868")
    ax_avg.set_xlabel("CUDA block size")
    ax_avg.set_ylabel("Average seconds/step")
    ax_avg.grid(True, linestyle="--", alpha=0.35)
    for block_size, value in zip(block_sizes, average_values):
        ax_avg.text(block_size, value, f"{value:.6f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


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
    cuda_png = outdir / 'performance_cuda_block_sizes.png'
    steps_png = outdir / 'performance_steps.png'

    plot_total(df, total_png)
    plot_average(df, avg_png)
    plot_speedup(df, speed_png)
    plot_cuda_block_sizes(df, cuda_png)
    plot_steps(step_series, steps_png)

    outputs = [total_png, avg_png, speed_png]
    if any(item["Program"].startswith("CUDA_MPI") and not math.isnan(item["BlockSizeVal"]) for item in df):
        outputs.append(cuda_png)
    if step_series:
        outputs.append(steps_png)
    print("\n".join(f"Wrote: {path}" for path in outputs))


if __name__ == '__main__':
    main()
