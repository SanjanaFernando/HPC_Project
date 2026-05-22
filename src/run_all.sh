#!/usr/bin/env bash
set -euo pipefail

OPENMP_THREADS=8
MPI_PROCESSES=4
CUDA_BLOCK_SIZES="64,128,256,512"
SKIP_MPI=0
SKIP_OPENMP=0
SKIP_CUDA=0

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [-t threads] [-p processes] [--skip-mpi] [--skip-cuda] [--cuda-block-sizes sizes]

Options:
  -t, --threads        OpenMP thread count (default: 8)
  -p, --processes      MPI process count (default: 4)
      --skip-mpi       Build and run only sequential + OpenMP
      --skip-openmp    Build and run only sequential + CUDA + MPI
      --skip-cuda      Skip CUDA + MPI build and run
      --cuda-block-sizes  Comma-separated CUDA block sizes (default: 64,128,256,512)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--threads)
            OPENMP_THREADS="${2:?Missing value for $1}"
            shift 2
            ;;
        -p|--processes)
            MPI_PROCESSES="${2:?Missing value for $1}"
            shift 2
            ;;
        --skip-mpi)
            SKIP_MPI=1
            shift
            ;;
        --skip-openmp)
            SKIP_OPENMP=1
            shift
            ;;
        --skip-cuda)
            SKIP_CUDA=1
            shift
            ;;
        --cuda-block-sizes)
            CUDA_BLOCK_SIZES="${2:?Missing value for $1}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_root"

results_dir="$script_root/results"
log_dir="$results_dir/logs"
mkdir -p "$results_dir" "$log_dir"

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command '$command_name' was not found in PATH." >&2
        exit 1
    fi
}

compile_program() {
    local compiler="$1"
    local output_name="$2"
    shift 2
    echo "Compiling $output_name..."
    "$compiler" "$@"
}

run_and_log() {
    local log_path="$1"
    shift
    "$@" > "$log_path" 2>&1
}

extract_metric() {
    local pattern="$1"
    local file_path="$2"
    awk -v pat="$pattern" '
        $0 ~ pat { print $(NF-1); exit }
    ' "$file_path"
}

record_result() {
    local program="$1"
    local log_path="$2"
    local output_path="$3"
    local block_size="$4"

    local total_seconds average_seconds output_exists output_bytes output_hash
    total_seconds="$(extract_metric 'Total simulation time' "$log_path" || true)"
    average_seconds="$(extract_metric 'Average time' "$log_path" || true)"

    if [[ -f "$output_path" ]]; then
        output_exists="true"
        output_bytes="$(stat -c%s "$output_path")"
        output_hash="$(sha256sum "$output_path" | awk '{print $1}')"
    else
        output_exists="false"
        output_bytes=""
        output_hash=""
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$program" \
        "$block_size" \
        "$total_seconds" \
        "$average_seconds" \
        "$output_exists" \
        "$output_bytes" \
        "$output_hash" \
        "$log_path" \
        "$output_path"
}

require_command gcc
require_command python3
if [[ "$SKIP_MPI" -eq 0 ]]; then
    require_command mpicc
    require_command mpirun
fi

seq_bin="nbody_seq_bench"
omp_bin="nbody_omp_bench"
mpi_bin="nbody_mpi_bench"
cuda_bin="nbody_cuda_mpi_bench"

rm -f "$script_root/$seq_bin" "$script_root/$omp_bin" "$script_root/$mpi_bin" "$script_root/$cuda_bin" || true

# Compile sequential
compile_program gcc "$seq_bin" -O2 -o "$script_root/$seq_bin" nbody_sequential.c -lm

# Compile OpenMP
if [[ "$SKIP_OPENMP" -eq 0 ]]; then
    compile_program gcc "$omp_bin" -O2 -fopenmp -o "$script_root/$omp_bin" nbody_openmp.c -lm
fi

if [[ "$SKIP_MPI" -eq 0 ]]; then
    # Compile MPI (place output path explicitly to avoid filename collisions)
    compile_program mpicc "$mpi_bin" -O2 -o "$script_root/$mpi_bin" nbody_mpi.c -lm
fi

cuda_available=0
if [[ "$SKIP_CUDA" -eq 0 ]]; then
    if command -v nvcc >/dev/null 2>&1 && command -v mpicxx >/dev/null 2>&1; then
        cuda_available=1
        compile_program nvcc "$cuda_bin" -O2 -ccbin mpicxx -o "$script_root/$cuda_bin" nbody_cuda_mpi.cu
    else
        echo "Skipping CUDA + MPI build: nvcc and/or mpicxx not found in PATH." >&2
    fi
fi

seq_log="$log_dir/sequential.log"
omp_log="$log_dir/openmp.log"
mpi_log="$log_dir/mpi.log"
summary_csv="$results_dir/performance_summary.csv"
summary_txt="$results_dir/performance_summary.txt"

printf 'Program,BlockSize,TotalSeconds,AverageSeconds,OutputExists,OutputBytes,OutputHash,LogFile,OutputFile\n' > "$summary_csv"
: > "$summary_txt"

printf 'Running sequential...\n'
run_and_log "$seq_log" "$script_root/$seq_bin"
seq_record="$(record_result 'Sequential' "$seq_log" "$results_dir/final_particles_sequential.txt" '')"
printf '%s\n' "$seq_record" >> "$summary_csv"

if [[ "$SKIP_OPENMP" -eq 0 ]]; then
    printf 'Running OpenMP with %s threads...\n' "$OPENMP_THREADS"
    OMP_NUM_THREADS="$OPENMP_THREADS" run_and_log "$omp_log" "$script_root/$omp_bin"
    omp_record="$(record_result 'OpenMP' "$omp_log" "$results_dir/final_particles_openmp.txt" '')"
    printf '%s\n' "$omp_record" >> "$summary_csv"
fi

if [[ "$SKIP_MPI" -eq 0 ]]; then
    printf 'Running MPI with %s processes...\n' "$MPI_PROCESSES"
    # Allow running as root inside some WSL/OpenMPI setups
    run_and_log "$mpi_log" mpirun --allow-run-as-root -np "$MPI_PROCESSES" "$script_root/$mpi_bin"
    mpi_record="$(record_result 'MPI' "$mpi_log" "$results_dir/final_particles_mpi.txt" '')"
    printf '%s\n' "$mpi_record" >> "$summary_csv"
fi

if [[ "$cuda_available" -eq 1 ]]; then
    read -r -a cuda_block_sizes <<< "${CUDA_BLOCK_SIZES//,/ }"
    for block_size in "${cuda_block_sizes[@]}"; do
        if [[ -z "$block_size" ]]; then
            continue
        fi
        printf 'Running CUDA + MPI with %s processes and block size %s...\n' "$MPI_PROCESSES" "$block_size"
        cuda_log_bs="$log_dir/cuda_mpi_bs${block_size}.log"
        cuda_output_bs="$results_dir/final_particles_cuda_mpi_bs${block_size}.txt"
        run_and_log "$cuda_log_bs" mpirun --allow-run-as-root -np "$MPI_PROCESSES" "$script_root/$cuda_bin" --block-size "$block_size"
        cuda_record="$(record_result "CUDA_MPI_bs${block_size}" "$cuda_log_bs" "$cuda_output_bs" "$block_size")"
        printf '%s\n' "$cuda_record" >> "$summary_csv"
    done
fi

python3 - <<'PY' "$summary_csv" "$summary_txt"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
txt_path = Path(sys.argv[2])

rows = []
with csv_path.open(newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        try:
            row['TotalSecondsValue'] = float(row['TotalSeconds'])
        except (TypeError, ValueError):
            row['TotalSecondsValue'] = float('inf')
        try:
            row['AverageSecondsValue'] = float(row['AverageSeconds'])
        except (TypeError, ValueError):
            row['AverageSecondsValue'] = float('inf')
        rows.append(row)

rows_sorted = sorted(rows, key=lambda item: item['TotalSecondsValue'])
baseline = next((row for row in rows if row['Program'] == 'Sequential' and row['TotalSecondsValue'] != float('inf')), None)

with txt_path.open('w', encoding='utf-8') as handle:
    handle.write('Performance comparison (lower is faster)\n')
    handle.write('Program      BlockSize   Total(s)    Avg(s)      OutputBytes\n')
    handle.write('-------------------------------------------------------------\n')
    for row in rows_sorted:
        block_size = row['BlockSize'] if row['BlockSize'] else 'n/a'
        total = row['TotalSeconds'] if row['TotalSeconds'] else 'n/a'
        avg = row['AverageSeconds'] if row['AverageSeconds'] else 'n/a'
        size = row['OutputBytes'] if row['OutputBytes'] else 'n/a'
        handle.write(f"{row['Program']:<12} {block_size:<10} {total:<10} {avg:<10} {size}\n")

    if baseline is not None:
        handle.write('\nSpeedup vs Sequential\n')
        for row in rows_sorted:
            if row['TotalSecondsValue'] != float('inf') and row['TotalSecondsValue'] > 0:
                speedup = baseline['TotalSecondsValue'] / row['TotalSecondsValue']
                handle.write(f"{row['Program']:<12} {speedup:.2f}x\n")
PY

printf '\nPerformance summary saved to %s\n' "$summary_csv"
printf 'Human-readable summary saved to %s\n\n' "$summary_txt"

cat "$summary_csv"
printf '\nDetailed comparison:\n'
cat "$summary_txt"

# Attempt to generate plots using plot_performance.py (optional dependency: matplotlib)
printf '\nGenerating performance plots (if matplotlib available)...\n'
if python3 - <<'PY' 2>/dev/null
try:
    import matplotlib  # noqa: F401
    raise SystemExit(0)
except Exception:
    raise SystemExit(2)
PY
then
    if python3 plot_performance.py --csv "$summary_csv" --outdir "$results_dir/performance" --logsdir "$log_dir"; then
        printf 'Plots generated in %s/performance/ and %s\n' "$results_dir" "$results_dir"
    else
        printf 'plot_performance.py ran but failed (check output)\n'
    fi
else
    printf 'matplotlib not found in the active Python environment. Install with: pip3 install matplotlib\n'
fi
