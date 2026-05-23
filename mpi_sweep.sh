#!/usr/bin/env bash
set -euo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_root"

bin_dir="$script_root/bin"
results_dir="$script_root/results"
log_dir="$results_dir/logs"
mkdir -p "$bin_dir" "$results_dir" "$log_dir"

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Required command '$cmd' was not found in PATH." >&2
        exit 1
    fi
}

require_command mpicc
require_command mpirun
require_command python3

processors="${1:-1,2,3,4}"
IFS=',' read -r -a proc_list <<< "$processors"

mpi_bin="$bin_dir/nbody_mpi_sweep"

printf 'Compiling MPI binary from src/mpi/nbody_mpi.c...\n'
mpicc -O2 -o "$mpi_bin" src/mpi/nbody_mpi.c -lm

csv_path="$results_dir/mpi_scaling.csv"
echo "Processors,TotalSeconds" > "$csv_path"

for p in "${proc_list[@]}"; do
    echo "Running MPI with $p processes..."
    log_path="$log_dir/mpi_sweep_${p}.log"

    mpirun --allow-run-as-root -np "$p" "$mpi_bin" > "$log_path" 2>&1

    total_seconds=$(grep -oP 'Total simulation time : \K[0-9]*\.?[0-9]+' "$log_path" | tail -n1 || true)
    if [[ -z "$total_seconds" ]]; then
        echo "ERROR: could not parse total time from $log_path" >&2
        tail -n 20 "$log_path" >&2
        exit 1
    fi

    echo "$p,$total_seconds" >> "$csv_path"
    printf "  -> %s processes: %s seconds\n" "$p" "$total_seconds"
done

printf '\nSaved MPI scaling CSV to %s\n' "$csv_path"

python3 plot_mpi_scaling.py --csv "$csv_path" --outdir "$results_dir/performance"
printf 'Generated plot: %s/performance/mpi_scaling.png\n' "$results_dir"
