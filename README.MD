# N-Body HPC Project

This project contains multiple implementations of the same direct $O(N^2)$ gravitational N-body simulation with a simple Euler integrator.

## Files

- `nbody_sequential.c` - sequential baseline
- `nbody_openmp.c` - OpenMP shared-memory version
- `nbody_mpi.c` - MPI version
- `nbody_cuda_mpi.cu` - CUDA + MPI version
- `particle.h` - shared particle structure
- `run_all.sh` - WSL/Linux benchmark script
- `plot_performance.py` - generates benchmark charts from the saved logs and summary CSV

## Shared input and output

All programs use the same text input format with 7 floats per particle line:

`mass x y z vx vy vz`

The shared input file is `inputs/initial_particles.txt`.

Each backend writes its final state to `results/`:

- `results/final_particles_sequential.txt`
- `results/final_particles_openmp.txt`
- `results/final_particles_mpi.txt`
- `results/final_particles_cuda_mpi.txt`

## Build

All commands below assume the current directory is this `src/` folder.

### Linux / WSL

```bash
gcc -O2 -o nbody_seq nbody_sequential.c -lm
gcc -O3 -fopenmp -o nbody_omp nbody_openmp.c -lm
mpicc -O2 -o nbody_mpi nbody_mpi.c -lm
nvcc -O2 -ccbin mpicxx -o nbody_cuda_mpi nbody_cuda_mpi.cu
```

If `nvcc -ccbin mpicxx` does not work on your system, try:

```bash
mpicxx -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart \
    -o nbody_cuda_mpi nbody_cuda_mpi.cu
```

### Windows (PowerShell / MinGW)

```powershell
gcc -O2 -o nbody_seq.exe nbody_sequential.c -lm
gcc -O2 -fopenmp -o nbody_omp.exe nbody_openmp.c -lm
mpicc -O2 -o nbody_mpi.exe nbody_mpi.c -lm
```

CUDA + MPI is recommended from WSL on Windows.

## Run

Make sure the `results/` directory exists before the first run:

```bash
mkdir -p results
```

### Linux / WSL

```bash
./nbody_seq

OMP_NUM_THREADS=4 ./nbody_omp

mpirun -np 4 ./nbody_mpi

mpirun -np 4 ./nbody_cuda_mpi
```

### Windows (PowerShell)

```powershell
.\nbody_seq.exe

$env:OMP_NUM_THREADS = 4
.\nbody_omp.exe

mpirun -np 4 .\nbody_mpi.exe
```

## Benchmark script

Run all available builds, capture logs, and save a performance summary from WSL/Linux:

```bash
chmod +x ./run_all.sh
./run_all.sh -t 8 -p 4
```

Useful flags:

```bash
./run_all.sh --skip-mpi
./run_all.sh --skip-cuda
./run_all.sh --clean
```

The script writes:

- `results/logs/`
- `results/performance_summary.csv`
- `results/performance_summary.txt`

If CUDA tools are available, it also benchmarks `nbody_cuda_mpi.cu` and saves `results/final_particles_cuda_mpi.txt`.

## Charts

Generate PNG charts from the benchmark output:

```bash
python plot_performance.py --csv results/performance_summary.csv --logsdir results/logs --outdir results
```

This creates:

- `results/performance_total.png`
- `results/performance_average.png`
- `results/performance_speedup.png`
- `results/performance_steps.png`

## Notes

- `N`, `STEPS`, `DT`, and `SOFTENING` are defined at the top of each source file.
- The per-step log output is printed every 20 steps for the sequential and OpenMP versions, and every 20 steps in the MPI version.
- The CUDA + MPI version prints step timings from rank 0 and writes the final output to `results/final_particles_cuda_mpi.txt`.
- On some OpenMPI setups in WSL, running as root requires `--allow-run-as-root`, which is already handled by `run_all.sh`.
