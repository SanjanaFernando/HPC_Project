# N-Body HPC Project

## How to run

This repository contains three C implementations of an N-body simulation:

- `nbody_sequential.c` — sequential implementation
- `nbody_openmp.c` — OpenMP-parallel implementation
- `nbody_mpi.c` — MPI-parallel implementation

Prerequisites

- A C compiler (gcc/clang). On Windows, use MinGW or WSL for best compatibility.

Build

From the `src/` directory, build the programs as follows.

Linux / WSL:

```bash
gcc -O2 -o nbody_seq nbody_sequential.c -lm
gcc -O3 -march=native -fopenmp -o nbody_omp nbody_openmp.c -lm
mpicc -O2 -o nbody_mpi nbody_mpi.c -lm
```

Windows (PowerShell with MinGW):

```powershell
gcc -O2 -o nbody_seq.exe nbody_sequential.c -lm
gcc -O2 -fopenmp -o nbody_omp.exe nbody_openmp.c -lm
mpicc -O2 -o nbody_mpi.exe nbody_mpi.c -lm
```

Run

All three programs use `inputs/initial_particales.txt` as the shared source input and save per-method snapshots into `inputs/`. Final states are written to `results/final_particles_sequential.txt`, `results/final_particles_openmp.txt`, and `results/final_particles_mpi.txt`.

Linux / WSL:

```bash
# Sequential
./nbody_seq

# OpenMP (set threads, e.g. 4)
export OMP_NUM_THREADS=4
./nbody_omp

# MPI (set processes, e.g. 4)
mpirun -np 4 ./nbody_mpi
```

Notes for best performance

- Compile with `-O3 -march=native` and enable OpenMP: `-fopenmp`.
- Optionally enable fast-math with `-ffast-math` for extra speed (may change numerics).

PowerShell (Windows):

```powershell
# Sequential
.\nbody_seq.exe

# OpenMP (set threads, e.g. 4)
$env:OMP_NUM_THREADS = 4
.\nbody_omp.exe

# MPI (set processes, e.g. 4)
mpirun -np 4 .\nbody_mpi.exe
```

Benchmark script

Run all three builds, capture logs, and save a performance summary from WSL:

```bash
chmod +x ./run_all.sh
./run_all.sh -t 8 -p 4
```

This writes logs to `results/logs/`, a CSV summary to `results/performance_summary.csv`, and a readable comparison report to `results/performance_summary.txt`.

Notes

- The number of particles (`N`), number of steps (`STEPS`), time step (`DT`) and other simulation constants are set as macros at the top of each `.c` file — edit those values to run different configurations.
- If `initial_particles.txt` is not present, the programs will generate a random initial state and (in some variants) save it under `inputs/` or the project root.
- Ensure the `results/` directory exists before running, e.g. `mkdir -p results` (WSL/Linux) or `New-Item -ItemType Directory results` (PowerShell) if needed.

Files of interest

- [nbody_sequential.c](nbody_sequential.c)
- [nbody_openmp.c](nbody_openmp.c)
- [particle.h](particle.h)

Results

- Sequential final state: `results/final_particles_sequential.txt`
- OpenMP final state: `results/final_particles_openmp.txt`
