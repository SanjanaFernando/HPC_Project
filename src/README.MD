# N-Body HPC Project

## How to run

This repository contains two C implementations of an N-body simulation:

- `nbody_sequential.c` — sequential implementation
- `nbody_openmp.c` — OpenMP-parallel implementation

Prerequisites

- A C compiler (gcc/clang). On Windows, use MinGW or WSL for best compatibility.

Build

From the `src/` directory, build the programs as follows.

Linux / WSL:

```bash
gcc -O2 -o nbody_seq nbody_sequential.c -lm
gcc -O2 -fopenmp -o nbody_omp nbody_openmp.c -lm
```

Windows (PowerShell with MinGW):

```powershell
gcc -O2 -o nbody_seq.exe nbody_sequential.c -lm
gcc -O2 -fopenmp -o nbody_omp.exe nbody_openmp.c -lm
```

Run

Both programs use `initial_particles.txt` (in the `src/` folder) as the input file. They write final particle states to `results/final_particles_sequential.txt` and `results/final_particles_openmp.txt` respectively.

Linux / WSL:

```bash
# Sequential
./nbody_seq

# OpenMP (set threads, e.g. 4)
export OMP_NUM_THREADS=4
./nbody_omp
```

PowerShell (Windows):

```powershell
# Sequential
.\nbody_seq.exe

# OpenMP (set threads, e.g. 4)
$env:OMP_NUM_THREADS = 4
.\nbody_omp.exe
```

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
