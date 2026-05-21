# N-Body HPC Project

Three implementations of the same direct O(N²) gravitational N-body
simulation with the Euler integrator:

- `src/nbody_sequential.c` — single-threaded baseline
- `src/nbody_openmp.c` — OpenMP (shared memory)
- `src/nbody_cuda_mpi.cu` — CUDA + MPI (one GPU per MPI rank)

All three read the same `initial_particles.txt`, run identical physics, and
print a final checksum line so the backends can be compared.

## Layout

```
.
├── include/        shared headers (particle.h, nbody_io.h)
├── src/            the three simulation programs
└── results/        per-backend final-state output files (created at runtime)
```

## Prerequisites

- A C compiler (gcc/clang) for the sequential and OpenMP versions
- An OpenMP-capable compiler for the OpenMP version (gcc/clang/MSVC all work)
- For the CUDA + MPI version:
  - NVIDIA CUDA Toolkit (provides `nvcc`)
  - An MPI implementation, e.g. OpenMPI or MPICH (provides `mpirun`/`mpicxx`)
  - An NVIDIA GPU with a compatible driver

On Windows, use WSL2 for the CUDA + MPI build. MinGW handles the sequential
and OpenMP versions fine.

## Build

All commands assume the project root as the working directory.

### Linux / WSL

```bash
# Sequential
gcc -O2 -Iinclude -o nbody_seq src/nbody_sequential.c -lm

# OpenMP
gcc -O2 -fopenmp -Iinclude -o nbody_omp src/nbody_openmp.c -lm

# CUDA + MPI (preferred)
nvcc -O2 -ccbin mpicxx -Iinclude -o nbody_cuda_mpi src/nbody_cuda_mpi.cu

```

### Windows (PowerShell with MinGW) — sequential and OpenMP only

```powershell
gcc -O2 -Iinclude -o nbody_seq.exe src/nbody_sequential.c -lm
gcc -O2 -fopenmp -Iinclude -o nbody_omp.exe src/nbody_openmp.c -lm
```

For CUDA + MPI on Windows, use WSL2 with the Linux instructions above.

## Run

Make sure the `results/` directory exists before the first run:

```bash
mkdir -p results
```

(PowerShell: `New-Item -ItemType Directory results`.)

All three programs read `initial_particles.txt` from the current working
directory. If that file is absent, the program generates a deterministic
random initial state from `SEED=42` and writes it out, so subsequent runs
of any backend start from the same state.

### Linux / WSL

```bash
# Sequential
./nbody_seq

# OpenMP — set the thread count via OMP_NUM_THREADS
export OMP_NUM_THREADS=4
./nbody_omp

# CUDA + MPI — one MPI rank per GPU
mpirun -np 2 ./nbody_cuda_mpi
```

Use `-np 1` if you only have one GPU. Running more ranks than GPUs makes
them share GPUs round-robin, which works but isn't useful for benchmarks.

### Windows (PowerShell with MinGW) — sequential and OpenMP only

```powershell
.\nbody_seq.exe

$env:OMP_NUM_THREADS = 4
.\nbody_omp.exe
```

## What each run produces

Each program prints per-step timings, total wall time, average time per
step, and a final `Checksum:` line summing all positions and velocities.
The three backends' checksums should agree to several significant figures;
small differences are normal floating-point reduction-order effects.

| Backend  | Final state file                          |
|----------|-------------------------------------------|
| seq      | `results/final_particles_sequential.txt`  |
| openmp   | `results/final_particles_openmp.txt`      |
| cuda_mpi | `results/final_particles_cuda_mpi.txt`    |

## Notes

- `N`, `STEPS`, `DT`, `SOFTENING` and other parameters are `#define` macros
  at the top of each source file. Edit and rebuild to change configuration.
  All three files must use the same `N` for a meaningful checksum comparison.
- For the CUDA + MPI version, `N` must be divisible by the number of MPI
  ranks (the program checks at startup and aborts cleanly otherwise).
- The integrator is plain Euler with a very small softening parameter.
  The simulation is not physically stable — particles will fly apart on a
  short time horizon. The project's purpose is the parallel performance
  comparison, not the underlying physics.

## Files of interest

- [include/particle.h](include/particle.h)
- [include/nbody_io.h](include/nbody_io.h)
- [src/nbody_sequential.c](src/nbody_sequential.c)
- [src/nbody_openmp.c](src/nbody_openmp.c)
- [src/nbody_cuda_mpi.cu](src/nbody_cuda_mpi.cu)