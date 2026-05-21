// nbody_cuda_mpi.cu
// CUDA + MPI version of the simple Euler N-body simulation.
// Mirrors nbody_sequential.c / nbody_openmp.c style. Keeps the same physics
// (Euler integrator, identical SOFTENING, identical force expression).
//
// One GPU per MPI rank.
// Each rank owns a contiguous slice of particles. All ranks keep a full copy
// of every particle's mass + position on both host and device. After each
// step, positions are Allgathered so every rank has the up-to-date positions
// for the next force computation. Velocities live only on the owning rank.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "particle.h"

#define N            10240
#define SOFTENING    1e-9f
#define DT           0.01f
#define G            1.0f
#define STEPS        100
#define INITIAL_FILE "inputs/initial_particles.txt"
#define SEED         42u
#define BLOCK_SIZE   128

// ---- tiny CUDA error helper ---------------------------------------------------
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                          \
} while (0)

// ---- host-side file I/O (identical idea to the other two) --------------------
static int loadParticles(const char *filename, Particle *particles, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f %f %f %f %f %f %f",
                   &particles[i].mass,
                   &particles[i].x, &particles[i].y, &particles[i].z,
                   &particles[i].vx, &particles[i].vy, &particles[i].vz) != 7) {
            fclose(fp);
            return 0;
        }
        particles[i]._pad = 0.0f;
    }
    fclose(fp);
    return 1;
}

static int saveParticles(const char *filename, Particle *particles, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return 0;
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                particles[i].mass, particles[i].x, particles[i].y, particles[i].z,
                particles[i].vx, particles[i].vy, particles[i].vz);
    }
    fclose(fp);
    return 1;
}

static void generateParticles(Particle *particles, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        particles[i].mass = 20.0f + ((float)rand() / RAND_MAX) * 80.0f;
        particles[i].x  = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].y  = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].z  = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].vx = 0.0f;
        particles[i].vy = 0.0f;
        particles[i].vz = 0.0f;
        particles[i]._pad = 0.0f;
    }
}

// ---- CUDA kernels ------------------------------------------------------------
// Force kernel: each thread computes the velocity update for one OWNED particle.
//   d_pos      : full N positions+mass (Particle layout, read-only this kernel)
//   d_vel      : velocity slice for this rank (local_n entries, Particle layout)
//   first_idx  : global index of the first owned particle
//   local_n    : number of owned particles
//   total_n    : total particle count N
__global__ void bodyForceKernel(const Particle *d_pos,
                                Particle *d_vel,
                                int first_idx,
                                int local_n,
                                int total_n)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= local_n) return;

    int gi = first_idx + li;
    float xi = d_pos[gi].x;
    float yi = d_pos[gi].y;
    float zi = d_pos[gi].z;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < total_n; j++) {
        float dx = d_pos[j].x - xi;
        float dy = d_pos[j].y - yi;
        float dz = d_pos[j].z - zi;
        float distSqr  = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist  = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float f = G * d_pos[j].mass * invDist3;
        Fx += dx * f;
        Fy += dy * f;
        Fz += dz * f;
    }

    // Same convention as the C versions: "force" is really acceleration here.
    d_vel[li].vx += Fx * DT;
    d_vel[li].vy += Fy * DT;
    d_vel[li].vz += Fz * DT;
}

// Position update for owned particles. Writes back into the global d_pos array
// at the owned slice; the Allgather afterwards distributes everyone's slice.
__global__ void positionUpdateKernel(Particle *d_pos,
                                     const Particle *d_vel,
                                     int first_idx,
                                     int local_n)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= local_n) return;
    int gi = first_idx + li;
    d_pos[gi].x += d_vel[li].vx * DT;
    d_pos[gi].y += d_vel[li].vy * DT;
    d_pos[gi].z += d_vel[li].vz * DT;
}

// ---- main --------------------------------------------------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "N (%d) must be divisible by number of ranks (%d)\n", N, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int local_n  = N / size;
    int first_idx = rank * local_n;

    // Pick a GPU. Wraps in case there are fewer devices than ranks.
    int nDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % nDevices));

    // Rank 0 loads or generates particles, then broadcasts to everyone.
    Particle *particles = (Particle *) malloc(N * sizeof(Particle));
    if (!particles) {
        fprintf(stderr, "[rank %d] host alloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        if (!loadParticles(INITIAL_FILE, particles, N)) {
            generateParticles(particles, N, SEED);
            if (!saveParticles(INITIAL_FILE, particles, N))
                fprintf(stderr, "Warning: failed to save %s\n", INITIAL_FILE);
        }
    }
    MPI_Bcast(particles, N * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Save a copy of the initial snapshot specific to the CUDA run so
       each variant records its own initial input for later comparison. */
    if (rank == 0) {
        if (!saveParticles("inputs/initial_particles_cuda.txt", particles, N)) {
            fprintf(stderr, "Warning: failed to save inputs/initial_particles_cuda.txt\n");
        }
    }

    // GPU buffers: full positions (N) + local velocities (local_n).
    Particle *d_pos = NULL;
    Particle *d_vel = NULL;
    CUDA_CHECK(cudaMalloc(&d_pos, N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_vel, local_n * sizeof(Particle)));

    CUDA_CHECK(cudaMemcpy(d_pos, particles, N * sizeof(Particle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, particles + first_idx,
                          local_n * sizeof(Particle), cudaMemcpyHostToDevice));

    // Banner from rank 0.
    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, rank % nDevices);
        printf("CUDA + MPI N-body simulation\n");
        printf("----------------------------\n");
        printf("Particles       : %d\n", N);
        printf("Steps           : %d\n", STEPS);
        printf("MPI ranks       : %d  (1 GPU per rank)\n", size);
        printf("Local per rank  : %d\n", local_n);
        printf("Rank 0 GPU      : %s\n", prop.name);
        printf("Softening       : %.1e\n", SOFTENING);
        printf("Time step (dt)  : %.3f\n\n", DT);
    }

    int blocks = (local_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double total_time = 0.0;

    for (int step = 0; step < STEPS; step++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // Forces on owned particles.
        bodyForceKernel<<<blocks, BLOCK_SIZE>>>(d_pos, d_vel, first_idx, local_n, N);

        // Position update for owned particles (writes into global slice).
        positionUpdateKernel<<<blocks, BLOCK_SIZE>>>(d_pos, d_vel, first_idx, local_n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Bring owned positions back to host, Allgather so every rank has the
        // full updated positions, then push back to device for next step.
        CUDA_CHECK(cudaMemcpy(particles + first_idx, d_pos + first_idx,
                              local_n * sizeof(Particle), cudaMemcpyDeviceToHost));

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                      particles, local_n * sizeof(Particle), MPI_BYTE,
                      MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_pos, particles, N * sizeof(Particle),
                              cudaMemcpyHostToDevice));

        double t1 = MPI_Wtime();
        double elapsed = t1 - t0;
        total_time += elapsed;

        if (rank == 0 && step % 10 == 0)
            printf("Step %d: %.4f seconds\n", step, elapsed);
    }

    // Bring final state for the owned slice back to the host. d_pos contains
    // up-to-date positions+mass for ALL particles, but d_vel only contains
    // velocities for this rank's slice. Copy each from the right buffer.
    {
        Particle *tmp_pos = (Particle *) malloc(local_n * sizeof(Particle));
        Particle *tmp_vel = (Particle *) malloc(local_n * sizeof(Particle));
        CUDA_CHECK(cudaMemcpy(tmp_pos, d_pos + first_idx,
                              local_n * sizeof(Particle), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_vel, d_vel,
                              local_n * sizeof(Particle), cudaMemcpyDeviceToHost));
        for (int i = 0; i < local_n; i++) {
            particles[first_idx + i].mass = tmp_pos[i].mass;
            particles[first_idx + i].x    = tmp_pos[i].x;
            particles[first_idx + i].y    = tmp_pos[i].y;
            particles[first_idx + i].z    = tmp_pos[i].z;
            particles[first_idx + i].vx   = tmp_vel[i].vx;
            particles[first_idx + i].vy   = tmp_vel[i].vy;
            particles[first_idx + i].vz   = tmp_vel[i].vz;
        }
        free(tmp_pos);
        free(tmp_vel);
    }

    // Gather final state to rank 0.
    MPI_Gather(rank == 0 ? MPI_IN_PLACE : particles + first_idx,
               local_n * sizeof(Particle), MPI_BYTE,
               particles,
               local_n * sizeof(Particle), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nTotal simulation time : %.4f seconds\n", total_time);
        printf("Average time/step     : %.6f seconds\n", total_time / STEPS);

        double sum_x = 0.0, sum_v = 0.0;
        for (int i = 0; i < N; i++) {
            sum_x += particles[i].x + particles[i].y + particles[i].z;
            sum_v += particles[i].vx + particles[i].vy + particles[i].vz;
        }
        printf("Checksum: sum(pos) = %.6e   sum(vel) = %.6e\n", sum_x, sum_v);

        FILE *fp = fopen("results/final_particles_cuda_mpi.txt", "w");
        if (fp) {
            for (int i = 0; i < N; i++) {
                fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                        particles[i].mass,
                        particles[i].x, particles[i].y, particles[i].z,
                        particles[i].vx, particles[i].vy, particles[i].vz);
            }
            fclose(fp);
        } else {
            fprintf(stderr, "Warning: could not open results/final_particles_cuda_mpi.txt\n");
        }
    }

    cudaFree(d_pos);
    cudaFree(d_vel);
    free(particles);
    MPI_Finalize();
    return 0;
}
