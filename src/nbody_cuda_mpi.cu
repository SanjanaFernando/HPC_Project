/* CUDA + MPI N-body simulation.
 *
 * Same algorithm as the sequential and OpenMP versions (direct O(N^2)
 * pairwise force with the Euler integrator).
 *
 * Decomposition: one GPU per MPI rank. Each rank owns a contiguous slice
 * of N/size particles and computes the force on its owned particles only.
 * Every rank keeps a full copy of all N positions/masses on both host and
 * device so the force kernel can read all interaction partners without
 * extra communication during the kernel call. After each timestep the
 * updated positions are Allgathered across ranks. Velocities are private
 * to the owning rank. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "particle.h"
#include "nbody_io.h"

#define N            10240
#define SOFTENING    1e-9f
#define DT           0.01f
#define G            1.0f
#define STEPS        100
#define INITIAL_FILE "initial_particles.txt"
#define RESULT_FILE  "results/final_particles_cuda_mpi.txt"
#define SEED         42u
#define BLOCK_SIZE   128

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                             \
                __FILE__, __LINE__, cudaGetErrorString(_e));                  \
        MPI_Abort(MPI_COMM_WORLD, 1);                                         \
    }                                                                         \
} while (0)

/* Compute the velocity update for each owned particle by summing the
 * pairwise contribution from every particle in d_pos. One CUDA thread
 * handles one owned particle. */
__global__ void body_force_kernel(const Particle *d_pos,
                                  Particle       *d_vel,
                                  int             first_idx,
                                  int             local_n,
                                  int             total_n)
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

        float a = G * d_pos[j].mass * invDist3;

        Fx += dx * a;
        Fy += dy * a;
        Fz += dz * a;
    }

    d_vel[li].vx += Fx * DT;
    d_vel[li].vy += Fy * DT;
    d_vel[li].vz += Fz * DT;
}

/* Advance positions of owned particles using their updated velocities. */
__global__ void position_update_kernel(Particle       *d_pos,
                                       const Particle *d_vel,
                                       int             first_idx,
                                       int             local_n)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= local_n) return;
    int gi = first_idx + li;
    d_pos[gi].x += d_vel[li].vx * DT;
    d_pos[gi].y += d_vel[li].vy * DT;
    d_pos[gi].z += d_vel[li].vz * DT;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "N (%d) must be divisible by the number of ranks (%d)\n",
                    N, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int local_n   = N / size;
    int first_idx = rank * local_n;

    /* Bind this rank to a GPU. Ranks beyond the GPU count wrap round-robin. */
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % n_devices));

    /* Rank 0 loads or generates the initial state, then broadcasts it. */
    Particle *particles = (Particle *) malloc(N * sizeof(Particle));
    if (!particles) {
        fprintf(stderr, "[rank %d] host allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        if (!load_particles(INITIAL_FILE, particles, N)) {
            generate_particles(particles, N, SEED);
            if (!save_particles(INITIAL_FILE, particles, N))
                fprintf(stderr, "Warning: failed to save %s\n", INITIAL_FILE);
        }
    }
    MPI_Bcast(particles, N * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Device buffers: full positions for every rank, owned velocities only. */
    Particle *d_pos = NULL;
    Particle *d_vel = NULL;
    CUDA_CHECK(cudaMalloc(&d_pos, N       * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_vel, local_n * sizeof(Particle)));

    CUDA_CHECK(cudaMemcpy(d_pos, particles,
                          N * sizeof(Particle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, particles + first_idx,
                          local_n * sizeof(Particle), cudaMemcpyHostToDevice));

    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, rank % n_devices);
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

        body_force_kernel      <<<blocks, BLOCK_SIZE>>>(d_pos, d_vel,
                                                        first_idx, local_n, N);
        position_update_kernel <<<blocks, BLOCK_SIZE>>>(d_pos, d_vel,
                                                        first_idx, local_n);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Pull the owned slice back to host, share it with all ranks,
         * then refresh device positions for the next step. */
        CUDA_CHECK(cudaMemcpy(particles + first_idx, d_pos + first_idx,
                              local_n * sizeof(Particle),
                              cudaMemcpyDeviceToHost));

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                      particles, local_n * sizeof(Particle), MPI_BYTE,
                      MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_pos, particles,
                              N * sizeof(Particle),
                              cudaMemcpyHostToDevice));

        double elapsed = MPI_Wtime() - t0;
        total_time += elapsed;

        if (rank == 0 && step % 10 == 0)
            printf("Step %3d  time: %.4f s\n", step, elapsed);
    }

    /* `particles` already has up-to-date positions for every rank from the
     * last Allgather. Drop the owned velocities in on top to complete it. */
    {
        Particle *vel_slice = (Particle *) malloc(local_n * sizeof(Particle));
        CUDA_CHECK(cudaMemcpy(vel_slice, d_vel,
                              local_n * sizeof(Particle),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < local_n; i++) {
            particles[first_idx + i].vx = vel_slice[i].vx;
            particles[first_idx + i].vy = vel_slice[i].vy;
            particles[first_idx + i].vz = vel_slice[i].vz;
        }
        free(vel_slice);
    }

    /* Gather final velocities to rank 0. Positions are already complete on
     * every rank, but we use an Allgather-style call here so the velocity
     * slices from non-zero ranks reach rank 0. */
    if (rank == 0) {
        MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   particles, local_n * sizeof(Particle), MPI_BYTE,
                   0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(particles + first_idx, local_n * sizeof(Particle), MPI_BYTE,
                   NULL, 0, MPI_DATATYPE_NULL,
                   0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("\nTotal simulation time : %.4f seconds\n", total_time);
        printf("Average time/step     : %.6f seconds\n", total_time / STEPS);
        print_checksum(particles, N);

        if (!save_particles(RESULT_FILE, particles, N))
            fprintf(stderr, "Warning: failed to write %s\n", RESULT_FILE);
    }

    cudaFree(d_pos);
    cudaFree(d_vel);
    free(particles);
    MPI_Finalize();
    return 0;
}
