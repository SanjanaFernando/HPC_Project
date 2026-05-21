#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "particle.h"

#define N 10240
#define STEPS 100
#define DT 0.01f
#define G 1.0f
#define EPS 1e-9f

#define INITIAL_FILE "inputs/initial_particles.txt"
#define INITIAL_SNAPSHOT_FILE "inputs/initial_particles_mpi.txt"
#define FINAL_FILE "results/final_particles_mpi.txt"
#define SEED 42u

/* ====================== I/O ====================== */
int load_particles(const char *filename, Particle *particles, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f %f %f %f %f %f %f",
                   &particles[i].mass, &particles[i].x, &particles[i].y, &particles[i].z,
                   &particles[i].vx, &particles[i].vy, &particles[i].vz) != 7) {
            fclose(fp);
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

int save_particles(const char *filename, Particle *particles, int n) {
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

void initialize_particles(Particle *particles, int n) {
    srand(SEED);
    for (int i = 0; i < n; i++) {
        particles[i].mass = 20.0f + ((float)rand() / RAND_MAX) * 80.0f;
        particles[i].x   = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].y   = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].z   = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].vx = particles[i].vy = particles[i].vz = 0.0f;
    }
}

/* ====================== Main ====================== */
int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N (%d) must be divisible by processes (%d)\n", N, size);
        MPI_Finalize();
        return 1;
    }

    int local_n = N / size;

    Particle *all_particles  = malloc(N * sizeof(Particle));
    Particle *local_particles = malloc(local_n * sizeof(Particle));

    if (!all_particles || !local_particles) {
        fprintf(stderr, "Memory allocation failed on rank %d\n", rank);
        MPI_Finalize();
        return 1;
    }

    MPI_Datatype MPI_PARTICLE;
    MPI_Type_contiguous(7, MPI_FLOAT, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);

    /* Root loads or generates initial data */
    if (rank == 0) {
        if (!load_particles(INITIAL_FILE, all_particles, N)) {
            initialize_particles(all_particles, N);
            save_particles(INITIAL_FILE, all_particles, N);
        }
        save_particles(INITIAL_SNAPSHOT_FILE, all_particles, N);
    }

    MPI_Scatter(all_particles, local_n, MPI_PARTICLE,
                local_particles, local_n, MPI_PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI N-body simulation (optimized v2)\n");
        printf("-----------------------------------\n");
        printf("Particles : %d\n", N);
        printf("Steps     : %d\n", STEPS);
        printf("Processes : %d\n", size);
        printf("Softening : %.1e\n", EPS);
        printf("dt        : %.3f\n\n", DT);
    }

    double total_time = 0.0;

    for (int step = 0; step < STEPS; step++) {
        double step_start = MPI_Wtime();

        MPI_Allgather(local_particles, local_n, MPI_PARTICLE,
                      all_particles,  local_n, MPI_PARTICLE,
                      MPI_COMM_WORLD);

        /* Force calculation */
        for (int i = 0; i < local_n; i++) {
            float fx = 0.0f, fy = 0.0f, fz = 0.0f;
            const Particle *p_i = &local_particles[i];

            for (int j = 0; j < N; j++) {
                const Particle *p_j = &all_particles[j];

                float dx = p_j->x - p_i->x;
                float dy = p_j->y - p_i->y;
                float dz = p_j->z - p_i->z;

                float distSqr = dx*dx + dy*dy + dz*dz + EPS;
                float dist    = sqrtf(distSqr);
                float invDist3 = 1.0f / (distSqr * dist);

                fx += G * p_j->mass * dx * invDist3;
                fy += G * p_j->mass * dy * invDist3;
                fz += G * p_j->mass * dz * invDist3;
            }

            local_particles[i].vx += fx * DT;
            local_particles[i].vy += fy * DT;
            local_particles[i].vz += fz * DT;
        }

        /* Position update */
        for (int i = 0; i < local_n; i++) {
            local_particles[i].x += local_particles[i].vx * DT;
            local_particles[i].y += local_particles[i].vy * DT;
            local_particles[i].z += local_particles[i].vz * DT;
        }

        double elapsed = MPI_Wtime() - step_start;
        total_time += elapsed;

        if (rank == 0 && step % 20 == 0) {
            printf("Step %3d   time: %.4f s\n", step, elapsed);
        }
    }

    MPI_Gather(local_particles, local_n, MPI_PARTICLE,
               all_particles, local_n, MPI_PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nTotal simulation time : %.4f seconds\n", total_time);
        printf("Average time/step     : %.6f seconds\n", total_time / STEPS);
        save_particles(FINAL_FILE, all_particles, N);
    }

    MPI_Type_free(&MPI_PARTICLE);
    free(local_particles);
    free(all_particles);

    MPI_Finalize();
    return 0;
}