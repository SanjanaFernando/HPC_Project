#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "particle.h"

#define N 10240
#define STEPS 100
#define DT 0.01
#define G 1.0
#define EPS 1e-9

void initialize_particles(Particle *particles, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        particles[i].x = (double)rand() / RAND_MAX;
        particles[i].y = (double)rand() / RAND_MAX;
        particles[i].z = (double)rand() / RAND_MAX;

        particles[i].vx = 0.0;
        particles[i].vy = 0.0;
        particles[i].vz = 0.0;

        particles[i].mass = 1.0 + (double)rand() / RAND_MAX;
    }
}

void save_particles(const char *filename, Particle *particles, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing.\n", filename);
        return;
    }

    for (int i = 0; i < n; i++) {
        fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\n",
                particles[i].x, particles[i].y, particles[i].z,
                particles[i].vx, particles[i].vy, particles[i].vz,
                particles[i].mass);
    }

    fclose(fp);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Error: N (%d) must be divisible by number of processes (%d).\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    int local_n = N / size;

    Particle *all_particles = (Particle *)malloc(N * sizeof(Particle));
    Particle *local_particles = (Particle *)malloc(local_n * sizeof(Particle));

    if (!all_particles || !local_particles) {
        printf("Memory allocation failed on rank %d\n", rank);
        MPI_Finalize();
        return 1;
    }

    MPI_Datatype MPI_PARTICLE;
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);

    if (rank == 0) {
        initialize_particles(all_particles, N);
    }

    MPI_Scatter(all_particles, local_n, MPI_PARTICLE,
                local_particles, local_n, MPI_PARTICLE,
                0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    for (int step = 0; step < STEPS; step++) {

        MPI_Allgather(local_particles, local_n, MPI_PARTICLE,
                      all_particles, local_n, MPI_PARTICLE,
                      MPI_COMM_WORLD);

        for (int i = 0; i < local_n; i++) {
            double fx = 0.0, fy = 0.0, fz = 0.0;

            for (int j = 0; j < N; j++) {
                double dx = all_particles[j].x - local_particles[i].x;
                double dy = all_particles[j].y - local_particles[i].y;
                double dz = all_particles[j].z - local_particles[i].z;

                double distSqr = dx * dx + dy * dy + dz * dz + EPS;
                double dist = sqrt(distSqr);
                double invDist3 = 1.0 / (distSqr * dist);

                fx += G * all_particles[j].mass * dx * invDist3;
                fy += G * all_particles[j].mass * dy * invDist3;
                fz += G * all_particles[j].mass * dz * invDist3;
            }

            local_particles[i].vx += fx * DT;
            local_particles[i].vy += fy * DT;
            local_particles[i].vz += fz * DT;
        }

        for (int i = 0; i < local_n; i++) {
            local_particles[i].x += local_particles[i].vx * DT;
            local_particles[i].y += local_particles[i].vy * DT;
            local_particles[i].z += local_particles[i].vz * DT;
        }
    }

    double end_time = MPI_Wtime();

    MPI_Gather(local_particles, local_n, MPI_PARTICLE,
               all_particles, local_n, MPI_PARTICLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI Execution Time with %d processes: %f seconds\n", size, end_time - start_time);
        save_particles("results/final_particles_mpi.txt", all_particles, N);
    }

    MPI_Type_free(&MPI_PARTICLE);
    free(local_particles);
    free(all_particles);

    MPI_Finalize();
    return 0;
}