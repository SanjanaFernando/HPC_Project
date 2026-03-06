#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "particle.h"

#define N          10240       // Number of particles (adjust: 1024, 4096, 8192, 16384...)
#define SOFTENING  1e-9f      // Softening parameter to avoid singularity
#define DT         0.01f      // Time step
#define G          1.0f       // Gravitational constant (normalized units)
#define STEPS      100        // Number of simulation steps
#define INITIAL_FILE "initial_particles.txt"
#define SEED 42u

int loadParticles(const char *filename, Particle *particles, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        return 0;
    }

    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f %f %f %f %f %f %f",
                   &particles[i].mass,
                   &particles[i].x,
                   &particles[i].y,
                   &particles[i].z,
                   &particles[i].vx,
                   &particles[i].vy,
                   &particles[i].vz) != 7) {
            fclose(fp);
            return 0;
        }
    }

    fclose(fp);
    return 1;
}

int saveParticles(const char *filename, Particle *particles, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        return 0;
    }

    for (int i = 0; i < n; i++) {
        fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                particles[i].mass, particles[i].x, particles[i].y, particles[i].z,
                particles[i].vx, particles[i].vy, particles[i].vz);
    }

    fclose(fp);
    return 1;
}

void generateParticles(Particle *particles, int n, unsigned int seed) {
    srand(seed);

    for (int i = 0; i < n; i++) {
        particles[i].mass = 20.0f + ((float)rand() / RAND_MAX) * 80.0f;
        particles[i].x  = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].y  = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].z  = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].vx = 0.0f;
        particles[i].vy = 0.0f;
        particles[i].vz = 0.0f;
    }
}

void bodyForce(Particle *particles, float dt, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;

            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist  = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float f = G * particles[j].mass * invDist3;

            Fx += dx * f;
            Fy += dy * f;
            Fz += dz * f;
        }

        particles[i].vx += Fx * dt;
        particles[i].vy += Fy * dt;
        particles[i].vz += Fz * dt;
    }
}

int main() {
    Particle *particles = (Particle *) malloc(N * sizeof(Particle));
    if (!particles) {
        fprintf(stderr, "Memory allocation failed for particles!\n");
        return 1;
    }

    if (!loadParticles(INITIAL_FILE, particles, N)) {
        generateParticles(particles, N, SEED);
        if (!saveParticles(INITIAL_FILE, particles, N)) {
            fprintf(stderr, "Warning: failed to save %s\n", INITIAL_FILE);
        }
    }

    // Optional: save initial state
    FILE *fp = fopen("initial_particles_openmp.txt", "w");
    if (fp) {
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    particles[i].mass, particles[i].x, particles[i].y, particles[i].z,
                    particles[i].vx, particles[i].vy, particles[i].vz);
        }
        fclose(fp);
    }

    printf("OpenMP N-body simulation\n");
    printf("------------------------\n");
    printf("Particles       : %d\n", N);
    printf("Steps           : %d\n", STEPS);
    printf("Threads         : %d\n", omp_get_max_threads());
    printf("Softening       : %.1e\n", SOFTENING);
    printf("Time step (dt)  : %.3f\n\n", DT);

    double total_time = 0.0;

    for (int step = 0; step < STEPS; step++) {
        double start = omp_get_wtime();

        bodyForce(particles, DT, N);

        // Update positions (Euler method)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            particles[i].x += particles[i].vx * DT;
            particles[i].y += particles[i].vy * DT;
            particles[i].z += particles[i].vz * DT;
        }

        double end = omp_get_wtime();
        double elapsed = end - start;
        total_time += elapsed;

        if (step % 20 == 0) {
            printf("Step %3d   time: %.4f s\n", step, elapsed);
        }
    }

    printf("\nTotal simulation time : %.4f seconds\n", total_time);
    printf("Average time/step     : %.6f seconds\n", total_time / STEPS);

    // Save final state
    fp = fopen("results/final_particles_openmp.txt", "w");
    if (fp) {
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    particles[i].mass, particles[i].x, particles[i].y, particles[i].z,
                    particles[i].vx, particles[i].vy, particles[i].vz);
        }
        fclose(fp);
    }

    free(particles);
    return 0;
}