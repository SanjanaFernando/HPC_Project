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

// Optimized body force using Structure-of-Arrays (SoA) layout and SIMD
void bodyForceSoA(float *x, float *y, float *z, float *mass,
                  float *vx, float *vy, float *vz, float dt, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float yi = y[i];
        float zi = z[i];

        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        // Vectorize inner accumulation across j
        #pragma omp simd reduction(+:Fx,Fy,Fz)
        for (int j = 0; j < n; j++) {
            float dx = x[j] - xi;
            float dy = y[j] - yi;
            float dz = z[j] - zi;

            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist  = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float f = G * mass[j] * invDist3;

            Fx += dx * f;
            Fy += dy * f;
            Fz += dz * f;
        }

        vx[i] += Fx * dt;
        vy[i] += Fy * dt;
        vz[i] += Fz * dt;
    }
}

int main() {
    // Allocate Particle array for loading/generation then convert to SoA
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

    // Allocate SoA arrays
    float *mass = (float*) malloc(sizeof(float) * N);
    float *x    = (float*) malloc(sizeof(float) * N);
    float *y    = (float*) malloc(sizeof(float) * N);
    float *z    = (float*) malloc(sizeof(float) * N);
    float *vx   = (float*) malloc(sizeof(float) * N);
    float *vy   = (float*) malloc(sizeof(float) * N);
    float *vz   = (float*) malloc(sizeof(float) * N);
    if (!mass || !x || !y || !z || !vx || !vy || !vz) {
        fprintf(stderr, "Memory allocation failed for SoA arrays!\n");
        return 1;
    }

    // Convert AoS -> SoA for better memory access patterns
    for (int i = 0; i < N; i++) {
        mass[i] = particles[i].mass;
        x[i]    = particles[i].x;
        y[i]    = particles[i].y;
        z[i]    = particles[i].z;
        vx[i]   = particles[i].vx;
        vy[i]   = particles[i].vy;
        vz[i]   = particles[i].vz;
    }

    // Optional: save initial state (SoA) for reference
    FILE *fp = fopen("inputs/initial_particles_openmp.txt", "w");
    if (fp) {
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    mass[i], x[i], y[i], z[i], vx[i], vy[i], vz[i]);
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

        // compute forces and update velocities (SoA)
        bodyForceSoA(x, y, z, mass, vx, vy, vz, DT, N);

        // Update positions (Euler) — vectorized
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            x[i] += vx[i] * DT;
            y[i] += vy[i] * DT;
            z[i] += vz[i] * DT;
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

    // Save final state (from SoA)
    fp = fopen("results/final_particles_openmp.txt", "w");
    if (fp) {
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    mass[i], x[i], y[i], z[i], vx[i], vy[i], vz[i]);
        }
        fclose(fp);
    }

    free(mass); free(x); free(y); free(z); free(vx); free(vy); free(vz);
    free(particles);
    return 0;
}