/* OpenMP N-body simulation.
 *
 * Same algorithm as the sequential version (direct O(N^2) force, Euler
 * integrator). The outer i-loop is parallelised; each thread reads from
 * the shared particles array and writes only to particles[i], so no
 * synchronisation is required. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "particle.h"
#include "nbody_io.h"

#define N            10240
#define SOFTENING    1e-9f
#define DT           0.01f
#define G            1.0f
#define STEPS        100
#define INITIAL_FILE "initial_particles.txt"
#define RESULT_FILE  "results/final_particles_openmp.txt"
#define SEED         42u

static void body_force(Particle *particles, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;

            float distSqr  = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist  = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float a = G * particles[j].mass * invDist3;

            Fx += dx * a;
            Fy += dy * a;
            Fz += dz * a;
        }

        particles[i].vx += DT * Fx;
        particles[i].vy += DT * Fy;
        particles[i].vz += DT * Fz;
    }
}

int main(void) {
    Particle *particles = (Particle *) malloc(N * sizeof(Particle));
    if (!particles) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    if (!load_particles(INITIAL_FILE, particles, N)) {
        generate_particles(particles, N, SEED);
        if (!save_particles(INITIAL_FILE, particles, N))
            fprintf(stderr, "Warning: failed to save %s\n", INITIAL_FILE);
    }

    int actual_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    printf("OpenMP N-body simulation\n");
    printf("------------------------\n");
    printf("Particles       : %d\n", N);
    printf("Steps           : %d\n", STEPS);
    printf("Threads         : %d\n", actual_threads);
    printf("Softening       : %.1e\n", SOFTENING);
    printf("Time step (dt)  : %.3f\n\n", DT);

    double total_time = 0.0;

    for (int step = 0; step < STEPS; step++) {
        double t0 = omp_get_wtime();

        body_force(particles, N);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            particles[i].x += particles[i].vx * DT;
            particles[i].y += particles[i].vy * DT;
            particles[i].z += particles[i].vz * DT;
        }

        double elapsed = omp_get_wtime() - t0;
        total_time += elapsed;

        if (step % 10 == 0)
            printf("Step %3d  time: %.4f s\n", step, elapsed);
    }

    printf("\nTotal simulation time : %.4f seconds\n", total_time);
    printf("Average time/step     : %.6f seconds\n", total_time / STEPS);
    print_checksum(particles, N);

    if (!save_particles(RESULT_FILE, particles, N))
        fprintf(stderr, "Warning: failed to write %s\n", RESULT_FILE);

    free(particles);
    return 0;
}