// nbody_sequential.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "particle.h" // or define struct here

#define N 10240         // start small, later increase to 4096, 8192, etc.
#define SOFTENING 1e-9f // avoid division by zero / singularity
#define DT 0.01f        // time step
#define G 1.0f          // gravitational constant (can be 1 in normalized units)
#define STEPS 100       // number of time steps (increase for longer sim)
#define INITIAL_FILE "initial_particles.txt"
#define SEED 42u

int loadParticles(const char *filename, Particle *particles, int n)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        return 0;
    }

    for (int i = 0; i < n; i++)
    {
        if (fscanf(fp, "%f %f %f %f %f %f %f",
                   &particles[i].mass,
                   &particles[i].x,
                   &particles[i].y,
                   &particles[i].z,
                   &particles[i].vx,
                   &particles[i].vy,
                   &particles[i].vz) != 7)
        {
            fclose(fp);
            return 0;
        }
    }

    fclose(fp);
    return 1;
}

int saveParticles(const char *filename, Particle *particles, int n)
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        return 0;
    }

    for (int i = 0; i < n; i++)
    {
        fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                particles[i].mass, particles[i].x, particles[i].y, particles[i].z,
                particles[i].vx, particles[i].vy, particles[i].vz);
    }

    fclose(fp);
    return 1;
}

void generateParticles(Particle *particles, int n, unsigned int seed)
{
    srand(seed);

    for (int i = 0; i < n; i++)
    {
        particles[i].mass = 20.0f + ((float)rand() / RAND_MAX) * 80.0f;
        particles[i].x = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].y = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].z = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].vx = 0.0f;
        particles[i].vy = 0.0f;
        particles[i].vz = 0.0f;
    }
}

void bodyForce(Particle *particles, float dt, int n)
{
    for (int i = 0; i < n; i++)
    {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++)
        {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;

            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float force = G * particles[j].mass * invDist3; // F = G m_j / r^3 * direction (since acc = F/m_i, but m_i cancels later)

            Fx += dx * force;
            Fy += dy * force;
            Fz += dz * force;
        }

        // Update velocity (acceleration = force, since we absorbed 1/m_i already in many codes; here we assume unit mass or adjust)
        particles[i].vx += dt * Fx;
        particles[i].vy += dt * Fy;
        particles[i].vz += dt * Fz;
    }
}

int main()
{
    Particle *particles = (Particle *)malloc(N * sizeof(Particle));
    if (!particles)
    {
        printf("Memory allocation failed!\n");
        return 1;
    }

    if (!loadParticles(INITIAL_FILE, particles, N))
    {
        generateParticles(particles, N, SEED);
        if (!saveParticles(INITIAL_FILE, particles, N))
        {
            fprintf(stderr, "Warning: failed to save %s\n", INITIAL_FILE);
        }
    }

    FILE *fp = fopen("initial_particles_sequential.txt", "w");
    if (fp)
    {
        for (int i = 0; i < N; i++)
        {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    particles[i].mass, particles[i].x, particles[i].y, particles[i].z,
                    particles[i].vx, particles[i].vy, particles[i].vz);
        }
        fclose(fp);
    }

    clock_t start, end;
    double total_time = 0.0;

    printf("Starting sequential N-body simulation (%d particles, %d steps)...\n", N, STEPS);

    for (int step = 0; step < STEPS; step++)
    {
        start = clock();

        bodyForce(particles, DT, N);

        // Update positions after velocity (basic Euler)
        for (int i = 0; i < N; i++)
        {
            particles[i].x += particles[i].vx * DT;
            particles[i].y += particles[i].vy * DT;
            particles[i].z += particles[i].vz * DT;
        }

        end = clock();
        double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
        total_time += elapsed;

        if (step % 10 == 0)
        {
            printf("Step %d: %.4f seconds\n", step, elapsed);
        }
    }

    printf("Total simulation time: %.4f seconds\n", total_time);
    printf("Average time per step: %.6f seconds\n", total_time / STEPS);

    // Save final state
    fp = fopen("output-files/final_particles_sequential.txt", "w");
    if (fp)
    {
        for (int i = 0; i < N; i++)
        {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    particles[i].mass, particles[i].x, particles[i].y, particles[i].z,
                    particles[i].vx, particles[i].vy, particles[i].vz);
        }
        fclose(fp);
    }

    free(particles);
    return 0;
}