#ifndef NBODY_IO_H
#define NBODY_IO_H

/* Header-only I/O and initialisation helpers shared by the three N-body
 * backends. Header-only so each backend can be a single-file build (no
 * linker invocations for a tiny utility). */

#include <stdio.h>
#include <stdlib.h>

#include "particle.h"

/* Read n particles from `filename` (one per line, 7 floats: mass x y z vx vy vz).
 * Returns 1 on success, 0 on any failure (missing file, short read, parse error). */
static inline int load_particles(const char *filename, Particle *particles, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;

    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f %f %f %f %f %f %f",
                   &particles[i].mass,
                   &particles[i].x,  &particles[i].y,  &particles[i].z,
                   &particles[i].vx, &particles[i].vy, &particles[i].vz) != 7) {
            fclose(fp);
            return 0;
        }
        particles[i]._pad = 0.0f;
    }

    fclose(fp);
    return 1;
}

/* Write n particles to `filename` in the same format read by load_particles.
 * Returns 1 on success, 0 if the file could not be opened. */
static inline int save_particles(const char *filename, const Particle *particles, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return 0;

    for (int i = 0; i < n; i++) {
        fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                particles[i].mass,
                particles[i].x,  particles[i].y,  particles[i].z,
                particles[i].vx, particles[i].vy, particles[i].vz);
    }

    fclose(fp);
    return 1;
}

/* Fill `particles` with a deterministic random initial state derived from
 * `seed`. Masses are drawn from [20, 100], positions from [-1, 1] in each
 * axis, velocities are zero. */
static inline void generate_particles(Particle *particles, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        particles[i].mass = 20.0f + ((float)rand() / RAND_MAX) * 80.0f;
        particles[i].x    = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].y    = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].z    = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
        particles[i].vx   = 0.0f;
        particles[i].vy   = 0.0f;
        particles[i].vz   = 0.0f;
        particles[i]._pad = 0.0f;
    }
}

/* Print a simple checksum of all particle positions and velocities, used to
 * sanity-check that the three backends produce equivalent final state. */
static inline void print_checksum(const Particle *particles, int n) {
    double sum_x = 0.0, sum_v = 0.0;
    for (int i = 0; i < n; i++) {
        sum_x += particles[i].x  + particles[i].y  + particles[i].z;
        sum_v += particles[i].vx + particles[i].vy + particles[i].vz;
    }
    printf("Checksum: sum(pos) = %.6e   sum(vel) = %.6e\n", sum_x, sum_v);
}

#endif /* NBODY_IO_H */