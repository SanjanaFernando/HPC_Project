#ifndef PARTICLE_H
#define PARTICLE_H

/* Single particle record used by all three N-body backends.
 *
 * The trailing _pad field rounds the struct up to 32 bytes. This eliminates
 * false sharing between adjacent particles in cache-line-sized accesses,
 * which matters once OpenMP threads or CUDA warps start writing to
 * neighbouring entries of the array. The pad is never read or written by
 * the simulation itself. */

typedef struct {
    float mass;
    float x,  y,  z;    /* position */
    float vx, vy, vz;   /* velocity */
    float _pad;
} Particle;

#endif /* PARTICLE_H */