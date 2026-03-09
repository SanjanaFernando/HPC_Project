// particles.h

#ifndef PARTICLE_H
#define PARTICLE_H

typedef struct
{
    float mass;
    float x, y, z;    // position
    float vx, vy, vz; // velocity
} Particle;

#endif