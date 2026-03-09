# High Performance Computing Project Submission
## N-Body Simulation with SDL3

#### Contributers :
4514  
4516  
4520  

### Background
N-body simulation is a simulation of dynamic set of particles, under the influence of a single or
multiple physical forces. Particle simulation can be seen from different perspectives, from caustic
simulations to astrophysics simulation. For this project, let see the N-bodies as the cosmological
objects. As the cosmological bodies are extreme to maintain in simulation levels, these particle systems will heavily depend on various assumptions. N-body simulations are simple in principle but as the number of the particles involved in the simulation increases, number of interactions that needed to be calculated increases. According to the Newtonian Gravitational Law,
$$
  F = G \frac{m_1 m_2}{r^2}
$$

As the number of particles increases, interactions needed to be calculated increases in $𝑁^2$, resulting a
computational complexity of $𝑂(𝑁^2)$.

### Description
Project aims to create an N-body Simulation (Particle Simulation) with each particle having a volume
and a mass. Therefore according to the gravitational laws, forces in between each particle has to be
calculated. By using approximation methods such as Tree Methods, Particle Mesh Methods etc computational complexity can be reduced to $𝑂(𝑁 \log{N})$, with a loss of accuracy. Instead of reducing the complexity via algorithmic changes, project intend to use parallelization techniques to optimize the original algorithm with complexity of $𝑂(𝑁^2)$ for faster execution.

#### Libraries Used

* OpenMP - Shared Memory  
* MPI - Distributed Memory  
* CUDA - GPU Utilization for Parallelization
* SDL3 - Visualization of the Simulation

### Dev Instructions
Windows Subsystem for Linux (WSL2) environment is used as the development environment.  

`Makefile` defines all the build instrcutions.  
To build all the executables use the following command  
```bash
make
``` 
To build a specific executable use the following commands
```Bash
make sequential
make openmp

#  Not implemented yet
make mpi
make cuda
```
`run` Bash script contains all the necessary information to run the project.
```bash
./run seq # Run the sequential executable
./run omp # Run the openmp executable

# NOTE - For CUDA, MPI methods still not implemeted yet
```
To clean the project - Remove the executables and Object files ue the following command. 
```bash
make clean
```