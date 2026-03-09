########################################################################
############################ Compiler ##################################
########################################################################

CC = gcc

# Base compile flags
CFLAGS = -std=c11 -Wall -O3 -march=native -Iinclude

# Libraries
MATH_LIB = -lm
OPENMP = -fopenmp

# Future libraries (disabled for now)
MPI =
CUDA =
SDL =

########################################################################
######################## Project Settings ##############################
########################################################################

SRCDIR = src
OBJDIR = obj
EXEDIR = exe

SEQ_TARGET = $(EXEDIR)/nbody_sequential
OMP_TARGET = $(EXEDIR)/nbody_openmp

########################################################################
######################## Source Files ##################################
########################################################################

SEQ_SRC = $(SRCDIR)/nbody_sequential.c
OMP_SRC = $(SRCDIR)/nbody_openmp.c

SEQ_OBJ = $(OBJDIR)/nbody_sequential.o
OMP_OBJ = $(OBJDIR)/nbody_openmp.o

########################################################################
######################## Build Targets #################################
########################################################################

all: dirs sequential openmp

dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(EXEDIR)

########################################################################
# Sequential Build
########################################################################

sequential: $(SEQ_TARGET)

$(SEQ_TARGET): $(SEQ_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(MATH_LIB)

$(SEQ_OBJ): $(SEQ_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

########################################################################
# OpenMP Build
########################################################################

openmp: $(OMP_TARGET)

$(OMP_TARGET): $(OMP_OBJ)
	$(CC) $(CFLAGS) $(OPENMP) -o $@ $^ $(MATH_LIB)

$(OMP_OBJ): $(OMP_SRC)
	$(CC) $(CFLAGS) $(OPENMP) -c $< -o $@

########################################################################
######################## Cleaning ######################################
########################################################################

.PHONY: clean

clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(EXEDIR)/*

########################################################################
######################## Run Commands ##################################
########################################################################

.PHONY: run-seq run-omp

# Run sequential executable
run-seq: sequential
	@echo "Running sequential executable..."
	@./$(SEQ_TARGET)

# Run OpenMP executable
run-omp: openmp
	@echo "Running OpenMP executable..."
	@./$(OMP_TARGET)