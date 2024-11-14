#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00:00:30
#SBATCH -A naiss2024-22-531
#SBATCH --nodes=1
#SBATCH -p main
#SBATCH --ntasks-per-node=8
#SBATCH -e solver.e
#SBATCH -o solver.o


# CC main.cpp -Wall -O3 -o solver.out
srun ./solver.out 2 2 2