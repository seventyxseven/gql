#!/bin/bash
#SBATCH --ntasks-per-node=24 
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1 
#SBATCH --mem-per-cpu=4000 
 
module load mpi/openmpi-x86_64
source "${HOME}/GQL/dedalus/bin/activate"
 
mpirun -n 24 "${PWD}/GQL_all.py"
