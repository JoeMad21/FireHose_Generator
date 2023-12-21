#!/bin/bash
#SBATCH --job-name FireHose_Generator
#SBATCH --ipus=1
--partition=p64
#SBATCH --nodelist=gc-poplar-03
#SBATCH --ntasks 1
#SBATCH --time=00:05:00

srun ./gen_demo --device 1 --con_task 1 --source 1 --dimension 6