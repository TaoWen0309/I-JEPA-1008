#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_001-2023fa-15

## change the partition number to use different number of GPUs
#SBATCH --partition=c2s60p
#SBATCH --cpus-per-task=24

#SBATCH --time=4:00:00
#SBATCH --job-name=unzip

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

singularity exec --nv \
--bind /scratch \
--overlay /scratch/tw2672/pytorch/torch2cuda8.ext3:ro \
/scratch/tw2672/pytorch/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;python unzip.py
"
