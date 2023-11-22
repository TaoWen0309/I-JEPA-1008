#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_001-2023fa-15

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

#SBATCH --time=2:00:00
#SBATCH --job-name=unzip
#SBATCH --output=unzip.out
#SBATCH --error=unzip.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

singularity exec --nv \
--bind /scratch \
--overlay /scratch/tw2672/pytorch/torch2cuda8.ext3:ro \
/scratch/tw2672/pytorch/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;python unzip.py
"
