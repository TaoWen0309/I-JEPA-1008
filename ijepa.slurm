#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_001-2023fa-15

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24

#SBATCH --time=2:00:00
#SBATCH --job-name=ijepa
#SBATCH --output=ijepa.out
#SBATCH --error=ijepa.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

singularity exec --nv \
--bind /scratch \
--overlay /scratch/tw2672/pytorch/torch2cuda8.ext3:ro \
/scratch/tw2672/pytorch/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;python main_distributed.py --fname configs/in1k_vith14_ep300.yaml \
--folder /scratch/tw2672/ijepa \
--partition n1s8-v100-1 \
--nodes 1 \
--tasks-per-node 1 \
--time 1000
"
