#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_001-2023fa-15

## change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

#SBATCH --partition=n1s16-v100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24

#SBATCH --time=2:00:00
#SBATCH --job-name=ijepa
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/tw2672/tmp_images /tmp
cp -rp /scratch/tw2672/pytorch /tmp
echo "tmp created"

singularity exec --nv \
--bind /scratch \
--overlay /tmp/torch2cuda8.ext3:ro \
/scratch/tw2672/pytorch/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;python main.py --fname configs/in1k_vith14_ep300.yaml
"
