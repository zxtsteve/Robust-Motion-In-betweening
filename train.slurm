#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --qos=long-high-prio 
#SBATCH --time=6-23:59:59
#SBATCH -p res-gpu-small
#SBATCH --mem=28g #最大28
# --output=train_repre7a_551551_auto10.out

#* not A100
#SBATCH --gres=gpu # request a single GPU
# --nodelist=gpu[2]

#* A100
# --gres=gpu:ampere:1 # request a single A100
# --exclude=gpu[0-8]

path_to_your_repo='/home2/sgtn88/Robust-Motion-In-betweening'
export PYTHONPATH=$PYTHONPATH:$path_to_your_repo
cd $path_to_your_repo


python train.py