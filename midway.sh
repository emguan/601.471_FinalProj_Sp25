#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="sliding_test"
#SBATCH --output=sliding.out
#SBATCH --mem=16G
#SBATCH --nodelist=gpuz01

source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh

module load anaconda
conda activate ssm_project

module load cuda/12.1

python test.py

