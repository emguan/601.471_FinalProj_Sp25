#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name="midway_run"
#SBATCH --output=jesusplease.out
#SBATCH --mem=16G
#SBATCH --nodelist=gpuz01

module load cuda/12.1
module load anaconda

source activate new_env

python test.py
