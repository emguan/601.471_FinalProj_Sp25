#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="midway_run"
#SBATCH --output=midway_results_128.out
#SBATCH --mem=16G
#SBATCH --nodelist=gpuz01


module load cuda/12.1
python test.py

