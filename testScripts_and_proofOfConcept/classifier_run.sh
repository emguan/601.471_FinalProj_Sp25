#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="classifier_run"
#SBATCH --output=adding_padding.out
#SBATCH --mem=16G
#SBATCH --nodelist=gpuz01


module load cuda/12.1
python classifier_exploration.py
