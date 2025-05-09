#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="overly_fit"
#SBATCH --output=overly_fit.out
#SBATCH --mem=16G
#SBATCH --nodelist=gpuz01


module load cuda/12.1
python overly_fit_demo.py

