#!/bin/bash -l
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=8:30:00
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH -o logs/%j.log

conda activate myenv

srun ./run_train.sh $num_samples $model_width $temperature

