#!/bin/bash
#SBATCH --job-name=saliency
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/saliency_%j.out
#SBATCH --error=logs/saliency_%j.err

module load class/default
module load cs137/2026spring

cd /cluster/tufts/c26sp1cs0137/jnaran01/cnn-weather-forecasting

export MODEL_PATH="/cluster/tufts/c26sp1cs0137/jnaran01/cnn-weather-forecasting/best_model_20260324_210752.pt"
export NORM_PATH="/cluster/tufts/c26sp1cs0137/jnaran01/cnn-weather-forecasting/norm_stats.pt"

python compute_saliency_new.py