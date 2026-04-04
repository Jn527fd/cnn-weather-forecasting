#!/bin/bash
#SBATCH --job-name=part2_saliency
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/part2_%j.out
#SBATCH --error=logs/part2_%j.err

module load class/default
module load cs137/2026spring

cd /cluster/tufts/c26sp1cs0137/jnaran01/cnn-weather-forecasting

export MODEL_PATH="/cluster/tufts/c26sp1cs0137/jnaran01/cnn-weather-forecasting/best_model_20260329_190855.pt"
export NORM_PATH="/cluster/tufts/c26sp1cs0137/jnaran01/cnn-weather-forecasting/norm_stats.pt"

python saliency_complete.py
