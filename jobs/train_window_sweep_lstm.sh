#!/bin/bash
#SBATCH --job-name=specom_lstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/specom_lstm_%j.log
#SBATCH --error=logs/specom_lstm_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

module purge
module load AI_env/v1

echo "Python:"
which python3
python3 --version

python3 -u scripts/01_check_environment.py
python3 -u scripts/06_train_window_sweep_lstm.py

echo "Job finished: $(date)"
