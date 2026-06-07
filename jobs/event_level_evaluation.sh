#!/bin/bash
#SBATCH --job-name=event_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/event_eval_%j.log
#SBATCH --error=logs/event_eval_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "Event-level evaluation started: $(date)"

module purge
module load AI_env/v1

python3 -u scripts/07_event_level_evaluation.py

echo "Event-level evaluation finished: $(date)"
