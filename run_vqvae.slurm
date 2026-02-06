#!/bin/bash
#SBATCH --job-name=vqvae_exp
#SBATCH --output=/home/$USER/masterarbeit/VQ_VAE_0602/logs/vqvae_%A_%a.out
#SBATCH --error=/home/$USER/masterarbeit/VQ_VAE_0602/logs/vqvae_%A_%a.err
#SBATCH --array=0-3%4
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short

set -euo pipefail

# === Ins Projektverzeichnis wechseln ===
cd ~/masterarbeit/VQ_VAE_0602
mkdir -p logs

# === Conda aktivieren ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_env

# === Range-Logik ===
START=$((SLURM_ARRAY_TASK_ID * 6))
END=$((START + 5))

echo "--------------------------------------"
echo "Starte Experimente ${START}-${END}"
echo "Hostname: $(hostname)"
echo "SLURM Job-ID: ${SLURM_JOB_ID}, Array-Task: ${SLURM_ARRAY_TASK_ID}"
echo "--------------------------------------"

python -m src.main --range ${START}-${END}

echo "--------------------------------------"
echo "Job ${SLURM_ARRAY_TASK_ID} abgeschlossen."
echo "--------------------------------------"
