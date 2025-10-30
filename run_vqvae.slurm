#!/bin/bash
#SBATCH --job-name=vqvae_exp
#SBATCH --output=logs/vqvae_%A_%a.out
#SBATCH --error=logs/vqvae_%A_%a.err
#SBATCH --array=0-23%6        # 24 Jobs, max. 6 gleichzeitig
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short 

# === Umgebung vorbereiten ===
conda activate tf_env                # Conda-Environment aktivieren

# === Ins Projektverzeichnis wechseln ===
cd ~/masterarbeit/VQ_VAE_neu
mkdir -p logs                         # Log-Ordner sicherstellen

# === Range-Logik ===
# 24 Experimente insgesamt → 6 pro Task
START=$((SLURM_ARRAY_TASK_ID * 6))
END=$((START + 5))

echo "--------------------------------------"
echo "Starte Experimente ${START}-${END}"
echo "Hostname: $(hostname)"
echo "SLURM Job-ID: ${SLURM_JOB_ID}, Array-Task: ${SLURM_ARRAY_TASK_ID}"
echo "--------------------------------------"

# === Python-Run ===
python -m src.main --range ${START}-${END}

echo "--------------------------------------"
echo "Job ${SLURM_ARRAY_TASK_ID} abgeschlossen."
echo "--------------------------------------"
