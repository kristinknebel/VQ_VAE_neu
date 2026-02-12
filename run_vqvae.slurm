#!/bin/bash
#SBATCH --job-name=vqvae_exp
#SBATCH --chdir=/users/stud/knebelkr/masterarbeit/VQ_VAE_0602
#SBATCH --output=logs/vqvae_%A_%a.out
#SBATCH --error=logs/vqvae_%A_%a.err
#SBATCH --array=0-23%4
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --mail-user=knebelkr@hu-berlin.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail


# === Conda aktivieren ===
source /opt/conda/etc/profile.d/conda.sh
conda activate tf_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# === Ins Projektverzeichnis wechseln ===
cd /users/stud/knebelkr/masterarbeit/VQ_VAE_0602


START=$SLURM_ARRAY_TASK_ID
END=$SLURM_ARRAY_TASK_ID

echo "--------------------------------------"
echo "Starte Experimente ${START}-${END}"
echo "Hostname: $(hostname)"
echo "SLURM Job-ID: ${SLURM_JOB_ID}, Array-Task: ${SLURM_ARRAY_TASK_ID}"
echo "--------------------------------------"

python -m src.main --range ${START}-${END}

echo "--------------------------------------"
echo "Job ${SLURM_ARRAY_TASK_ID} abgeschlossen."
echo "--------------------------------------"
