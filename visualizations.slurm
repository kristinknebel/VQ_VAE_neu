#!/bin/bash
#SBATCH --job-name=vqvae_plots
#SBATCH --chdir=/users/stud/knebelkr/masterarbeit/VQ_VAE_0602
#SBATCH --output=logs/plots_%A.out
#SBATCH --error=logs/plots_%A.err
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --mail-user=knebelkr@hu-berlin.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Logs-Ordner muss existieren, bevor Slurm schreibt
mkdir -p logs

# === Conda aktivieren ===
source /opt/conda/etc/profile.d/conda.sh
conda activate tf_env

# Für "python -m src...." / Imports aus src
export PYTHONPATH="$PWD"

echo "--------------------------------------"
echo "Starte Plot-Job"
echo "Hostname: $(hostname)"
echo "SLURM Job-ID: ${SLURM_JOB_ID}"
echo "Workdir: $(pwd)"
echo "Python: $(which python)"
echo "--------------------------------------"

# Optional: sanity check, dass runs existiert
# ls -lah runs | head -n 50

# === Plots erzeugen ===
# Passe bei Bedarf Parameter an (n_recon/tsne_points/only_latest/Filter)
python -m src.plot_all_runs \
  --runs_dir runs \
  --out_subdir plots \
  --n_recon 10 \
  --tsne_points 5000 \
  --tsne_perplexity 30

echo "--------------------------------------"
echo "Plot-Job abgeschlossen."
echo "--------------------------------------"
