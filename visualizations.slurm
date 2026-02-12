#!/bin/bash
#SBATCH --job-name=vqvae_plots
#SBATCH --chdir=/users/stud/knebelkr/masterarbeit/VQ_VAE_0602
#SBATCH --output=logs/plots_%A.out
#SBATCH --error=logs/plots_%A.err
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
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

for rd in \
  runs/20260212_170739_991_p2956471__L16_N256_C0_25_LR5e-04_A32 \


do
  echo "=== PLOTTING $rd ==="
  python -m src.plot_from_run \
    --run_dir "$rd" \
    --do_tsne --do_umap --dims both \
    --tsne_points 5000 --tsne_perplexity 30 \
    --only_single \
    #--whitelist "NORM,AFIB,AFLT,IMI,ASMI,AMI,ISC_,ISCAL,ISCIN,ISCIL,LVH,IVCD,IRBBB" \
    --per_class 200
done


echo "--------------------------------------"
echo "Plot-Job abgeschlossen."
echo "--------------------------------------"
