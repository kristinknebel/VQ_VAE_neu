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
  runs/20260212_112308_091_p4087741__L16_N256_C0_1_LR5e-04_A32 \
  runs/20260212_112310_058_p4017839__L16_N256_C0_1_LR1e-04_A32 \
  runs/20260212_112312_524_p3261039__L16_N256_C0_5_LR5e-04_A32 \
  runs/20260212_112313_928_p3757862__L16_N256_C0_25_LR5e-04_A32 \
  runs/20260212_112315_366_p3742951__L16_N256_C0_25_LR1e-04_A32 \
  runs/20260212_112317_152_p2968807__L16_N256_C0_5_LR1e-04_A32 \
  runs/20260212_112318_602_p2989747__L16_N512_C0_1_LR5e-04_A32 \
  runs/20260212_112320_022_p3203705__L16_N512_C0_1_LR1e-04_A32 \
  runs/20260212_112928_642_p2855931__L32_N512_C0_1_LR1e-04_A32 \
  runs/20260212_113031_862_p4092753__L32_N512_C0_5_LR5e-04_A32 \
  runs/20260212_113031_864_p4022834__L32_N512_C0_5_LR1e-04_A32 \
  runs/20260212_113031_901_p3266011__L32_N512_C0_25_LR5e-04_A32 \


do
  echo "=== PLOTTING $rd ==="
  python -m src.plot_from_run \
    --run_dir "$rd" \
    --do_tsne --do_umap --dims both \
    --tsne_points 5000 --tsne_perplexity 30 \
    --only_single \
    --whitelist "NORM,AFIB,AFLT,IMI,ASMI,AMI,ISC_,ISCAL,ISCIN,ISCIL,LVH,IVCD,IRBBB" \
    --per_class 200
done


echo "--------------------------------------"
echo "Plot-Job abgeschlossen."
echo "--------------------------------------"
