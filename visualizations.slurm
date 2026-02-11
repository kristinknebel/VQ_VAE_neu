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
  runs/20260211_181943_289_p3179163__L16_N256_C0_1_LR1e-04_A32 \
  runs/20260211_181944_025_p3932576__L16_N256_C0_1_LR5e-04_A32 \
  runs/20260211_181945_214_p2886853__L16_N256_C0_25_LR1e-04_A32 \
  runs/20260211_181946_154_p3674502__L16_N256_C0_5_LR1e-04_A32 \
  runs/20260211_181947_886_p4003211__L16_N256_C0_25_LR5e-04_A32 \
  runs/20260211_182240_780_p2912246__L16_N256_C0_5_LR5e-04_A32 \
  runs/20260211_182506_864_p3936736__L16_N512_C0_1_LR1e-04_A32 \
  runs/20260211_182506_881_p3679448__L16_N512_C0_1_LR5e-04_A32 \
  runs/20260211_182506_993_p3183307__L16_N512_C0_25_LR1e-04_A32 \
  runs/20260211_182507_357_p4008167__L16_N512_C0_25_LR5e-04_A32 \
  runs/20260211_182507_376_p2890976__L16_N512_C0_5_LR1e-04_A32 \
  runs/20260211_182829_836_p2916357__L16_N512_C0_5_LR5e-04_A32 \
  runs/20260211_183046_060_p3188243__L32_N256_C0_1_LR1e-04_A32 \
  runs/20260211_183046_704_p2895988__L32_N256_C0_1_LR5e-04_A32 \
  runs/20260211_183049_896_p4012297__L32_N256_C0_25_LR1e-04_A32 \
  runs/20260211_183052_843_p3683584__L32_N256_C0_5_LR1e-04_A32 \
  runs/20260211_183053_041_p3941689__L32_N256_C0_25_LR5e-04_A32 \
  runs/20260211_183356_083_p2921293__L32_N256_C0_5_LR5e-04_A32 \
  runs/20260211_183615_042_p3687683__L32_N512_C0_25_LR5e-04_A32 \
  runs/20260211_183615_044_p2900097__L32_N512_C0_1_LR5e-04_A32 \
  runs/20260211_183615_076_p3192345__L32_N512_C0_1_LR1e-04_A32 \
  runs/20260211_183615_094_p4017239__L32_N512_C0_25_LR1e-04_A32 \
  runs/20260211_183620_715_p3945789__L32_N512_C0_5_LR1e-04_A32 \
  runs/20260211_183857_084_p3135149__L32_N512_C0_5_LR5e-04_A32
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
