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

# Parameter-Grid
TSNE_POINTS=(3000 5000 8000)
TSNE_PERP=(20 30 40)

UMAP_NEIGH=(5 15 30)
UMAP_DIST=(0.1 0.5)

for rd in \
  runs/20260212_170739_991_p2956471__L16_N256_C0_25_LR5e-04_A32 \
  runs/20260212_195934_057_p2994000__L32_N128_C0_5_LR5e-04_A32 \

do

  [ -d "$rd" ] || continue
  echo "=== RUN $rd ==="

  for pts in "${TSNE_POINTS[@]}"; do
    for perp in "${TSNE_PERP[@]}"; do
      for nn in "${UMAP_NEIGH[@]}"; do
        for md in "${UMAP_DIST[@]}"; do

          echo "t-SNE pts=$pts perp=$perp | UMAP nn=$nn md=$md"

          python -m src.plot_from_run \
            --run_dir "$rd" \
            --do_tsne --do_umap --dims both \
            --tsne_points "$pts" \
            --tsne_perplexity "$perp" \
            --umap_neighbors "$nn" \
            --umap_min_dist "$md" \
            --only_single \
            --whitelist "NORM,AFIB,AFLT,IMI,ASMI,AMI,ISC_,ISCAL,ISCIN,ISCIL,LVH,IVCD,IRBBB" \
            --per_class 200

        done
      done
    done
  done


#--whitelist "NORM,AFIB,AFLT,IMI,ASMI,AMI,ISC_,ISCAL,ISCIN,ISCIL,LVH,IVCD,IRBBB" \
#--tsne_points 5000 = Downsampling für Rechenzeit + Lesbarkeit
#--tsne_perplexity 30 = typische Neighborhood-Größe für stabile Cluster
#--umap_neighbors 15 (stadard)
#--umap_min_dist 0.1 (standard)
echo "--------------------------------------"
echo "Plot-Job abgeschlossen."
echo "--------------------------------------"
