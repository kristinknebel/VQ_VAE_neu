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
mkdir -p logs

source /opt/conda/etc/profile.d/conda.sh
conda activate tf_env
export PYTHONPATH="$PWD"

TSNE_POINTS=(3000 5000 8000)
TSNE_PERP=(20 30 40)
UMAP_NEIGH=(5 15 30)
UMAP_DIST=(0.1 0.5)

for rd in \
  runs/20260212_170739_991_p2956471__L16_N256_C0_25_LR5e-04_A32 \
  runs/20260212_195934_057_p2994000__L32_N128_C0_5_LR5e-04_A32
do
  [ -d "$rd" ] || { echo "[WARN] missing: $rd"; continue; }

  for pts in "${TSNE_POINTS[@]}"; do
    for perp in "${TSNE_PERP[@]}"; do
      for nn in "${UMAP_NEIGH[@]}"; do
        for md in "${UMAP_DIST[@]}"; do

          OUT_DIR="$rd/plots_grid/tsne_pts${pts}_perp${perp}__umap_nn${nn}_md${md}"
          mkdir -p "$OUT_DIR"

          echo "=== $rd | pts=$pts perp=$perp nn=$nn md=$md ==="

          python -m src.plot_from_run \
            --run_dir "$rd" \
            --out_dir "$OUT_DIR" \
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
done

echo "[OK] Plot-Job abgeschlossen."
