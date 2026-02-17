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

#TSNE_POINTS=(3000 5000 8000)
#TSNE_PERP=(20 30 40)
#UMAP_NEIGH=(5 15 30)
#UMAP_DIST=(0.1 0.5)

shopt -s nullglob
for rd in runs/20260212_*
do
  [ -d "$rd" ] || continue
  base=$(basename "$rd")
  [[ "$base" < "20260212_170000" ]] && continue


  echo "=== PLOTTING $rd ==="
  python -m src.plot_from_run \
    --run_dir "$rd" \
    --do_tsne --do_umap --dims both \
    --tsne_points 5000 --tsne_perplexity 30 \
    --umap_neighbors 15 --umap_min_dist 0.5 \
    --only_single \
    --per_class 200
done

echo "[OK] Plot-Job abgeschlossen."
