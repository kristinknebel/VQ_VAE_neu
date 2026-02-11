# src/plot_all_runs.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

from src.plot_from_run import (
    plot_training_history,
    plot_codebook_usage,
    plot_latent_embeddings,   # <-- statt plot_latent_tsne
    plot_reconstructions_from_cache,
)

def is_run_dir(p: Path) -> bool:
    # Euer experiment.py erzeugt Ordner wie: YYYYMMDD_HHMMSS__...
    return p.is_dir() and re.match(r"^\d{8}_\d{6}__", p.name) is not None and (p / "params.json").exists()

def load_params(run_dir: Path) -> dict:
    return json.loads((run_dir / "params.json").read_text(encoding="utf-8"))

def run_matches_filters(run_dir: Path, want_latent_dim: Optional[int], want_num_emb: Optional[int],
                        want_beta: Optional[float], want_lr: Optional[float]) -> bool:
    p = load_params(run_dir)
    if want_latent_dim is not None and int(p.get("LATENT_DIMENSIONS", -1)) != want_latent_dim:
        return False
    if want_num_emb is not None and int(p.get("NUM_EMBEDDINGS", -1)) != want_num_emb:
        return False
    if want_beta is not None and float(p.get("COMMITMENT_COST", -999.0)) != want_beta:
        return False
    if want_lr is not None and float(p.get("LEARNING_RATE", -999.0)) != want_lr:
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs", help="Basisordner mit Run-Verzeichnissen")
    ap.add_argument("--out_subdir", type=str, default="plots", help="Unterordner pro Run für Plot-Ausgaben")
    ap.add_argument("--n_recon", type=int, default=10, help="Anzahl Rekonstruktions-Beispiele")
    ap.add_argument("--tsne_points", type=int, default=5000, help="Max. Punkte für t-SNE")
    ap.add_argument("--tsne_perplexity", type=int, default=30, help="t-SNE Perplexity")
    ap.add_argument("--only_latest", action="store_true", help="Nur den neuesten Run plotten")
    ap.add_argument("--latent_dim", type=int, default=None, help="Filter: LATENT_DIMENSIONS")
    ap.add_argument("--num_embeddings", type=int, default=None, help="Filter: NUM_EMBEDDINGS")
    ap.add_argument("--commitment_cost", type=float, default=None, help="Filter: COMMITMENT_COST")
    ap.add_argument("--learning_rate", type=float, default=None, help="Filter: LEARNING_RATE")
    ap.add_argument("--only_single", action="store_true")
    ap.add_argument("--whitelist", type=str, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--per_class", type=int, default=None)
    ap.add_argument("--collapse_other_top", type=int, default=None)
    
    ap.add_argument("--do_tsne", action="store_true")
    ap.add_argument("--do_umap", action="store_true")
    ap.add_argument("--dims", type=str, default="both", choices=["2d","3d","both"])
    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    args = ap.parse_args()

    runs_root = Path(args.runs_dir)
    if not runs_root.exists():
        raise FileNotFoundError(runs_root)

    run_dirs: List[Path] = sorted([p for p in runs_root.iterdir() if is_run_dir(p)], key=lambda p: p.name)
    if not run_dirs:
        print(f"[WARN] Keine Run-Verzeichnisse in {runs_root} gefunden.")
        return

    # Filter anwenden
    run_dirs = [rd for rd in run_dirs if run_matches_filters(
        rd, args.latent_dim, args.num_embeddings, args.commitment_cost, args.learning_rate
    )]

    if not run_dirs:
        print("[WARN] Keine Runs passen auf die Filter.")
        return

    if args.only_latest:
        run_dirs = [run_dirs[-1]]

    print(f"[INFO] Anzahl Runs zum Plotten: {len(run_dirs)}")

    for rd in run_dirs:
        out_dir = rd / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[RUN] {rd.name}")
        try:
            plot_training_history(rd, out_dir)
            plot_codebook_usage(rd, out_dir)
            plot_latent_embeddings(
                rd, out_dir,
                max_points=args.tsne_points,
                tsne_perplexity=args.tsne_perplexity,
                random_state=42,
                do_tsne=args.do_tsne,
                do_umap=args.do_umap,
                dims=args.dims,
                only_single=args.only_single,
                whitelist=args.whitelist,
                top_k=args.top_k,
                per_class=args.per_class,
                collapse_other_top=args.collapse_other_top,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
            )
            plot_reconstructions_from_cache(rd, out_dir, n_examples=args.n_recon)
        except Exception as e:
            print(f"[ERROR] Plot fehlgeschlagen für {rd}: {e}")

    print("\n[OK] Fertig.")

if __name__ == "__main__":
    main()
