# src/plot_from_run.py
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# Projekt-Imports
from src.snippet_cache import get_cache_paths, load_cached_snippets
from src.models import VQVAE
from src.visualization import plot_ecg_reconstructions


# -----------------------------
# Helpers
# -----------------------------

def snippet_fingerprint(x: np.ndarray) -> str:
    xb = np.asarray(x, dtype=np.float32).tobytes(order="C")
    return hashlib.sha256(xb).hexdigest()[:16]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_param(params: dict, keys: List[str], cast=None, required: bool = True, default=None):
    for k in keys:
        if k in params and params[k] is not None:
            return cast(params[k]) if cast else params[k]
    if required:
        raise KeyError(f"Keiner der Keys gefunden: {keys} in params.json")
    return default


def recover_test_indices_from_cache(snippets: np.ndarray, test_fps: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Rekonstruiert Indizes in 'snippets' für die Fingerprints aus test_fps.
    Rückgabe: (indices, missing_fps)
    """
    fp_to_indices: Dict[str, List[int]] = {}
    for i in range(len(snippets)):
        fp = snippet_fingerprint(snippets[i])
        fp_to_indices.setdefault(fp, []).append(i)

    indices: List[int] = []
    missing: List[str] = []
    used: Dict[str, int] = {}  # fp -> wie viele bereits verwendet (für Duplikate)

    for fp in test_fps.tolist():
        lst = fp_to_indices.get(fp)
        if not lst:
            missing.append(fp)
            continue
        k = used.get(fp, 0)
        if k >= len(lst):
            missing.append(fp)
            continue
        indices.append(lst[k])
        used[fp] = k + 1

    return np.asarray(indices, dtype=np.int64), missing


# -----------------------------
# Plot-Funktionen
# -----------------------------

def plot_training_history(run_dir: Path, out_dir: Path) -> None:
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        print(f"[WARN] history.json nicht gefunden in {run_dir}")
        return

    hist = load_json(hist_path)
    if "loss" not in hist:
        print(f"[WARN] history.json hat kein 'loss' Feld: {hist_path}")
        return

    epochs = np.arange(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, hist["loss"], label="loss")
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    out = out_dir / "training_history.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Training-Verlauf gespeichert: {out}")


def plot_codebook_usage(run_dir: Path, out_dir: Path) -> None:
    counts_path = run_dir / "codebook_counts.npy"
    idx_path = run_dir / "codebook_used_indices.npy"
    usage_json = run_dir / "codebook_usage.json"

    unique_indices = None
    counts = None
    num_embeddings = None

    if counts_path.exists() and idx_path.exists():
        counts = np.load(counts_path)
        unique_indices = np.load(idx_path)
        # num_embeddings evtl. aus usage_json
        if usage_json.exists():
            stats = load_json(usage_json)
            num_embeddings = stats.get("num_embeddings")
    elif usage_json.exists():
        stats = load_json(usage_json)
        unique_indices = np.asarray(stats.get("unique_indices", []))
        counts = np.asarray(stats.get("counts", []))
        num_embeddings = stats.get("num_embeddings")
    else:
        print(f"[WARN] Keine Codebook-Usage Dateien gefunden in {run_dir}")
        return

    if unique_indices is None or counts is None or len(unique_indices) == 0:
        print(f"[WARN] Codebook-Usage ist leer in {run_dir}")
        return

    plt.figure(figsize=(10, 4))
    plt.bar(unique_indices, counts)
    title = "Codebook Usage"
    if num_embeddings is not None:
        title += f" ({len(unique_indices)}/{num_embeddings})"
    plt.title(title)
    plt.xlabel("Codebook Index")
    plt.ylabel("Count")
    plt.tight_layout()
    out = out_dir / "codebook_usage.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Codebook-Usage Plot gespeichert: {out}")


def plot_latent_tsne(run_dir: Path, out_dir: Path, max_points: int = 5000, perplexity: int = 30, random_state: int = 42) -> None:
    emb_path = run_dir / "snippet_embeddings.npy"
    lab_path = run_dir / "snippet_labels.json"

    if not emb_path.exists() or not lab_path.exists():
        print(f"[WARN] snippet_embeddings.npy oder snippet_labels.json fehlen in {run_dir}")
        return

    X = np.load(emb_path)  # (N, latent_dim)
    labels = np.asarray(load_json(lab_path))

    if len(X) == 0:
        print("[WARN] snippet_embeddings ist leer.")
        return

    n = len(X)
    if n > max_points:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(n, size=max_points, replace=False)
        Xs = X[sel]
        ls = labels[sel]
    else:
        Xs = X
        ls = labels

    # t-SNE 2D
    eff_perp = min(perplexity, max(5, (len(Xs) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=eff_perp,
        init="pca",
        learning_rate="auto",
        random_state=random_state
    )
    Y = tsne.fit_transform(Xs)

    # Labels -> Top-10 Klassen, Rest "OTHER"
    uniq, cnt = np.unique(ls, return_counts=True)
    top = uniq[np.argsort(-cnt)[:10]].tolist()
    top_set = set(top)
    ls_plot = np.array([lab if lab in top_set else "OTHER" for lab in ls], dtype=object)

    plt.figure(figsize=(8, 6))
    for lab in np.unique(ls_plot):
        m = (ls_plot == lab)
        plt.scatter(Y[m, 0], Y[m, 1], s=8, alpha=0.7, label=str(lab))
    plt.title(f"Latent t-SNE (n={len(Xs)}, perp={eff_perp})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2, fontsize="small", loc="best")
    plt.tight_layout()
    out = out_dir / "latent_tsne.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Latent t-SNE Plot gespeichert: {out}")


def plot_reconstructions_from_cache(run_dir: Path, out_dir: Path, n_examples: int = 10, seed: int = 42) -> None:
    cache_ref_path = run_dir / "cache_ref.json"
    test_fp_path = run_dir / "test_fingerprints.npy"
    weights_path = run_dir / "vqvae_final.weights.h5"
    params_path = run_dir / "params.json"

    for p in [cache_ref_path, test_fp_path, weights_path, params_path]:
        if not p.exists():
            print(f"[WARN] {p.name} fehlt in {run_dir} – Rekonstruktionen nicht möglich.")
            return

    cache_ref = load_json(cache_ref_path)
    cache_key = cache_ref["cache_key"]
    cache_root = Path(cache_ref["cache_root"])

    # Cache laden
    paths = get_cache_paths(cache_root, cache_key)
    if not paths["npz"].exists():
        print(f"[WARN] Cache-Datei fehlt: {paths['npz']}")
        return

    snippets, ecg_ids, labels = load_cached_snippets(paths["npz"])
    test_fps = np.load(test_fp_path)

    test_indices, missing = recover_test_indices_from_cache(snippets, test_fps)
    if len(missing) > 0:
        print(f"[WARN] {len(missing)} Test-Fingerprints nicht gematcht (erste 5): {missing[:5]}")
    if len(test_indices) == 0:
        print("[WARN] Keine Test-Indizes rekonstruiert – kann keine Rekonstruktionen plotten.")
        return

    # Subsample für Plot
    rng = np.random.default_rng(seed)
    m = min(n_examples, len(test_indices))
    sel = rng.choice(len(test_indices), size=m, replace=False)
    x = snippets[test_indices[sel]]  # (m, T, C)

    # Params laden (dein experiment.py speichert seed + params)
    run_params = load_json(params_path)

    latent_dim = get_param(run_params, ["LATENT_DIMENSIONS", "LATENT_DIM", "latent_dim"], cast=int)
    num_embeddings = get_param(run_params, ["NUM_EMBEDDINGS", "num_embeddings"], cast=int)
    commitment_cost = get_param(run_params, ["COMMITMENT_COST", "commitment_cost"], cast=float)

    input_shape = (x.shape[1], x.shape[2])

    model = VQVAE(
        input_shape=input_shape,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost
    )

    # Build (Subclassed model needs this before load_weights)
    _ = model(np.zeros((1,) + input_shape, dtype=np.float32))
    model.load_weights(weights_path)

    y = model.predict(x, verbose=0)

    out = out_dir / "reconstructions.png"
    plot_ecg_reconstructions(x, y, num_examples=m, filename=str(out))
    print(f"[OK] Rekonstruktionen gespeichert: {out}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Pfad zu einem run-Ordner (runs/<...>)")
    ap.add_argument("--out_dir", type=str, default=None, help="Output-Ordner für Plots (Default: <run_dir>/plots)")
    ap.add_argument("--n_recon", type=int, default=10, help="Anzahl Rekonstruktions-Beispiele")
    ap.add_argument("--tsne_points", type=int, default=5000, help="Max. Punkte für t-SNE")
    ap.add_argument("--tsne_perplexity", type=int, default=30, help="t-SNE Perplexity")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Run: {run_dir}")
    print(f"[INFO] Output: {out_dir}")

    plot_training_history(run_dir, out_dir)
    plot_codebook_usage(run_dir, out_dir)
    plot_latent_tsne(run_dir, out_dir, max_points=args.tsne_points, perplexity=args.tsne_perplexity)
    plot_reconstructions_from_cache(run_dir, out_dir, n_examples=args.n_recon)


if __name__ == "__main__":
    main()
