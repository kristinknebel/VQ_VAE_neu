# src/main.py
import sys
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # wichtig auf Servern ohne Display
import matplotlib.pyplot as plt

from . import config
from .experiment import start_run

# Daten & Preprocessing
from .data_loader import load_scp_codes, get_scp_code_list
from .preprocessing import create_snippets, batch_process_relevant_ecgs

# Modell & Training
from .models import VQVAE, train_and_evaluate_vqvae

# Visualisierung & Analyse
from .visualization import (
    visualize_latent_space,
    plot_ecg_reconstructions,
    visualize_codebook_embeddings,
    analyze_codebook_usage,
)

from pathlib import Path
from src.snippet_cache import (
    make_cache_key, get_cache_paths, try_load_or_build, compute_data_version
)

# ---------------------------------------------------------------------
# 1) Run-Parameter erfassen & Run starten
# ---------------------------------------------------------------------
params = {
    "LATENT_DIMENSIONS_TO_TEST": getattr(config, "LATENT_DIMENSIONS_TO_TEST", None),
    "NUM_EMBEDDINGS": getattr(config, "NUM_EMBEDDINGS", None),
    "COMMITMENT_COST": getattr(config, "COMMITMENT_COST", None),
    "LEARNING_RATE": getattr(config, "LEARNING_RATE", None),
    "AE_BATCH_SIZE": getattr(config, "AE_BATCH_SIZE", None),
    "AE_EPOCHS": getattr(config, "AE_EPOCHS", None),
    "SAMPLING_RATE": getattr(config, "SAMPLING_RATE", None),
    "SNIPPET_LENGTH_BEFORE_R": getattr(config, "SNIPPET_LENGTH_BEFORE_R", None),
    "SNIPPET_LENGTH_AFTER_R": getattr(config, "SNIPPET_LENGTH_AFTER_R", None),
    "VIS_N_COMPONENTS": getattr(config, "VIS_N_COMPONENTS", None),
    "VIS_METHOD": getattr(config, "VIS_METHOD", None),
    "DATA_DIR": str(getattr(config, "DATA_DIR", "")),
}

run = start_run(params, base_dir="runs", seed=42)
print(f" Neuer Trainingslauf: {run.run_dir}")

# ---------------------------------------------------------------------
# 2) Relevante Dateien finden & Labels laden
# ---------------------------------------------------------------------
print("Suche relevante PTB-XL-Dateien…")
filepath_list = batch_process_relevant_ecgs(
    config.DATA_DIR, config.RELEVANT_ECG_PATH, config.DATABASE_PATH
)
print(f"→ {len(filepath_list)} Dateien gefunden (vor evtl. Teilmenge).")

ecg_id_to_scp_str = load_scp_codes(config.RELEVANT_ECG_PATH)
ecg_id_to_scp_list = {ecg_id: get_scp_code_list(s)
                      for ecg_id, s in ecg_id_to_scp_str.items()}

# ---------------------------------------------------------------------
# 3) Snippets laden (aus Cache) oder einmalig erstellen
# ---------------------------------------------------------------------
MAX_FILES = 1000  # Teilmenge für schnelle Testläufe

# Alle Parameter, die die Snippet-Erzeugung beeinflussen, in den Key
cache_params = {
    "sampling_rate":        config.SAMPLING_RATE,
    "before_r":             config.SNIPPET_LENGTH_BEFORE_R,
    "after_r":              config.SNIPPET_LENGTH_AFTER_R,
    "max_files":            MAX_FILES,
    # nur eintragen, falls Preprocessing davon abhängt:
    # "leads": "all",
    # "normalization": "per_snippet_zscore",
}

# Daten-Version (ändert sich, wenn Eingabedaten sich ändern)
data_version = compute_data_version(
    Path(config.RELEVANT_ECG_PATH),
    Path(config.DATABASE_PATH) if getattr(config, "DATABASE_PATH", None) else None,
)

# Cache-Key & -Ort
key = make_cache_key(cache_params, data_version=data_version)
cache_root = Path("cache") / "snippets"

# Build-Funktion kapselt deinen bisherigen Extraktions-Loop
def _build_snippets():
    all_snippets, all_ecg_ids, all_scp_labels_raw = [], [], []
    for filepath in filepath_list[:MAX_FILES]:
        snippets, ecg_ids, scp_labels = create_snippets(
            filepath, ecg_id_to_scp_list,
            config.SAMPLING_RATE,
            config.SNIPPET_LENGTH_BEFORE_R,
            config.SNIPPET_LENGTH_AFTER_R,
        )
        if snippets is not None and len(snippets) > 0:
            all_snippets.extend(snippets)
            all_ecg_ids.extend(ecg_ids)
            all_scp_labels_raw.extend(scp_labels)

    return (
        np.asarray(all_snippets, dtype=np.float32),
        np.asarray(all_ecg_ids),
        np.asarray(all_scp_labels_raw),
    )

# Laden oder bauen
snippets, ecg_ids, labels, meta_dict, npz_path = try_load_or_build(
    cache_root=cache_root,
    key=key,
    build_fn=_build_snippets,
    meta={
        "params": cache_params,
        "data_version": data_version,
        "relevant_path": str(config.RELEVANT_ECG_PATH),
        "database_path": str(config.DATABASE_PATH),
    },
    verbose=True,
)

# Ab hier weiter wie gehabt
all_snippets = snippets
all_ecg_ids = ecg_ids
all_scp_labels_raw = labels
print(f"Snippets geladen: {all_snippets.shape}  (Quelle: {npz_path.name})")

snippet_length, num_channels = all_snippets.shape[1], all_snippets.shape[2]
print(f"Snippets: {all_snippets.shape} (Länge={snippet_length}, Kanäle={num_channels})")
print(f"Anzahl Labels (roh): {len(all_scp_labels_raw)}")

# Für Visualisierung: nur Single-Label-Snippets
single_label_indices = [i for i, lab in enumerate(all_scp_labels_raw) if lab.count('-') == 0]
single_label_snippets = all_snippets[single_label_indices]
single_label_labels = all_scp_labels_raw[single_label_indices]
print(f"Single-Label-Snippets für Visualisierung: {len(single_label_labels)}")

# ---------------------------------------------------------------------
# 3,5) Test, wie Snippets aussehen
# ---------------------------------------------------------------------
def snippet_stats(snippet):
    """Stats über Zeit; bei 2D erst pro Kanal, dann Mittelwert über Kanäle."""
    import numpy as np
    if snippet.ndim == 1:
        x = snippet
        mean = float(np.mean(x))
        std = float(np.std(x))
        vmin = float(np.min(x))
        vmax = float(np.max(x))
    else:
        m = np.mean(snippet, axis=0)
        s = np.std(snippet, axis=0)
        mean = float(np.mean(m))
        std = float(np.mean(s))
        vmin = float(np.min(snippet))
        vmax = float(np.max(snippet))
    return mean, std, vmin, vmax



print("\nBeispielhafte Original-Snippets (mit Stats-Overlay):")

num_examples = min(10, len(all_snippets))
# reproduzierbare Auswahl (wie im Testskript – du kannst auch rng = np.random.default_rng(42) nehmen)
example_indices = np.random.choice(len(all_snippets), size=num_examples, replace=False)

# gleicher Dateiname im Run-Ordner
snip_plot_file = run.run_dir / "test_plot_snippets.png"

plt.figure(figsize=(12, 2.2 * num_examples))
per_snippet_means = []
per_snippet_stds = []

for i, idx in enumerate(example_indices):
    snippet = all_snippets[idx]
    plt.subplot(num_examples, 1, i + 1)

    # Variante A: nur Lead 0 (wie im Testskript)
    if snippet.ndim == 2 and snippet.shape[1] > 0:
        plt.plot(snippet[:, 0], label=f"ECG_ID={all_ecg_ids[idx]}, Label={all_scp_labels_raw[idx]}")
    else:
        plt.plot(snippet, label=f"ECG_ID={all_ecg_ids[idx]}, Label={all_scp_labels_raw[idx]}")

    # Stats berechnen und als Overlay anzeigen (wie im Testskript)
    mean, std, vmin, vmax = snippet_stats(snippet)
    per_snippet_means.append(mean)
    per_snippet_stds.append(std)

    ax = plt.gca()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    txt = f"mean={mean:.3f}, std={std:.3f}, min={vmin:.3f}, max={vmax:.3f}"
    ax.text(
        x=x0 + 0.99*(x1 - x0),
        y=y0 + 0.95*(y1 - y0),
        s=txt, ha="right", va="top", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
    )
    plt.legend(loc="upper left", fontsize="x-small")

plt.suptitle("Beispiele: Originale EKG-Snippets (Lead 0) mit Stats", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(snip_plot_file, dpi=150)
plt.close()
print(f"Plot gespeichert: {snip_plot_file}")

# ---------------------------------------------------------------------
# 4) Train/Test-Split (stratifiziert nach Label)
# ---------------------------------------------------------------------
from sklearn.model_selection import train_test_split
train_snippets, test_snippets, labels_train, labels_test = train_test_split(
    all_snippets, all_scp_labels_raw,
    test_size=0.2, random_state=42, stratify=all_scp_labels_raw
)
input_shape = (train_snippets.shape[1], train_snippets.shape[2])
print(f"Split: train={train_snippets.shape[0]} / test={test_snippets.shape[0]}")

# ---------------------------------------------------------------------
# 5) VQ-VAE Training(s)
# ---------------------------------------------------------------------
latent_dimensions_to_test = getattr(config, "LATENT_DIMENSIONS_TO_TEST", [32])
trained_vq_vaes = {}

for latent_dim in latent_dimensions_to_test:
    print(f"\n--- Training VQ-VAE: Latent {latent_dim}, Codebook {config.NUM_EMBEDDINGS}, β={config.COMMITMENT_COST} ---")
    # Erwartete Signatur deiner Funktion:
    # train_and_evaluate_vqvae(train_snip, test_snip, input_shape, latent_dim,
    #                          num_embeddings, commitment_cost, epochs, batch_size)
    vq_vae_model, history, total_loss = train_and_evaluate_vqvae(
        train_snippets, test_snippets, input_shape, latent_dim,
        config.NUM_EMBEDDINGS, config.COMMITMENT_COST,
        config.AE_EPOCHS, config.AE_BATCH_SIZE
    )
    trained_vq_vaes[latent_dim] = vq_vae_model

    # History speichern (falls verfügbar)
    try:
        h = history.history if hasattr(history, "history") else history
        with open(run.run_dir / f"history_LD{latent_dim}.json", "w") as f:
            json.dump(h, f, indent=2)
    except Exception:
        pass

# ---------------------------------------------------------------------
# 6) Visualisierung & Codebook-Analyse mit gewählter LD
# ---------------------------------------------------------------------
VIS_METHOD = getattr(config, "VIS_METHOD", "TSNE")
VIS_N_COMPONENTS = getattr(config, "VIS_N_COMPONENTS", 3)

# Nutze LD=32 falls trainiert, sonst die erste verfügbare
chosen_latent_dim = 32 if 32 in trained_vq_vaes else next(iter(trained_vq_vaes.keys()))
chosen_vq_vae_model = trained_vq_vaes[chosen_latent_dim]

params_str = (
    f"LD{chosen_latent_dim}_Emb{config.NUM_EMBEDDINGS}_"
    f"Cost{config.COMMITMENT_COST}_LR{config.LEARNING_RATE:.0e}_"
    f"Epochs{config.AE_EPOCHS}_Batch{config.AE_BATCH_SIZE}"
)
print("Parameter:", params_str)

# (a) Rekonstruktionen
print("\n Visualisiere EKG-Rekonstruktionen…")
reconstructed_test_snippets = chosen_vq_vae_model.predict(test_snippets, verbose=0)
plot_ecg_reconstructions(
    test_snippets, reconstructed_test_snippets, num_examples=5,
    filename=str(run.run_dir / f"ecg_reconstruction_{params_str}.png")
)

# (b) Latent-Space (nur Single-Label, wenn vorhanden)
if len(single_label_labels) > 0:
    print("Visualisiere quantisierten Latent-Raum…")
    visualize_latent_space(
        chosen_vq_vae_model,
        single_label_snippets,
        single_label_labels,
        n_components=VIS_N_COMPONENTS,
        method=VIS_METHOD,
        filename=str(run.run_dir / f"VQ-VAE_3d_latent_space_quantized_{VIS_METHOD}_{params_str}.html"),
        params_info=params_str,
    )
else:
    print("Keine Single-Label-Snippets für die Latentraum-Visualisierung vorhanden.")

# (c) Codebook-Embeddings
print("Visualisiere Codebook-Embeddings…")
visualize_codebook_embeddings(
    chosen_vq_vae_model.vq_layer,
    n_components=VIS_N_COMPONENTS,
    method=VIS_METHOD,
    filename=str(run.run_dir / f"codebooks_embeddings_{VIS_METHOD}_{params_str}.html"),
    params_info=params_str,
)

# (d) Codebook-Nutzung pro Run speichern
print("Analysiere Codebook-Nutzung…")
stats = analyze_codebook_usage(chosen_vq_vae_model, (x for x in [test_snippets]), max_batches=1, plot=False)
with open(run.run_dir / "codebook_usage.json", "w") as f:
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}, f, indent=2)

plt.figure(figsize=(10, 4))
plt.bar(stats["unique_indices"], stats["counts"], color="steelblue")
plt.title(f"Codebook-Nutzung: {len(stats['unique_indices'])}/{stats['num_embeddings']}")
plt.xlabel("Index"); plt.ylabel("Häufigkeit"); plt.tight_layout()
plt.savefig(run.run_dir / "codebook_usage.png", dpi=150)
plt.close()

print(
    f"Codebook-Analyse: {len(stats['unique_indices'])}/{stats['num_embeddings']} "
    f"({stats['utilization']*100:.1f}% genutzt). Dateien im Run-Ordner."
)

print("Code ist erfolgreich durchgelaufen.")
