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
# 3) Snippets extrahieren (Teilmenge für Testläufe)
# ---------------------------------------------------------------------
all_snippets = []
all_ecg_ids = []
all_scp_labels_raw = []

MAX_FILES = 1000  # kleine Begrenzung für schnellere Test-Runs
print(f" Extrahiere Snippets aus bis zu {MAX_FILES} Dateien…")
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

if not all_snippets:
    print("Keine Snippets gefunden. Prüfe Pfade/Preprocessing.")
    sys.exit(0)

all_snippets = np.array(all_snippets)
all_ecg_ids = np.array(all_ecg_ids)
all_scp_labels_raw = np.array(all_scp_labels_raw)

snippet_length, num_channels = all_snippets.shape[1], all_snippets.shape[2]
print(f"Snippets: {all_snippets.shape} (Länge={snippet_length}, Kanäle={num_channels})")
print(f"Anzahl Labels (roh): {len(all_scp_labels_raw)}")

# Für Visualisierung: nur Single-Label-Snippets (dein Kriterium)
single_label_indices = [i for i, label in enumerate(all_scp_labels_raw) if label.count('-') == 0]
single_label_snippets = all_snippets[single_label_indices]
single_label_labels = all_scp_labels_raw[single_label_indices]
print(f"   Single-Label-Snippets für Visualisierung: {len(single_label_labels)}")

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
