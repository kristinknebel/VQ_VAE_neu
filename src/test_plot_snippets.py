# src/test_plot_snippets.py
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

print(f" Neuer Testlauf: {run.run_dir}")

params_str = (
    f"LD{chosen_latent_dim}_Emb{config.NUM_EMBEDDINGS}_"
    f"Cost{config.COMMITMENT_COST}_LR{config.LEARNING_RATE:.0e}_"
    f"Epochs{config.AE_EPOCHS}_Batch{config.AE_BATCH_SIZE}")

filename = str(run.run_dir / f"ecg_reconstruction_{params_str}_test_plot_snippets.png")

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
# 3,5) Test, wie Snippets aussehen
# ---------------------------------------------------------------------
print("\nBeispielhafte Original-Snippets (ungefiltert):")

num_examples = 10
example_indices = np.random.choice(len(all_snippets), size=num_examples, replace=False)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(example_indices):
    snippet = all_snippets[idx]

    # Falls das Snippet mehrkanalig ist (z. B. 12 Kanäle)
    if snippet.ndim == 2:
        plt.subplot(num_examples, 1, i + 1)
        # z.B. Kanal I anzeigen (index 0)
        plt.plot(snippet[:, 0], label=f"ECG_ID={all_ecg_ids[idx]}, Label={all_scp_labels_raw[idx]}")
        plt.legend(loc="upper right", fontsize="small")
    else:
        plt.subplot(num_examples, 1, i + 1)
        plt.plot(snippet, label=f"ECG_ID={all_ecg_ids[idx]}, Label={all_scp_labels_raw[idx]}")
        plt.legend(loc="upper right", fontsize="small")

plt.suptitle("Beispiele: Originale EKG-Snippets", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(filename)
plt.close()


