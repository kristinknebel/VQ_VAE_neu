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
matplotlib.use("Agg")  # wichtig auf Servern ohne Display -> plt.show() wird ignoriert
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
from itertools import product
import json
import datetime
import pandas as pd

DO_PLOTS = False #extra Skript für Plots

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

#run = start_run(params, base_dir="runs", seed=42)
#print(f" Neuer Trainingslauf: {run.run_dir}")

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
    "normalization": "zscore_per_channel",
    "split_level": "patient_id",
    "preprocessing_version": "v2_bandpass150_notch50",
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

# ---------------------------------------------------------------------
# Patient-ID Mapping aus ptbxl_database.csv laden
# ---------------------------------------------------------------------
ptbxl_csv = Path(config.DATA_DIR) /  "ptbxl_database.csv"
df_meta = pd.read_csv(ptbxl_csv)

# PTB-XL: Spalten heißen typischerweise "ecg_id" und "patient_id"
ecg_to_patient = dict(zip(df_meta["ecg_id"].astype(int), df_meta["patient_id"].astype(int)))

# Achtung: deine all_ecg_ids sind Strings wie "12345_lr" oder "12345"
# -> wir extrahieren die führende Zahl
def _to_ecg_int(ecg_id_str: str) -> int:
    return int(str(ecg_id_str).split("_")[0])

all_ecg_ids_int = np.array([_to_ecg_int(x) for x in all_ecg_ids], dtype=int)

# patient_id pro Snippet bestimmen
all_patient_ids = np.array([ecg_to_patient.get(eid, -1) for eid in all_ecg_ids_int], dtype=int)

# Safety-Check: wie viele fehlen?
n_missing = int(np.sum(all_patient_ids < 0))
if n_missing > 0:
    print(f"[WARN] {n_missing} Snippets haben keine patient_id (ecg_id nicht im CSV). Diese werden entfernt.")
    keep = all_patient_ids >= 0
    all_snippets = all_snippets[keep]
    all_scp_labels_raw = np.asarray(all_scp_labels_raw)[keep]
    all_ecg_ids = np.asarray(all_ecg_ids)[keep]
    all_ecg_ids_int = all_ecg_ids_int[keep]
    all_patient_ids = all_patient_ids[keep]


print(f"Snippets geladen: {all_snippets.shape}  (Quelle: {npz_path.name})")

snippet_length, num_channels = all_snippets.shape[1], all_snippets.shape[2]
print(f"Snippets: {all_snippets.shape} (Länge={snippet_length}, Kanäle={num_channels})")
print(f"Anzahl Labels (roh): {len(all_scp_labels_raw)}")

# ---------------------------------------------------------------------
# 4) Train/Test-Split OHNE Leakage (Patient-level, KEIN Stratify)
# ---------------------------------------------------------------------
from sklearn.model_selection import train_test_split

all_labels = np.asarray(all_scp_labels_raw)
all_snips  = np.asarray(all_snippets)

unique_patients = np.unique(all_patient_ids)

# Einfach zufälliger Patient-Split (kein Stratify!)
train_patients, test_patients = train_test_split(
    unique_patients,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_mask = np.isin(all_patient_ids, train_patients)
test_mask  = np.isin(all_patient_ids, test_patients)

train_snippets = all_snips[train_mask]
test_snippets  = all_snips[test_mask]
labels_train   = all_labels[train_mask]
labels_test    = all_labels[test_mask]
ecg_ids_train  = np.asarray(all_ecg_ids)[train_mask]
ecg_ids_test   = np.asarray(all_ecg_ids)[test_mask]

input_shape = (train_snippets.shape[1], train_snippets.shape[2])

print(f"Split (PATIENT-level, no stratify):")
print(f"  train_snips={train_snippets.shape[0]} / test_snips={test_snippets.shape[0]}")
print(f"  unique patients: train={len(np.unique(train_patients))} / test={len(np.unique(test_patients))}")
print(f"  unique ecg_ids: train={len(np.unique(ecg_ids_train))} / test={len(np.unique(ecg_ids_test))}")

# Harte Leakage-Checks
assert len(set(train_patients).intersection(set(test_patients))) == 0, "PATIENT Leakage!"
assert len(set(ecg_ids_train).intersection(set(ecg_ids_test))) == 0, "ECG_ID Leakage!"
print("✅ Kein Patient- oder ECG-Leakage.")


import hashlib

def snippet_fingerprint(x: np.ndarray) -> str:
    xb = np.asarray(x, dtype=np.float32).tobytes(order="C")
    return hashlib.sha256(xb).hexdigest()[:16]

test_fps = np.array([snippet_fingerprint(s) for s in test_snippets])

single_label_mask_test = np.array([lab.count('-') == 0 for lab in labels_test])
single_label_snippets_test = test_snippets[single_label_mask_test]
single_label_labels_test   = labels_test[single_label_mask_test]

np.save("cache/train_patients.npy", train_patients)
np.save("cache/test_patients.npy", test_patients)

# ---------------------------------------------------------------------
# 5) Automatisierte Hyperparameter-Experimente (wie bisher, aber mehrere)
# ---------------------------------------------------------------------
from itertools import product
import datetime

# === Suchraster definieren ===
LEARNING_RATES = [1e-4, 5e-4]
COMMITMENT_COSTS = [0.1, 0.25, 0.5]
LATENT_DIMENSIONS_TO_TEST = [16, 32]
NUM_EMBEDDINGS_TO_TEST = [64, 128]
AE_BATCH_SIZES = [32]
AE_EPOCHS = [100]

# === Alle Kombinationen erzeugen ===
experiments = list(product(
    LATENT_DIMENSIONS_TO_TEST,
    NUM_EMBEDDINGS_TO_TEST,
    COMMITMENT_COSTS,
    LEARNING_RATES,
    AE_BATCH_SIZES,
    AE_EPOCHS,
))
print(f"Gesamtanzahl Experimente: {len(experiments)}")

# === Range-Filter (für parallele Ausführung) ===
if len(sys.argv) == 3 and sys.argv[1] == "--range":
    start_idx, end_idx = map(int, sys.argv[2].split("-"))
    experiments_subset = experiments[start_idx:end_idx+1]
    print(f"Führe Experimente {start_idx}-{end_idx} aus ({len(experiments_subset)} Stück).")
else:
    experiments_subset = experiments
    print(f"Führe alle {len(experiments_subset)} Experimente aus.")

results_summary = []
# ---------------------------------------------------------------------
# Trainingsschleife
# ---------------------------------------------------------------------
for (latent_dim, num_emb, beta, lr, batch_size, epochs) in experiments_subset:
    exp_name = (
        f"LD{latent_dim}_Emb{num_emb}_Cost{beta}_LR{lr:.0e}_"
        f"Epochs{epochs}_Batch{batch_size}"
    )
    print(f"\n--- Starte Experiment: {exp_name} ---")

    params = {
        "LATENT_DIMENSIONS": latent_dim,
        "NUM_EMBEDDINGS": num_emb,
        "COMMITMENT_COST": beta,
        "LEARNING_RATE": lr,
        "AE_BATCH_SIZE": batch_size,
        "AE_EPOCHS": epochs,
        "SAMPLING_RATE": config.SAMPLING_RATE,
        "SNIPPET_LENGTH_BEFORE_R": config.SNIPPET_LENGTH_BEFORE_R,
        "SNIPPET_LENGTH_AFTER_R": config.SNIPPET_LENGTH_AFTER_R,
        "DATA_DIR": str(config.DATA_DIR),
    }

    # Eigenen Run-Ordner für jedes Experiment
    run = start_run(params, base_dir="runs", seed=42)
    print(f"  → Run-Ordner: {run.run_dir}")

    # Fingerprints in JEDEM Experiment-Run speichern (damit Plot-Skript nur run_dir braucht)
    np.save(run.run_dir / "test_fingerprints.npy", test_fps)

    
    # Optional hilfreich: cache key / meta referenzieren
    with open(run.run_dir / "cache_ref.json", "w") as f:
        json.dump(
          {"cache_key": key, "cache_root": str(cache_root),
           "data_version": data_version, "cache_params": cache_params},
          f, indent=2
        )

    # --- Training ---
    vq_vae_model, history, total_loss = train_and_evaluate_vqvae(
        train_snippets, test_snippets, input_shape, latent_dim,
        num_emb, beta, epochs, batch_size, learning_rate=lr
    )
    # --- Model speichern (wie MCG) ---
    vq_vae_model.save_weights(run.run_dir / "vqvae_final.weights.h5")
    # --- Codebook Embeddings speichern ---
    np.save(run.run_dir / "codebook_embeddings.npy",
            vq_vae_model.vq_layer.embeddings.numpy())
    # --- (A3) Quantisierte Snippet-Embeddings speichern (wie MCG beat_embeddings) ---
    quantized = vq_vae_model.get_latent_representation(test_snippets).numpy()  # (N, T', latent_dim)
    quantized_mean = quantized.mean(axis=1)  # (N, latent_dim) -> pro Snippet aggregiert
    
    np.save(run.run_dir / "snippet_embeddings.npy", quantized_mean)
    np.save(run.run_dir / "snippet_ecg_ids.npy", np.asarray(ecg_ids_test))
    with open(run.run_dir / "snippet_labels.json", "w") as f:
        json.dump([str(x) for x in labels_test], f, indent=2)

    params_str = (
        f"LD{latent_dim}_Emb{num_emb}_Cost{beta}_LR{lr:.0e}_"
        f"Epochs{epochs}_Batch{batch_size}"
    )

    # --- (a) Rekonstruktionen ---
    
    if DO_PLOTS:
        print("  → Speichere Rekonstruktionen…")
        reconstructed_test_snippets = vq_vae_model.predict(test_snippets, verbose=0)
        plot_ecg_reconstructions(
            test_snippets, reconstructed_test_snippets, num_examples=5,
            filename=str(run.run_dir / f"ecg_reconstruction_{params_str}.png")
        )

    # --- (b) Latent-Space ---
    if DO_PLOTS:
        print(f"  → Single-Label-Snippets (TEST) verfügbar: {len(single_label_labels_test)}")
        if len(single_label_labels_test) > 0:
            print("  → Visualisiere quantisierten Latent-Raum…")
            visualize_latent_space(
                vq_vae_model,
                single_label_snippets_test,
                single_label_labels_test,
                n_components=getattr(config, "VIS_N_COMPONENTS", 3),
                method=getattr(config, "VIS_METHOD", "TSNE"),
                filename=str(run.run_dir / f"VQ-VAE_3d_latent_space_quantized_TSNE_{params_str}.html"),
                params_info=params_str,
            )
        else:
            print("Keine Single-Label-Snippets (TEST) für die Latentraum-Visualisierung vorhanden.")

    # --- (c) Codebook Embeddings ---
    if DO_PLOTS:
        print("  → Visualisiere Codebook-Embeddings…")
        visualize_codebook_embeddings(
            vq_vae_model.vq_layer,
            n_components=getattr(config, "VIS_N_COMPONENTS", 3),
            method=getattr(config, "VIS_METHOD", "TSNE"),
            filename=str(run.run_dir / f"codebooks_embeddings_TSNE_{params_str}.html"),
            params_info=params_str,
        )

    # --- (d) Codebook-Nutzung ---
    print("  → Analysiere Codebook-Nutzung…")
    stats = analyze_codebook_usage(vq_vae_model, (x for x in [test_snippets]), max_batches=1, plot=False)
    np.save(run.run_dir / "codebook_used_indices.npy", np.asarray(stats["unique_indices"]))      # Codebook-Usage Rohdaten speichern (wie MCG)
    np.save(run.run_dir / "codebook_counts.npy", np.asarray(stats["counts"]))
    with open(run.run_dir / "codebook_usage.json", "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}, f, indent=2)
    if DO_PLOTS:
        plt.figure(figsize=(10, 4))
        plt.bar(stats["unique_indices"], stats["counts"], color="steelblue")
        plt.title(f"Codebook-Nutzung: {len(stats['unique_indices'])}/{stats['num_embeddings']}")
        plt.xlabel("Index"); plt.ylabel("Häufigkeit"); plt.tight_layout()
        plt.savefig(run.run_dir / "codebook_usage.png", dpi=150)
        plt.close()    

    # --- (e) Trainingsverlauf speichern ---
    if hasattr(history, "history"):
        with open(run.run_dir / "history.json", "w") as f:
            json.dump(history.history, f, indent=2)

    # --- (f) Zusammenfassung ergänzen ---
    results_summary.append({
        "run_dir": str(run.run_dir),
        "latent_dim": latent_dim,
        "num_embeddings": num_emb,
        "commitment_cost": beta,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "val_loss": float(np.min(history.history["val_loss"])) if hasattr(history, "history") else None,
        "codebook_utilization": stats["utilization"],
    })

# ---------------------------------------------------------------------
# 6) Zusammenfassung speichern
# ---------------------------------------------------------------------
summary_path = run.run_dir / "summary.json"
with open(summary_path, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\nAlle {len(experiments_subset)} Experimente abgeschlossen.")
print(f"→ Zusammenfassung gespeichert unter: {summary_path}")
print("Code ist erfolgreich durchgelaufen.")
