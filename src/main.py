# src/main.py

import sys
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

from src import config
from src.experiment import start_run

# 1) Parameter-Dict für den Snapshot (alles, was du später nachvollziehen willst)
params = {
    "LATENT_DIM": getattr(config, "LATENT_DIM", None),
    "NUM_EMBEDDINGS": getattr(config, "NUM_EMBEDDINGS", None),
    "COMMITMENT_COST": getattr(config, "COMMITMENT_COST", None),
    "LEARNING_RATE": getattr(config, "LEARNING_RATE", None),
    "AE_BATCH_SIZE": getattr(config, "AE_BATCH_SIZE", None),
    "AE_EPOCHS": getattr(config, "AE_EPOCHS", None),
    "SAMPLING_RATE": getattr(config, "SAMPLING_RATE", None),
    "SNIPPET_LENGTH_BEFORE_R": getattr(config, "SNIPPET_LENGTH_BEFORE_R", None),
    "SNIPPET_LENGTH_AFTER_R": getattr(config, "SNIPPET_LENGTH_AFTER_R", None),
    # gerne erweitern: Datenquelle, Split, Filter, Label-Set etc.
}

# 2) Run starten
run = start_run(params, base_dir="runs", seed=42)

# 3) Callbacks in fit() einhängen
callbacks = run.callbacks(monitor="val_loss", patience=8)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config.AE_EPOCHS,
    callbacks=callbacks,
    # batch_size=config.AE_BATCH_SIZE  # falls du tf.data nutzt, entfällt das
)






import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt # Import für plt.show()

# --- IMPORTS ANPASSEN ---
from .config import (DATA_DIR, RELEVANT_ECG_PATH, DATABASE_PATH,
    SAMPLING_RATE, SNIPPET_LENGTH_BEFORE_R, SNIPPET_LENGTH_AFTER_R,
    LATENT_DIMENSIONS_TO_TEST, AE_EPOCHS, AE_BATCH_SIZE, # Alte AE-Parameter
    NUM_EMBEDDINGS, COMMITMENT_COST,LEARNING_RATE, # NEUE VQ-VAE Parameter
    VIS_N_COMPONENTS, VIS_METHOD) # Visualisierungsparameter aus config

from .data_loader import load_scp_codes, get_scp_code_list
from .preprocessing import create_snippets, batch_process_relevant_ecgs

# NEUE IMPORTS FÜR VQ-VAE
from .models import VQVAE, train_and_evaluate_vqvae # Importieren Sie VQVAE und die neue Trainingsfunktion
from .visualization import visualize_latent_space, plot_ecg_reconstructions, visualize_codebook_embeddings # Neue Visualisierungsfunktionen


#
# Definiere String mit allen Parametern





# --- DATENLADE- UND VORVERARBEITUNG (BLEIBT GLEICH) ---
filepath_list = batch_process_relevant_ecgs(DATA_DIR, RELEVANT_ECG_PATH, DATABASE_PATH)

# sortierte Liste der SCP-Codes erhalten
ecg_id_to_scp_str = load_scp_codes(RELEVANT_ECG_PATH)
ecg_id_to_scp_list = {ecg_id: get_scp_code_list(scp_string)
                        for ecg_id, scp_string in ecg_id_to_scp_str.items()}

all_snippets = []
all_ecg_ids = []
all_scp_labels_raw = [] # Unveränderte Labels

# Begrenzung für Testzwecke beibehalten, aber beachten, dass dies die Datenmenge stark reduziert
# Für ernsthaftes Training sollten Sie dies entfernen oder erhöhen.
# Sie müssen hier möglicherweise auch eine Obergrenze für die Anzahl der gesamten Snippets einführen,
# da die Gesamtzahl der Snippets aus allen Dateien sehr groß werden kann.
# Eine Alternative ist, nur eine Teilmenge der `filepath_list` zu verarbeiten
# oder die Verarbeitung in Batches durchzuführen, wenn der Speicher knapp wird.
for filepath in filepath_list[:1000]: # Begrenzung für Testzwecke beibehalten
    snippets, ecg_ids, scp_labels = create_snippets(filepath, ecg_id_to_scp_list, SAMPLING_RATE, SNIPPET_LENGTH_BEFORE_R, SNIPPET_LENGTH_AFTER_R)
    
    if snippets is not None:
        all_snippets.extend(snippets)
        all_ecg_ids.extend(ecg_ids)
        all_scp_labels_raw.extend(scp_labels) # Füge die rohen Labels hinzu

if not all_snippets:
    print("Keine Snippets zur Verarbeitung gefunden. Stellen Sie sicher, dass die Dateipfade korrekt sind und Snippets extrahiert werden können.")
else:
    all_snippets = np.array(all_snippets)
    all_ecg_ids = np.array(all_ecg_ids)
    all_scp_labels_raw = np.array(all_scp_labels_raw)

    # Filtere die Daten, um nur Snippets mit einem einzigen SCP-Code zu behalten
    # Dies ist für die Visualisierung gut, aber das Training sollte auf allen Daten erfolgen.
    single_label_indices = [i for i, label in enumerate(all_scp_labels_raw) if label.count('-') == 0]
    single_label_snippets = all_snippets[single_label_indices]
    single_label_labels = all_scp_labels_raw[single_label_indices]

    snippet_length, num_channels = all_snippets.shape[1], all_snippets.shape[2]
    print(f"Form der extrahierten Snippets: (Anzahl, Länge={snippet_length}, Kanäle={num_channels})")
    print(f"Anzahl der extrahierten SCP-Labels (roh): {len(all_scp_labels_raw)}")
    print(f"Anzahl der Snippets mit einzelnem Label für Visualisierung: {len(single_label_labels)}")

    # Daten aufteilen für das Training mit allen Labels
    # `stratify` ist wichtig für unbalancierte Datensätze, stellt sicher, dass Klassenanteile in Train/Test gleich sind
    train_snippets, test_snippets, _, _, train_labels_for_stratify, test_labels_for_stratify = train_test_split(
        all_snippets, all_ecg_ids, all_scp_labels_raw, test_size=0.2, random_state=42, stratify=all_scp_labels_raw
    )
    # Beachten Sie, dass `all_ecg_ids` und `all_scp_labels_raw` im `train_test_split` verwendet werden,
    # aber Sie benötigen nur `train_snippets` und `test_snippets` für den VAE selbst.
    # Die `test_labels_for_stratify` werden für die Visualisierung verwendet.

    input_shape = (train_snippets.shape[1], train_snippets.shape[2])
    latent_dimensions_to_test = LATENT_DIMENSIONS_TO_TEST
    trained_vq_vaes = {} # Dictionary um VQ-VAE Modelle zu speichern

    # --- VQ-VAE TRAINING LOOP ANPASSEN ---
    for latent_dim in latent_dimensions_to_test:
        print(f"\n--- Training VQ-VAE mit Latenz-Dimension: {latent_dim}, Codebook-Größe: {NUM_EMBEDDINGS} ---")
        # Aufruf der neuen VQ-VAE spezifischen Trainingsfunktion
        vq_vae_model, history, total_loss = train_and_evaluate_vqvae(
            train_snippets, test_snippets, input_shape, latent_dim,
            NUM_EMBEDDINGS, COMMITMENT_COST, AE_EPOCHS, AE_BATCH_SIZE) 
        trained_vq_vaes[latent_dim] = vq_vae_model

    # --- VISUALISIERUNG ANPASSEN ---
    # Hier verwenden wir das Modell für die größte getestete Dimension, falls vorhanden.
    # Stellen Sie sicher, dass diese Dimension auch wirklich getestet wurde.
    if 32 in trained_vq_vaes: # Überprüfen, ob das Modell für LD x existiert
        chosen_latent_dim = 32 #evtl. anpassen, wenn man doch Liste für latent dims nimmt
        params_str = (
            f"LD{chosen_latent_dim}_Emb{NUM_EMBEDDINGS}_Cost{COMMITMENT_COST}_LR{LEARNING_RATE:.0e}_"
            f"Epochs{AE_EPOCHS}_Batch{AE_BATCH_SIZE}"
        )
        print("Parameter:"+params_str)
        chosen_vq_vae_model = trained_vq_vaes[chosen_latent_dim] # Nehmen Sie das trainierte VQ-VAE Modell

        # Rekonstruktionen visualisieren
        print("\nVisualisiere EKG-Rekonstruktionen...")
        reconstructed_test_snippets = chosen_vq_vae_model.predict(test_snippets)
        plot_ecg_reconstructions(test_snippets, reconstructed_test_snippets, num_examples=5, filename = "32_ecg_reconstruction_{params_str}.png")

        # Visualisierung des quantisierten Latent-Raums für einzelne Labels
        if len(single_label_labels) > 0: # Prüfen, ob überhaupt Single Labels vorhanden sind
            print("Visualisiere quantisierten Latent-Raum für einzelne Labels...")
            visualize_latent_space(
                chosen_vq_vae_model, # Das VQ-VAE Modell übergeben
                single_label_snippets, # Nur Snippets mit einzelnen Labels
                single_label_labels,   # Und deren Labels
                n_components=VIS_N_COMPONENTS,
                method=VIS_METHOD, filename = "VQ-VAE_3d_latent_space_quantized_{method}_{params_str}.html", params_info = params_str
            )
            print("Visualisierung Latent-Raum abgeschlossen.")
        else:
            print("Keine Snippets mit einzelnen Labels zum Visualisieren des Latent-Raums vorhanden.")

        # Visualisierung der Codebook-Embeddings
        print("Visualisiere Codebook-Embeddings...")
        visualize_codebook_embeddings(
            chosen_vq_vae_model.vq_layer, # Zugriff auf die VQ-Layer Instanz
            n_components=VIS_N_COMPONENTS,
            method=VIS_METHOD,
            filename = "codebooks_embeddings_{method}_{params_str}.html", params_info = params_str)
        plt.show() # Zeigt alle generierten Plots an (falls nicht bereits durch plotly.show() behandelt)

    else:
        print(f"Modell für Latenzdimension 32 wurde nicht trainiert oder gefunden. Keine Visualisierung des Latent-Raums.")

print(f"Code ist erfolgreich durchgelaufen.")
