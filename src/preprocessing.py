# src/preprocessing.py 
import numpy as np
import wfdb
import neurokit2 as nk
import os
import pandas as pd

from .config import DATA_DIR, RELEVANT_ECG_PATH, DATABASE_PATH, SAMPLING_RATE, SNIPPET_LENGTH_BEFORE_R, SNIPPET_LENGTH_AFTER_R, MIN_SNIPPETS_PER_FILE
from .data_loader import get_scp_code_list
#ecg_id_to_scp_list aus data_loader

# ---- Funktion zum Erstellen von Snippets aus der EKG-Datei ----
def create_snippets(filepath, ecg_id_to_scp_list,
                    SAMPLING_RATE, SNIPPET_LENGTH_BEFORE_R, SNIPPET_LENGTH_AFTER_R):
    """
    Liefert:
      - np.ndarray: (n_snippets, T, n_channels) [float32]
      - np.ndarray: (n_snippets,)               ecg_id (str)
      - np.ndarray: (n_snippets,)               scp_label (str)
    """
    file = filepath[:-4]  # Entferne .dat
    base_ecg_id = os.path.basename(filepath).split('.')[0]

    # Labels bestimmen
    try:
        base_ecg_id_int = int(base_ecg_id.split('_')[0])
        scp_codes = ecg_id_to_scp_list.get(base_ecg_id_int)
    except ValueError as e:
        print(f"Fehler beim Konvertieren der ECG-ID '{base_ecg_id}' in Integer: {e}")
        return None, None, None

    if not scp_codes:
        return None, None, None

    scp_label = "-".join(sorted(scp_codes))

    # Datensatz lesen
    try:
        record = wfdb.rdrecord(file)
        full_ecg = record.p_signal  # shape (N, n_channels), float
        n_samples, n_channels = full_ecg.shape

        # Referenzableitung für R-Peak-Detektion finden (bevorzugt "II")
        ref_idx = 0
        try:
            sig_names = getattr(record, "sig_name", None)
            if sig_names and "II" in sig_names:
                ref_idx = sig_names.index("II")
            elif n_channels > 1:
                ref_idx = 1  # PTB-XL: oft ist Kanal 1 (Index 1) Lead II in *_lr
        except Exception:
            ref_idx = min(1, n_channels - 1)

        # Clean + R-Peaks auf Referenzableitung (einmalig, nicht pro Kanal)
        ecg_cleaned = nk.ecg_clean(full_ecg[:, ref_idx], sampling_rate=SAMPLING_RATE)
        try:
            _, info = nk.ecg_process(ecg_cleaned, sampling_rate=SAMPLING_RATE)
            r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        except Exception as e_nk:
            print(f"Fehler bei nk.ecg_process in {filepath}: {e_nk}")
            r_peaks = np.array([], dtype=int)

        if r_peaks.size == 0:
            return None, None, None

        # Snippets schneiden
        min_len = SNIPPET_LENGTH_BEFORE_R + SNIPPET_LENGTH_AFTER_R
        snippets = []
        ecg_ids = []
        labels = []

        for r in r_peaks:
            start = r - SNIPPET_LENGTH_BEFORE_R
            stop = r + SNIPPET_LENGTH_AFTER_R
            if start < 0 or stop > n_samples:
                continue
            if (stop - start) != min_len:
                continue

            snippet = full_ecg[start:stop, :].astype(np.float32, copy=False)

            # Baseline entfernen (pro Kanal)
            snippet = snippet - np.mean(snippet, axis=0, keepdims=True)

            # z-Score Normierung (pro Snippet; über alle Kanäle gemeinsam stabilisieren)
            # Wenn du lieber pro Kanal normierst, benutze axis=0 bei std:
            std = np.std(snippet)
            if std > 0:
                snippet = snippet / std

            # Optional: Clip gegen Ausreißer
            snippet = np.clip(snippet, -8.0, 8.0)

            snippets.append(snippet)
            ecg_ids.append(base_ecg_id)
            labels.append(scp_label)

        n_snippets = len(snippets)
        if n_snippets < MIN_SNIPPETS_PER_FILE:
            # konsistent mit deinem bisherigen Verhalten
            print(f"Zu wenige valide Snippets ({n_snippets}) in Datei: {filepath}")
            return None, None, None

        print(f"Erfolgreich {n_snippets} valide Snippets aus Datei: {filepath} extrahiert (SCP-Label: {scp_label}).")

        return (
            np.asarray(snippets, dtype=np.float32),
            np.asarray(ecg_ids),
            np.asarray(labels)
        )

    except Exception as e_main:
        print(f"Hauptfehler beim Verarbeiten der Datei {filepath}: {e_main}")
        return None, None, None
    
# ---- Alle relevanten Dateien verarbeiten ----
def batch_process_relevant_ecgs(DATA_DIR, RELEVANT_ECG_PATH, DATABASE_PATH):
    filepath_list = []
    df_relevant_ecgs = pd.read_csv(RELEVANT_ECG_PATH)
    list_relevant_ecg_ids = df_relevant_ecgs['ecg_id'].tolist()

    df_database = pd.read_csv(DATABASE_PATH)
    ecg_id_to_filename = pd.Series(df_database['filename_lr'].values, index=df_database['ecg_id']).to_dict()

    for ecg_id in list_relevant_ecg_ids:
        if ecg_id in ecg_id_to_filename:
            base_filename = ecg_id_to_filename[ecg_id]
            dat_filepath = os.path.join(DATA_DIR, base_filename + ".dat")
            filepath_list.append(dat_filepath)
        else:
            print(f"Warnung: ECG-ID '{ecg_id}' nicht in der Datenbank gefunden.")
    return(filepath_list)
