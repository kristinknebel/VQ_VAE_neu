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
def create_snippets(filepath, ecg_id_to_scp_list, SAMPLING_RATE, SNIPPET_LENGTH_BEFORE_R, SNIPPET_LENGTH_AFTER_R):
    file = filepath[:-4]  # Entferne .dat
    base_ecg_id = os.path.basename(filepath).split('.')[0] 

    try:
        base_ecg_id_int = int(base_ecg_id.split('_')[0])
        scp_codes = ecg_id_to_scp_list.get(base_ecg_id_int)
        #print(scp_codes)
    except ValueError as e:
        print(f"Fehler beim Konvertieren der ECG-ID '{base_ecg_id}' in Integer: {e}")
        return None, None, None

    if scp_codes is None or not scp_codes:
        #print(f"Warnung: Keine SCP-Codes gefunden für ECG-ID {base_ecg_id}")
        return None, None, None

    # Erstelle ein eindeutiges Label aus der Liste der SCP-Codes 
    scp_label = "-".join(sorted(scp_codes)) # Konvertiere die sortierten Codes in einen String

    try:
        record = wfdb.rdrecord(file)
        full_ecg = record.p_signal
        num_channels = full_ecg.shape[1]
        r_peaks_all_channels = []

        for i in range(num_channels):
            ecg_channel = full_ecg[:, i]
            ecg_cleaned = nk.ecg_clean(ecg_channel, sampling_rate=SAMPLING_RATE)
            try:
                _, info = nk.ecg_process(ecg_cleaned, sampling_rate=SAMPLING_RATE)
                r_peaks = info["ECG_R_Peaks"]
            except Exception as e_nk:
                print(f"Fehler bei nk.ecg_process für Kanal {i} in {filepath}: {e_nk}")
                r_peaks = np.array([])

            r_peaks_all_channels.append(r_peaks)

        ecg_snippets_with_labels = []
        min_snippet_length = SNIPPET_LENGTH_BEFORE_R + SNIPPET_LENGTH_AFTER_R
        for r_peaks_channel in r_peaks_all_channels:
            r_peaks_channel = np.nan_to_num(r_peaks_channel).astype(int)
            start_list = r_peaks_channel - SNIPPET_LENGTH_BEFORE_R
            stop_list = r_peaks_channel + SNIPPET_LENGTH_AFTER_R
            for start, stop in zip(start_list, stop_list):
                if 0 <= start < len(full_ecg) and stop <= len(full_ecg) and len(full_ecg[start:stop]) == min_snippet_length:
                    snippet = full_ecg[start:stop, :]
                    # Normalisierung der Snippets
                    max_abs_val = np.max(np.abs(snippet))
                    if max_abs_val > 0:
                        normalized_snippet = snippet / max_abs_val
                    else:
                        normalized_snippet = snippet  # Vermeide Division durch Null

                    ecg_snippets_with_labels.append({'snippet': normalized_snippet, 'ecg_id': base_ecg_id, 'scp_label': scp_label})


        num_snippets = len(ecg_snippets_with_labels)
        if num_snippets < 5:
            print(f"Zu wenige valide Snippets ({num_snippets}) in Datei: {filepath}")
            return None, None, None
        else:
            print(f"Erfolgreich {num_snippets} valide Snippets aus Datei: {filepath} extrahiert (SCP-Label: {scp_label}).")

        ecg_snippets = [item['snippet'] for item in ecg_snippets_with_labels]
        ecg_ids = [item['ecg_id'] for item in ecg_snippets_with_labels]
        scp_labels = [item['scp_label'] for item in ecg_snippets_with_labels]

        return np.array(ecg_snippets), np.array(ecg_ids), np.array(scp_labels)

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
