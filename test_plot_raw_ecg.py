import pandas as pd
import wfdb
import matplotlib.pyplot as plt

# --- Pfade anpassen ---
PTBXL_PATH = "/pfad/zu/ptb-xl/"  # z. B. "/users/stud/knebelkr/datasets/ptb-xl/"
CSV_PATH = PTBXL_PATH + "ptbxl_database.csv"

# --- Metadaten laden ---
df = pd.read_csv(CSV_PATH)

# Beispiel: erste Aufnahme auswählen
record_name = df.iloc[0].ecg_filename
record_path = PTBXL_PATH + record_name

# --- EKG-Signal laden ---
signal, info = wfdb.rdsamp(record_path)

# --- Beispiel: Lead I plotten ---
plt.figure(figsize=(12, 4))
plt.plot(signal[:, 0], linewidth=1)
plt.title(f"PTB-XL Sample: {record_name}\nLead I")
plt.xlabel("Samples (500 Hz)")
plt.ylabel("Amplitude (µV)")
plt.grid(True)
plt.tight_layout()
plt.show()
