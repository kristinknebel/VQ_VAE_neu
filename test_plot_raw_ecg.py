import pandas as pd
import wfdb
import matplotlib.pyplot as plt

# Pfad zum Basisverzeichnes des Projekts (von diesem Skript aus gesehen)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Konfiguration der Dateipfade ---
DATA_DIR = os.path.join(BASE_PATH,'data','ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
CSV_PATH = DATA_DIR + "ptbxl_database.csv"

OUTPUT_DIR = "./plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Metadaten laden ---
df = pd.read_csv(CSV_PATH)

# Beispiel: erste Aufnahme auswählen
record_name = df.iloc[0].ecg_filename
record_path = os.path.join(PTBXL_PATH, record_name)

# --- EKG-Signal laden ---
signal, info = wfdb.rdsamp(record_path)

# --- Beispiel: Lead I plotten ---
plt.figure(figsize=(12, 4))
plt.plot(signal[:, 0], linewidth=1)
plt.title(f"PTB-XL Sample: {os.path.basename(record_name)}\nLead I")
plt.xlabel("Samples (500 Hz)")
plt.ylabel("Amplitude (µV)")
plt.grid(True)
plt.tight_layout()

# --- Plot speichern ---
output_file = os.path.join(OUTPUT_DIR, "ptbxl_sample_leadI.png")
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Plot gespeichert unter: {output_file}")
