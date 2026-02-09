#src/check_split_leakage.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
cache_ref = json.loads((run_dir / "cache_ref.json").read_text())

cache_root = Path(cache_ref["cache_root"])
cache_key  = cache_ref["cache_key"]
npz_path   = cache_root / f"snippets_{cache_key}.npz"

print("Using NPZ:", npz_path)
with np.load(npz_path, allow_pickle=False) as d:
    ecg_ids = d["ecg_ids"].astype(str)

print("Unique ecg_ids in cache:", len(np.unique(ecg_ids)))

# --- PTB-XL DB laden (Pfad ggf. anpassen!) ---
db = pd.read_csv("data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")

# ecg_ids in deinem Code sind Strings wie "12345_lr" -> int vorne rausziehen
def to_int(x):
    try:
        return int(str(x).split("_")[0])
    except:
        return None

ecg_int = np.array([to_int(x) for x in ecg_ids])
mask = np.array([v is not None for v in ecg_int])
ecg_int = ecg_int[mask]

sub = db[db["ecg_id"].isin(ecg_int)][["ecg_id","patient_id"]].drop_duplicates()

# 1) Wie viele ecg_id pro patient?
g = sub.groupby("patient_id")["ecg_id"].nunique()
print("Patients in used ecg_ids:", g.shape[0])
print("Max ecg_id per patient:", g.max())
print("Mean ecg_id per patient:", g.mean())

# Optional: Liste von Patienten mit >1 recording
multi = g[g > 1]
print("Patients with >1 ecg_id:", len(multi))
if len(multi) > 0:
    print(multi.head(10))
