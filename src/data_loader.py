# src/data_loader.py 

import pandas as pd
import os
import ast
from .config import  RELEVANT_ECG_PATH

# ---- Laden der SCP-Codes ---- 
"""Lädt die SCP-Codes aus der CSV und gibt sie als Dictionary zurück."""
def load_scp_codes(filepath):
    df_scp_codes = pd.read_csv(RELEVANT_ECG_PATH)
    ecg_id_to_scp_str = pd.Series(df_scp_codes['relevante_scp_codes'].values, index=df_scp_codes['ecg_id']).to_dict()
    return ecg_id_to_scp_str

def get_scp_code_list(scp_string):
    """Extrahiert die Liste der SCP-Codes aus dem String im Dictionary-Format."""
    print(f"Verarbeite SCP-String: '{scp_string}'")
    try:
        scp_dict = ast.literal_eval(scp_string)
        print(f"Nach literal_eval: {scp_dict}, Typ: {type(scp_dict)}")
        if isinstance(scp_dict, dict):
            # Extrahiere die Schlüssel (die SCP-Codes) und sortiere sie
            scp_codes = sorted(list(scp_dict.keys()))
            print(f"Extrahierte SCP-Codes: {scp_codes}")
            return scp_codes
        else:
            print(f"Unerwartetes Format nach literal_eval: {scp_dict}")
            return []
    except (SyntaxError, ValueError):
        print(f"Fehler bei literal_eval für: '{scp_string}'")
        return []
    
# Beispiel für die Verwendung im Modul (wird nicht ausgeführt, wenn importiert)
if __name__ == '__main__':
    ecg_id_to_scp_str = load_scp_codes(RELEVANT_ECG_PATH)
    ecg_id_to_scp_list = {ecg_id: get_scp_code_list(scp_string)
                          for ecg_id, scp_string in ecg_id_to_scp_str.items()}