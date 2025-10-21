# src/test_plot_snippets.py
import sys
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # serverfreundlich
import matplotlib.pyplot as plt

from pathlib import Path

from . import config
from .experiment import start_run

# Daten & Preprocessing
from .data_loader import load_scp_codes, get_scp_code_list
from .preprocessing import create_snippets, batch_process_relevant_ecgs

# Caching
from .snippet_cache import (
    make_cache_key, try_load_or_build, compute_data_version
)


def snippet_stats(snippet):
    """
    Erwartet:
        1D: (T,) oder 2D: (T, C)
    Gibt Stats pro Snippet (über Zeit) zurück.
    Für 2D wird über alle Kanäle gemittelt (erst pro Kanal, dann Mittel).
    """
    import numpy as np
    if snippet.ndim == 1:
        x = snippet
        mean = float(np.mean(x))
        std = float(np.std(x))
        vmin = float(np.min(x))
        vmax = float(np.max(x))
    else:
        # pro Kanal über Zeit, dann über Kanäle mitteln
        m = np.mean(snippet, axis=0)
        s = np.std(snippet, axis=0)
        mean = float(np.mean(m))
        std = float(np.mean(s))
        vmin = float(np.min(snippet))
        vmax = float(np.max(snippet))
    return mean, std, vmin, vmax

# ---------------------------------------------------------------------
# 1) Run für Artefakte starten
# ---------------------------------------------------------------------
params = {
    "SAMPLING_RATE": getattr(config, "SAMPLING_RATE", None),
    "SNIPPET_LENGTH_BEFORE_R": getattr(config, "SNIPPET_LENGTH_BEFORE_R", None),
    "SNIPPET_LENGTH_AFTER_R": getattr(config, "SNIPPET_LENGTH_AFTER_R", None),
    "DATA_DIR": str(getattr(config, "DATA_DIR", "")),
}
run = start_run(params, base_dir="runs", seed=42)
print(f"Neuer Testlauf: {run.run_dir}")
filename = str(run.run_dir / "_test_plot_snippets.png")

# ---------------------------------------------------------------------
# 2) Relevante Dateien & Labels
# ---------------------------------------------------------------------
print("Suche relevante PTB-XL-Dateien…")
filepath_list = batch_process_relevant_ecgs(
    config.DATA_DIR, config.RELEVANT_ECG_PATH, config.DATABASE_PATH
)
print(f"→ {len(filepath_list)} Dateien gefunden.")

ecg_id_to_scp_str = load_scp_codes(config.RELEVANT_ECG_PATH)
ecg_id_to_scp_list = {ecg_id: get_scp_code_list(s)
                      for ecg_id, s in ecg_id_to_scp_str.items()}

# ---------------------------------------------------------------------
# 3) Snippets aus Cache oder bauen (Teilmenge per MAX_FILES)
# ---------------------------------------------------------------------
MAX_FILES = 1000  # für schnelle Sichtprüfung; volle Menge: len(filepath_list)

cache_params = {
    "sampling_rate":        config.SAMPLING_RATE,
    "before_r":             config.SNIPPET_LENGTH_BEFORE_R,
    "after_r":              config.SNIPPET_LENGTH_AFTER_R,
    "max_files":            MAX_FILES,
}

data_version = compute_data_version(
    Path(config.RELEVANT_ECG_PATH),
    Path(config.DATABASE_PATH) if getattr(config, "DATABASE_PATH", None) else None,
)

key = make_cache_key(cache_params, data_version=data_version)
cache_root = Path("cache") / "snippets"

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

    if not all_snippets:
        raise RuntimeError("Keine Snippets gefunden. Prüfe Pfade/Preprocessing.")

    return (
        np.asarray(all_snippets, dtype=np.float32),
        np.asarray(all_ecg_ids),
        np.asarray(all_scp_labels_raw),
    )

snippets, ecg_ids, labels, meta_dict, npz_path = try_load_or_build(
    cache_root=cache_root,
    key=key,
    build_fn=_build_snippets,
    meta={
        "params": cache_params,
        "data_version": data_version,
        "relevant_path": str(config.RELEVANT_ECG_PATH),
        "database_path": str(config.DATABASE_PATH),
        "n_files": int(min(MAX_FILES, len(filepath_list))),
    },
    verbose=True,
)

all_snippets = snippets
all_ecg_ids = ecg_ids
all_scp_labels_raw = labels
print(f"Snippets geladen: {all_snippets.shape}  (Quelle: {npz_path.name})")

snippet_length, num_channels = all_snippets.shape[1], all_snippets.shape[2]
print(f"Snippets: {all_snippets.shape} (Länge={snippet_length}, Kanäle={num_channels})")
print(f"Anzahl Labels (roh): {len(all_scp_labels_raw)}")

# ---------------------------------------------------------------------
# 3.5) Stichprobe plotten + Stats-Overlay
# ---------------------------------------------------------------------
print("\nBeispielhafte Original-Snippets (mit Stats-Overlay):")

num_examples = min(10, len(all_snippets))
example_indices = np.random.choice(len(all_snippets), size=num_examples, replace=False)

plt.figure(figsize=(12, 2.2 * num_examples))
per_snippet_means = []
per_snippet_stds = []

for i, idx in enumerate(example_indices):
    snippet = all_snippets[idx]
    plt.subplot(num_examples, 1, i + 1)

    # Variante A: nur Lead 0
    if snippet.ndim == 2 and snippet.shape[1] > 0:
        plt.plot(snippet[:, 0], label=f"ECG_ID={all_ecg_ids[idx]}, Label={all_scp_labels_raw[idx]}")
    else:
        plt.plot(snippet, label=f"ECG_ID={all_ecg_ids[idx]}, Label={all_scp_labels_raw[idx]}")

    # Stats berechnen und speichern
    mean, std, vmin, vmax = snippet_stats(snippet)
    per_snippet_means.append(mean)
    per_snippet_stds.append(std)

    # Overlay-Text (oben rechts)
    txt = f"mean={mean:.3f}, std={std:.3f}, min={vmin:.3f}, max={vmax:.3f}"
    ax = plt.gca()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(
        x= x0 + 0.99*(x1 - x0),
        y= y0 + 0.95*(y1 - y0),
        s= txt,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
    )

    plt.legend(loc="upper left", fontsize="x-small")

plt.suptitle("Beispiele: Originale EKG-Snippets (Lead 0) mit Stats", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(filename, dpi=150)
plt.close()

print(f"Plot gespeichert: {filename}")

# ---------------------------------------------------------------------
# Optional: Zusatzplots zur Verteilung von Mittelwerten / Std in der Stichprobe
# ---------------------------------------------------------------------
if len(per_snippet_means) > 1:
    plt.figure(figsize=(10, 4))
    plt.hist(per_snippet_means, bins=20)
    plt.title("Verteilung der Snippet-Mittelwerte (Stichprobe)")
    plt.xlabel("mean")
    plt.ylabel("Häufigkeit")
    out_mean = str(run.run_dir / "_test_snippet_means_hist.png")
    plt.tight_layout()
    plt.savefig(out_mean, dpi=150)
    plt.close()
    print(f"Zusatzplot gespeichert: {out_mean}")

if len(per_snippet_stds) > 1:
    plt.figure(figsize=(10, 4))
    plt.hist(per_snippet_stds, bins=20)
    plt.title("Verteilung der Snippet-Standardabweichungen (Stichprobe)")
    plt.xlabel("std")
    plt.ylabel("Häufigkeit")
    out_std = str(run.run_dir / "_test_snippet_stds_hist.png")
    plt.tight_layout()
    plt.savefig(out_std, dpi=150)
    plt.close()
    print(f"Zusatzplot gespeichert: {out_std}")
