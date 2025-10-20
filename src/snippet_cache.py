# src/snippet_cache.py
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable, Iterable

import numpy as np

# ---------------------------
# Key/Meta/Datei-Management
# ---------------------------

def _stable_json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _sha16(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]

def make_cache_key(params: Dict[str, Any], data_version: Optional[str] = None) -> str:
    """
    Erzeuge einen stabilen Key aus Preprocessing-Parametern + optionaler Daten-Version.
    'params' sollte alles enthalten, was die Snippet-Generierung beeinflusst
    (Samplingrate, Fensterlängen, Lead-Auswahl, Filter, …).
    """
    payload = dict(params)
    if data_version:
        payload["_data_version"] = data_version
    blob = _stable_json_dumps(payload).encode("utf-8")
    return _sha16(blob)

def get_cache_paths(cache_root: Path, key: str) -> Dict[str, Path]:
    """
    Liefert Pfade für .npz (Snippets), .meta.json (Meta) und .lock (Lock).
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    return {
        "npz": cache_root / f"snippets_{key}.npz",
        "meta": cache_root / f"snippets_{key}.meta.json",
        "lock": cache_root / f"snippets_{key}.lock",
    }

def load_cached_snippets(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return data["snippets"], data["ecg_ids"], data["labels"]

def save_cached_snippets(npz_path: Path,
                         snippets: np.ndarray,
                         ecg_ids: np.ndarray,
                         labels: np.ndarray) -> None:
    # Sparsam und konsistent speichern
    np.savez_compressed(
        npz_path,
        snippets=np.asarray(snippets, dtype=np.float32),
        ecg_ids=np.asarray(ecg_ids),
        labels=np.asarray(labels),
    )

def write_metadata(meta_path: Path, meta: Dict[str, Any]) -> None:
    meta_path.write_text(_stable_json_dumps(meta), encoding="utf-8")

# ---------------------------
# Daten-Version (ändert sich, wenn Eingabedaten sich ändern)
# ---------------------------

def _dir_signature(path: Path, pattern: str = "**/*") -> str:
    """
    Sehr günstige Heuristik: zählt Dateien und nimmt max-mtime.
    Reicht für „ändert sich, ja/nein“ bei großen Datenordnern.
    """
    if not path.exists():
        return "missing"
    n = 0
    latest_mtime = 0.0
    for p in path.glob(pattern):
        if p.is_file():
            n += 1
            try:
                latest_mtime = max(latest_mtime, p.stat().st_mtime)
            except Exception:
                pass
    return f"n{n}_t{int(latest_mtime)}"

def compute_data_version(
    relevant_path: Path,
    database_path: Optional[Path] = None,
) -> str:
    """
    Kombiniert Signaturen relevanter Eingabeorte (z. B. RELEVANT_ECG_PATH, DATABASE_PATH).
    Nutze das Ergebnis als 'data_version' in make_cache_key(...).
    """
    parts = []
    parts.append(("relevant", _dir_signature(relevant_path)))
    if database_path:
        parts.append(("db", _dir_signature(database_path)))
    payload = _stable_json_dumps(dict(parts)).encode("utf-8")
    return _sha16(payload)

# ---------------------------
# Locking (einfach & robust)
# ---------------------------

class FileLock:
    def __init__(self, lock_path: Path, poll_interval: float = 0.2, timeout: float = 600.0):
        self.lock_path = lock_path
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._acquired = False

    def acquire(self):
        start = time.time()
        while True:
            try:
                # O_EXCL sorgt dafür, dass die Erstellung fehlschlägt, wenn Datei bereits existiert
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self._acquired = True
                return
            except FileExistsError:
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Timeout beim Warten auf Lock: {self.lock_path}")
                time.sleep(self.poll_interval)

    def release(self):
        if self._acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except Exception:
                pass
        self._acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

# ---------------------------
# High-Level Helper
# ---------------------------

def try_load_or_build(
    cache_root: Path,
    key: str,
    build_fn: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    meta: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Path]:
    """
    Versucht, Snippets aus dem Cache zu laden; falls nicht vorhanden, ruft 'build_fn'
    (dein Code, der Snippets erstellt) auf, speichert Ergebnis + Meta und liefert alles zurück.

    Rückgabe:
      snippets, ecg_ids, labels, meta_dict, npz_path
    """
    paths = get_cache_paths(cache_root, key)
    if paths["npz"].exists():
        if verbose:
            print(f"🔄 Lade Snippets aus Cache: {paths['npz'].name}")
        snippets, ecg_ids, labels = load_cached_snippets(paths["npz"])
        meta_dict = meta or {}
        # ggf. vorhandene Meta laden und mergen
        if paths["meta"].exists():
            try:
                existing = json.loads(paths["meta"].read_text(encoding="utf-8"))
                meta_dict = {**existing, **meta_dict}
            except Exception:
                pass
        return snippets, ecg_ids, labels, meta_dict, paths["npz"]

    # sonst bauen (mit Lock gegen Parallel-Schreiben)
    if verbose:
        print(f"Baue Snippets (kein Cache für key={key}) …")
    with FileLock(paths["lock"]):
        # Check: vielleicht hat ein anderer Prozess gerade fertig geschrieben
        if paths["npz"].exists():
            if verbose:
                print(f"Cache inzwischen vorhanden, lade: {paths['npz'].name}")
            snippets, ecg_ids, labels = load_cached_snippets(paths["npz"])
            meta_dict = meta or {}
            if paths["meta"].exists():
                try:
                    existing = json.loads(paths["meta"].read_text(encoding="utf-8"))
                    meta_dict = {**existing, **meta_dict}
                except Exception:
                    pass
            return snippets, ecg_ids, labels, meta_dict, paths["npz"]

        # wirklich erstellen
        snippets, ecg_ids, labels = build_fn()
        save_cached_snippets(paths["npz"], snippets, ecg_ids, labels)
        meta_dict = meta or {}
        meta_dict.update({
            "key": key,
            "n_snippets": int(len(snippets)),
            "snippet_shape": list(snippets.shape[1:]) if len(snippets) > 0 else None,
            "dtype": "float32",
            "created_at": int(time.time()),
        })
        write_metadata(paths["meta"], meta_dict)
        if verbose:
            print(f"Snippets gespeichert: {paths['npz'].name}  (n={len(snippets)})")
        return snippets, ecg_ids, labels, meta_dict, paths["npz"]
