# src/experiment.py
from __future__ import annotations
import os, json, time, subprocess, random
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Optional
import numpy as np
import tensorflow as tf

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time()%1)*1000):03d}_p{os.getpid()}"

def _short_tag(params: Dict[str, Any], keys=("LATENT_DIMENSIONS", "LATENT_DIM", "NUM_EMBEDDINGS","COMMITMENT_COST","LEARNING_RATE","AE_BATCH_SIZE")) -> str:
    def fmt(k, v):
        if v is None: return ""
        if isinstance(v, float):
            # kurze Schreibweise für LR etc.
            if "LEARNING_RATE" in k: 
                return f"LR{v:.0e}"
            return f"{k[:1]}{v}".replace(".", "_")
        return f"{k[:1]}{v}"
    parts = []
    for k in keys:
        if k in params:
            parts.append(fmt(k, params[k]))
    return "_".join(p for p in parts if p)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_serializable(obj):
    if is_dataclass(obj): 
        return asdict(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def snapshot_params(params: Dict[str, Any], run_dir: Path):
    # JSON + TXT
    with open(run_dir / "params.json", "w") as f:
        json.dump({k: to_serializable(v) for k, v in params.items()}, f, indent=2)
    with open(run_dir / "params.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

def snapshot_git(run_dir: Path):
    try:
        rev = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.STDOUT).decode().strip()
        diff = subprocess.check_output(["git","status","--porcelain"], stderr=subprocess.STDOUT).decode()
        with open(run_dir / "git.txt", "w") as f:
            f.write(f"commit: {rev}\n")
            f.write("status:\n")
            f.write(diff)
    except Exception:
        # nicht kritisch
        pass

def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class RunContext:
    """Kapselt Pfade + Keras-Callbacks für einen Trainingslauf."""
    def __init__(self, base_dir: Path, params: Dict[str, Any], seed: int = 42):
        set_seeds(seed)
        tag = f"{_now_tag()}__{_short_tag(params)}" if params else _now_tag()
        self.run_dir = ensure_dir(base_dir / tag)
        self.ckpt_best = self.run_dir / "best.h5"
        self.ckpt_last = self.run_dir / "last.h5"
        self.csv_path = self.run_dir / "metrics.csv"
        self.tb_dir = ensure_dir(self.run_dir / "tb")
        # Snapshots
        snapshot_params({"seed": seed, **params}, self.run_dir)
        snapshot_git(self.run_dir)

    def callbacks(self, monitor: str = "val_loss", patience: int = 8):
        cbs = [
            tf.keras.callbacks.CSVLogger(str(self.csv_path), append=False),
            tf.keras.callbacks.ModelCheckpoint(str(self.ckpt_best), monitor=monitor, save_best_only=True, save_weights_only=False),
            tf.keras.callbacks.ModelCheckpoint(str(self.ckpt_last), monitor=monitor, save_best_only=False, save_weights_only=False),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=str(self.tb_dir)),
        ]
        return cbs

def start_run(params: Dict[str, Any], base_dir: str | Path = "runs", seed: int = 42) -> RunContext:
    base = ensure_dir(Path(base_dir))
    return RunContext(base, params, seed=seed)
