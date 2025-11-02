#!/usr/bin/env python3

# --- bootstrap sys.path to project root ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------

from pathlib import Path
import importlib, sys
print("Chequeo de entorno ACUS220")

pkgs = ["numpy","pandas","matplotlib","librosa","soundfile","torch","sklearn"]
missing = []
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "OK")
        print(f"  - {p:<10} {v}")
    except Exception:
        missing.append(p)
if missing:
    print("\n Faltan paquetes:", ", ".join(missing))
    sys.exit(1)

# Comprobación rápida de CUDA (si existe)
try:
    import torch
    print(f"\n torch.cuda.is_available(): {torch.cuda.is_available()}")
except Exception:
    pass

# Verificación de formas esperadas con una senoidal
import numpy as np
from src.features import compute_stft_db, compute_logmel_db, compute_mfcc
sr, dur = 22050, 2.0
t = np.linspace(0, dur, int(sr*dur), endpoint=False, dtype=np.float32)
y = 0.1*np.sin(2*np.pi*440*t).astype(np.float32)
stft = compute_stft_db(y, sr)
mel  = compute_logmel_db(y, sr, n_mels=64)
mfcc = compute_mfcc(y, sr, n_mfcc=13, n_mels=64)
print(f"\n Shapes -> STFT {stft.shape} | Log-Mel {mel.shape} | MFCC {mfcc.shape}")
print("\n Entorno OK.")