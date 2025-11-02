#!/usr/bin/env python3
# Convierte data/raw/tiny_dataset/<clase>/*.wav → .npz por archivo y metadata.csv.

from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

from src.utils_audio import load_wav, pad_or_trim
from src.features import compute_stft_db, compute_logmel_db, compute_mfcc, compute_gammatone_db

# --- bootstrap sys.path to project root ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------


def build_dataset(raw_root: Path, out_dir: Path, sr=22050, duration=2.0, n_mels=64, n_mfcc=13):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    classes = [p for p in sorted(raw_root.iterdir()) if p.is_dir()]
    assert classes, f"No hay carpetas de clase en {raw_root}"

    for cls_dir in classes:
        label = cls_dir.name
        npz_dir = out_dir / label
        npz_dir.mkdir(parents=True, exist_ok=True)

        for wav in sorted(cls_dir.glob("*.wav")):
            npz_path = npz_dir / (wav.stem + ".npz")
            if not npz_path.exists():
                y, _ = load_wav(wav, sr=sr)
                y = pad_or_trim(y, int(sr*duration))
                stft_db = compute_stft_db(y, sr)
                mel_db  = compute_logmel_db(y, sr, n_mels=n_mels)
                mfcc    = compute_mfcc(y, sr, n_mfcc=n_mfcc, n_mels=n_mels)
                gammatone_db = compute_gammatone_db(y, sr, n_bins=n_mels)  # puede ser None

                if gammatone_db is None:
                    np.savez_compressed(npz_path,
                        stft_db=stft_db.astype(np.float32),
                        mel_db =mel_db.astype(np.float32),
                        mfcc   =mfcc.astype(np.float32),
                        meta   =json.dumps({"wav": str(wav), "label": label})
                    )
                else:
                    np.savez_compressed(npz_path,
                        stft_db=stft_db.astype(np.float32),
                        mel_db =mel_db.astype(np.float32),
                        mfcc   =mfcc.astype(np.float32),
                        gammatone_db=gammatone_db.astype(np.float32),
                        meta   =json.dumps({"wav": str(wav), "label": label})
                    )
            rows.append({"wav": str(wav), "npz": str(npz_path), "label": label})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Construye mini-dataset NPZ desde WAVs.")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/tiny_dataset"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/tiny_specs"))
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_mfcc", type=int, default=13)
    args = ap.parse_args()

    print(f"🔧 Construyendo desde {args.raw} -> {args.out}")
    df = build_dataset(args.raw, args.out, args.sr, args.duration, args.n_mels, args.n_mfcc)
    print("Listo. metadata.csv:", args.out / "metadata.csv")
    print(df["label"].value_counts())