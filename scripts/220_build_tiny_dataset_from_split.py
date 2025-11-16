#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
220_build_tiny_dataset_from_split.py

Lee los WAV ya particionados en:
    data/raw/tiny_dataset_split/{train,val,test}/<clase>/*.wav

y construye un dataset de features en:
    data/processed/tiny_specs_split/

Estructura salida:
    data/processed/tiny_specs_split/
        train/<clase>/*.npz
        val/<clase>/*.npz
        test/<clase>/*.npz
        metadata_split.csv   # columnas: wav, npz, label, split

Uso básico:
    python scripts/220_build_tiny_dataset_from_split.py

Con opciones:
    python scripts/220_build_tiny_dataset_from_split.py \
        --raw-split data/raw/tiny_dataset_split \
        --out data/processed/tiny_specs_split \
        --sr 22050 --duration 2.0 --n-mels 64 --n-mfcc 13
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Bootstrap para poder importar src/ desde scripts/ ---
ROOT = Path(__file__).resolve().parents[1]  # carpeta ACUS220_Audio_DeepLearning
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------

from src.utils_audio import load_wav, pad_or_trim
from src.features import (
    compute_stft_db,
    compute_logmel_db,
    compute_mfcc,
    compute_gammatone_db,
)


def build_from_split(
    raw_split_root: Path,
    out_root: Path,
    sr: int = 22050,
    duration: float = 2.0,
    n_mels: int = 64,
    n_mfcc: int = 13,
) -> pd.DataFrame:
    """
    Recorre data/raw/tiny_dataset_split/{train,val,test}/<clase>/*.wav
    y genera .npz con features + un CSV metadata_split.csv.
    """
    assert raw_split_root.exists(), f"No existe {raw_split_root}"
    splits = ["train", "val", "test"]

    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    print(f"Entrada (splits WAV): {raw_split_root}")
    print(f"Salida (features):    {out_root}")
    print(f" sr={sr}, duration={duration}s, n_mels={n_mels}, n_mfcc={n_mfcc}")

    for split in splits:
        split_dir = raw_split_root / split
        if not split_dir.exists():
            print(f"⚠️  Split {split} no existe en {raw_split_root}, se omite.")
            continue

        class_dirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
        if not class_dirs:
            print(f"Split {split} sin clases, se omite.")
            continue

        for cls_dir in class_dirs:
            label = cls_dir.name
            wavs = sorted(cls_dir.glob("*.wav"))
            if not wavs:
                print(f"Split {split}, clase {label} sin WAVs.")
                continue

            npz_cls_dir = out_root / split / label
            npz_cls_dir.mkdir(parents=True, exist_ok=True)

            print(f"{split} / {label}: {len(wavs)} archivos")

            for wav_path in wavs:
                npz_path = npz_cls_dir / f"{wav_path.stem}.npz"

                if not npz_path.exists():
                    # 1) cargar y ajustar longitud
                    y, _ = load_wav(wav_path, sr=sr)
                    y = pad_or_trim(y, int(sr * duration))

                    # 2) extraer features
                    stft_db = compute_stft_db(y, sr)
                    mel_db = compute_logmel_db(y, sr, n_mels=n_mels)
                    mfcc = compute_mfcc(y, sr, n_mfcc=n_mfcc, n_mels=n_mels)
                    gammatone_db = compute_gammatone_db(y, sr, n_bins=n_mels)

                    # 3) armar diccionario a guardar
                    save_kwargs = dict(
                        stft_db=stft_db.astype(np.float32),
                        mel_db=mel_db.astype(np.float32),
                        mfcc=mfcc.astype(np.float32),
                        meta=json.dumps(
                            {
                                "wav": str(wav_path),
                                "label": label,
                                "split": split,
                            }
                        ),
                    )
                    if gammatone_db is not None:
                        save_kwargs["gammatone_db"] = gammatone_db.astype(np.float32)

                    np.savez_compressed(npz_path, **save_kwargs)

                rows.append(
                    {
                        "wav": str(wav_path),
                        "npz": str(npz_path),
                        "label": label,
                        "split": split,
                    }
                )

    df = pd.DataFrame(rows)
    meta_path = out_root / "metadata_split.csv"
    df.to_csv(meta_path, index=False)
    print(f"\n metadata_split.csv guardado en: {meta_path}")

    print("\nResumen por split/clase:")
    if not df.empty:
        print(df.groupby(["split", "label"]).size())
    else:
        print(" No se generaron filas, revisa entradas.")

    return df


def main():
    ap = argparse.ArgumentParser(
        description="Construye features desde tiny_dataset_split (train/val/test)."
    )
    ap.add_argument(
        "--raw-split",
        type=Path,
        default=Path("data/raw/tiny_dataset_split"),
        help="Carpeta con train/val/test/<clase>/*.wav",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/tiny_specs_split"),
        help="Carpeta de salida para los .npz",
    )
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--n-mfcc", type=int, default=13)
    args = ap.parse_args()

    build_from_split(
        raw_split_root=args.raw_split,
        out_root=args.out,
        sr=args.sr,
        duration=args.duration,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
    )


if __name__ == "__main__":
    main()