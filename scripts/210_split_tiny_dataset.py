#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
210_split_tiny_dataset.py

Crea particiones reproducibles train/val/test a partir de:
    data/raw/tiny_dataset/<clase>/*.wav

Genera una estructura tipo:
    data/raw/tiny_dataset_split/
        train/<clase>/*.wav
        val/<clase>/*.wav
        test/<clase>/*.wav

Además escribe un CSV con el detalle de los splits:
    data/raw/tiny_dataset_split/split_metadata.csv

Uso básico:
    python scripts/210_split_tiny_dataset.py

Uso con opciones:
    python scripts/210_split_tiny_dataset.py \
        --raw data/raw/tiny_dataset \
        --out data/raw/tiny_dataset_split \
        --train-frac 0.6 --val-frac 0.2 --test-frac 0.2 \
        --seed 42

python scripts/210_split_tiny_dataset.py \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --seed 123
        
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def make_splits(
    raw_root: Path,
    out_root: Path,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    # Comprobaciones básicas
    assert raw_root.exists(), f"No existe la carpeta de entrada: {raw_root}"
    classes = [p for p in sorted(raw_root.iterdir()) if p.is_dir()]
    assert classes, f"No se encontraron subcarpetas de clase en {raw_root}"

    total_frac = train_frac + val_frac + test_frac
    assert abs(total_frac - 1.0) < 1e-6, (
        f"Las fracciones deben sumar 1.0 (train+val+test). "
        f"Actualmente suman {total_frac:.3f}"
    )

    rng = np.random.RandomState(seed)
    rows = []

    print(f"Entrada: {raw_root}")
    print(f"Salida:  {out_root}")
    print(f"Fracciones -> train={train_frac:.2f}, val={val_frac:.2f}, test={test_frac:.2f}")
    print(f"Semilla: {seed}")

    # Creamos carpetas raíz de salida
    for split in ["train", "val", "test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    for cls_dir in classes:
        label = cls_dir.name
        wavs = sorted(cls_dir.glob("*.wav"))
        if not wavs:
            print(f"⚠️  Clase {label} sin WAVs, se omite.")
            continue

        wavs = np.array(wavs)
        idx = np.arange(len(wavs))
        rng.shuffle(idx)
        wavs = wavs[idx]

        n = len(wavs)
        n_train = int(np.floor(n * train_frac))
        n_val = int(np.floor(n * val_frac))
        n_test = n - n_train - n_val  # lo que sobra va a test

        splits = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * n_test
        )
        assert len(splits) == n

        print(f" ▶ Clase {label}: total={n}, train={n_train}, val={n_val}, test={n_test}")

        # Creamos subcarpetas por clase
        for split in ["train", "val", "test"]:
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

        # Copiamos archivos y registramos metadata
        for wav_path, split in zip(wavs, splits):
            dst = out_root / split / label / wav_path.name
            shutil.copy2(wav_path, dst)

            rows.append({
                "orig_path": str(wav_path),
                "new_path": str(dst),
                "label": label,
                "split": split,
            })

    df = pd.DataFrame(rows)
    out_csv = out_root / "split_metadata.csv"
    df.to_csv(out_csv, index=False)
    print(f"CSV de splits guardado en: {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Crea splits train/val/test para tiny_dataset.")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/tiny_dataset"),
                    help="Carpeta raíz con subcarpetas de clase (WAVs).")
    ap.add_argument("--out", type=Path, default=Path("data/raw/tiny_dataset_split"),
                    help="Carpeta raíz de salida con train/val/test.")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla para barajar los WAVs (reproducible).")
    args = ap.parse_args()

    df = make_splits(
        raw_root=args.raw,
        out_root=args.out,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    print("\nResumen por split/clase:")
    print(df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()