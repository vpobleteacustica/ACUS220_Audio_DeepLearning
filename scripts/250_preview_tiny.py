# scripts/250_preview_tiny.py
# -*- coding: utf-8 -*-
"""
Genera previsualizaciones (grids) de STFT / Log-Mel / MFCC por clase
y una grilla resumen de muestras de todas las clases.

Entrada:
  data/processed/tiny_specs/metadata.csv  (producido por 200_build_tiny_dataset.py)

Salidas:
  data/processed/tiny_specs/_preview_<CLASE>.png
  data/processed/tiny_specs/_preview_all.png
"""

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data/processed/tiny_specs")
META_CSV = DATA_DIR / "metadata.csv"

def load_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    # claves esperadas
    for k in ("stft_db", "mel_db", "mfcc", "meta"):
        if k not in d.files:
            raise KeyError(f"Falta clave '{k}' en {npz_path}")
    stft_db = d["stft_db"]
    mel_db  = d["mel_db"]
    mfcc    = d["mfcc"]
    meta    = json.loads(d["meta"].item())
    return stft_db, mel_db, mfcc, meta

def grid_per_class(df: pd.DataFrame, label: str, k: int = 10, out_png: Path | None = None, seed: int = 0):
    sub = df[df["label"] == label]
    if sub.empty:
        print(f"⚠️  Clase sin muestras: {label}")
        return
    sub = sub.sample(min(k, len(sub)), random_state=seed).reset_index(drop=True)
    n = len(sub)

    fig, axs = plt.subplots(n, 3, figsize=(9, 2.2*n))
    if n == 1:
        axs = np.array([axs])

    for i, (_, r) in enumerate(sub.iterrows()):
        stft_db, mel_db, mfcc, meta = load_npz(Path(r["npz"]))
        axs[i, 0].imshow(stft_db, aspect="auto", origin="lower"); axs[i, 0].set_title("STFT dB")
        axs[i, 1].imshow(mel_db,  aspect="auto", origin="lower"); axs[i, 1].set_title("Log-Mel dB")
        axs[i, 2].imshow(mfcc,    aspect="auto", origin="lower"); axs[i, 2].set_title("MFCC")
        for j in range(3):
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        # Anotar el nombre del archivo original
        axs[i, 0].text(4, 4, Path(meta["wav"]).name, color="w", fontsize=8,
                       bbox=dict(facecolor="k", alpha=0.4))

    plt.suptitle(f"Clase: {label}", y=1.02)
    plt.tight_layout()
    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160)
        print(f"Guardado: {out_png}")
    plt.close(fig)

def grid_all_classes(df: pd.DataFrame, labels: list[str], k: int = 5, out_png: Path | None = None, seed: int = 0):
    # Una fila por clase, cada fila muestra k espectrogramas Log-Mel
    rows = len(labels)
    if rows == 0:
        print("⚠️  No hay clases para mostrar.")
        return
    cols = k
    fig, axs = plt.subplots(rows, cols, figsize=(2.2*cols, 2.0*rows), squeeze=False)

    rng = np.random.default_rng(seed)
    for i, label in enumerate(labels):
        sub = df[df["label"] == label]
        if sub.empty:
            for j in range(cols):
                axs[i, j].axis("off")
            axs[i, 0].set_ylabel(label, rotation=0, ha="right", va="center")
            continue
        take = min(cols, len(sub))
        idxs = rng.choice(len(sub), size=take, replace=False)
        picks = sub.iloc[idxs].reset_index(drop=True)
        for j in range(cols):
            ax = axs[i, j]
            if j < take:
                _, mel_db, _, _ = load_npz(Path(picks.loc[j, "npz"]))
                ax.imshow(mel_db, aspect="auto", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
        axs[i, 0].set_ylabel(label, rotation=0, ha="right", va="center")

    plt.suptitle("Log-Mel por clase", y=1.02)
    plt.tight_layout()
    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160)
        print(f"Guardado: {out_png}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="Imágenes por clase (grids por clase)")
    ap.add_argument("--k_all", type=int, default=5, help="Imágenes por clase en el grid global")
    ap.add_argument("--seed", type=int, default=0, help="Semilla para muestreo")
    args = ap.parse_args()

    assert META_CSV.exists(), f"No existe {META_CSV}. Corre primero: python -m scripts.200_build_tiny_dataset"
    df = pd.read_csv(META_CSV)
    assert not df.empty, "metadata.csv está vacío."

    print("Conteo por clase:")
    print(df["label"].value_counts())

    labels = sorted(df["label"].unique())

    # Grids por clase (STFT / Log-Mel / MFCC)
    for cls in labels:
        grid_per_class(df, cls, k=args.k, seed=args.seed, out_png=DATA_DIR / f"_preview_{cls}.png")

    # Grid global (solo Log-Mel), k_all por clase
    grid_all_classes(df, labels, k=args.k_all, seed=args.seed, out_png=DATA_DIR / "_preview_all.png")

if __name__ == "__main__":
    main()