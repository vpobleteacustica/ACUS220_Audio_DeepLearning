#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
250_preview_tiny.py

Visualiza un mosaico de espectrogramas (p.ej. Log-Mel) del tiny_dataset
para usar en la clase ACUS220.

- Lee data/processed/tiny_specs/metadata.csv
- Toma N ejemplos por clase
- Aplica:
    * clipping dinámico en dB (max-80)
    * recorte inteligente de columnas silenciosas
- Dibuja un mosaico especie × ejemplos con estética tipo paper.
"""

from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- bootstrap sys.path a la raíz del repo ---
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------


def prepare_for_plot(spec: np.ndarray,
                     dynamic_range_db: float = 80.0) -> tuple[np.ndarray, float, float]:
    """
    Prepara un espectrograma para visualización:

    1) Clipping dinámico en rango [max - dynamic_range_db, max] dB.
    2) Recorte inteligente de columnas silenciosas:
       - Calcula energía por columna.
       - Umbral adaptativo con percentil 10.
       - Si el caso es extremo (casi todo silencio), aplica un fallback
         usando un umbral relativo al máximo.

    Devuelve:
        spec_vis : np.ndarray   -> espectrograma listo para imshow
        vmin, vmax : floats     -> límites de color (dB)
    """
    # 1) Clipping dinámico
    spec_max = float(spec.max())
    vmin = spec_max - dynamic_range_db
    vmax = spec_max
    spec_clipped = np.clip(spec, vmin, vmax)

    # 2) Energía por columna
    col_energy = spec_clipped.mean(axis=0)

    # Umbral adaptativo
    thr = np.percentile(col_energy, 10)
    mask = col_energy > thr

    # Fallback si casi todo es silencio
    if mask.sum() < 5:
        thr2 = col_energy.max() - 60.0
        mask = col_energy > thr2
        if not np.any(mask):
            # No hay nada que recortar de forma razonable
            return spec_clipped, vmin, vmax

    spec_vis = spec_clipped[:, mask]
    return spec_vis, vmin, vmax


def main(args: argparse.Namespace) -> None:
    meta_path: Path = args.metadata
    df = pd.read_csv(meta_path)

    if "npz" not in df.columns:
        raise ValueError(f"metadata.csv no tiene columna 'npz': {meta_path}")

    # Ordenar las clases para que la figura sea estable entre ejecuciones
    labels = sorted(df["label"].unique())
    n_classes = len(labels)
    n_cols = args.n_per_class

    # Figura: algo alta y estilizada
    fig, axes = plt.subplots(
        n_classes,
        n_cols,
        figsize=(n_cols * 2.2, n_classes * 2.2),
        dpi=200,
        squeeze=False,
        constrained_layout=True,
    )

    last_im = None  # para el colorbar

    for i, label in enumerate(labels):
        sub = df[df["label"] == label].copy()

        # Tomamos los primeros n_per_class (tiny dataset, no hace falta random)
        sub = sub.head(n_cols)

        for j in range(n_cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])

            if j >= len(sub):
                # No hay suficientes muestras; dejamos el eje vacío
                ax.axis("off")
                continue

            row = sub.iloc[j]
            npz_path = Path(row["npz"])

            if not npz_path.exists():
                ax.axis("off")
                continue

            data = np.load(npz_path, allow_pickle=True)

            if args.feature not in data:
                raise KeyError(
                    f"El archivo {npz_path} no contiene la clave '{args.feature}'. "
                    f"Claves disponibles: {list(data.keys())}"
                )

            spec = data[args.feature]  # (n_mels, T) o similar

            # Preprocesado para visualización
            spec_vis, vmin, vmax = prepare_for_plot(spec)

            im = ax.imshow(
                spec_vis,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            last_im = im

            # Solo ponemos nombre de la especie en la primera columna
            if j == 0:
                ax.set_ylabel(label.replace("_", " "), fontsize=9)

    # Título general
    fig.suptitle(
        f"Ejemplos de {args.feature} por especie (tiny_dataset)",
        fontsize=12,
        y=1.02,
    )

    # Un solo colorbar para toda la figura
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            fraction=0.015,
            pad=0.01,
        )
        cbar.set_label("Nivel (dB)", fontsize=9)

    out_path = args.out
    fig.savefig(out_path, bbox_inches="tight")
    print(f"✅ Figura guardada en: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preview de espectrogramas del tiny_dataset.")
    ap.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/tiny_specs/metadata.csv"),
        help="Ruta a metadata.csv",
    )
    ap.add_argument(
        "--feature",
        type=str,
        default="mel_db",
        help="Clave del .npz a visualizar (mel_db, stft_db, gammatone_db, etc.)",
    )
    ap.add_argument(
        "--n-per-class",
        type=int,
        default=5,
        help="Número de ejemplos por clase a mostrar.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("figures/tiny_specs_preview.png"),
        help="Ruta de salida para la figura.",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    main(args)