#!/usr/bin/env python3
# scripts/360_infer_batch.py
# -*- coding: utf-8 -*-
"""
Inferencia en BATCH usando el mejor modelo disponible
o una semilla específica.

Genera:
- CSV con predicciones
- PNG por cada audio (espectrograma + predicción)
- Gráfico resumen de desempeño (conteo de clases predichas)

EJEMPLOS DE USO (desde la carpeta raíz del repo):

  # 1) Usar el MEJOR modelo según run_summary.csv
  python scripts/360_infer_batch.py \
      --wav-dir data/raw/tiny_dataset/Batrachyla_taeniata

  # 2) Forzar una semilla concreta
  python scripts/360_infer_batch.py \
      --wav-dir data/raw/tiny_dataset/Batrachyla_taeniata \
      --seed 42

  # 3) Ejemplo para LAS 5 ESPECIES
  python scripts/360_infer_batch.py --wav-dir data/raw/tiny_dataset/Batrachyla_leptopus
  python scripts/360_infer_batch.py --wav-dir data/raw/tiny_dataset/Batrachyla_taeniata
  python scripts/360_infer_batch.py --wav-dir data/raw/tiny_dataset/Calyptocephalella_gayi
  python scripts/360_infer_batch.py --wav-dir data/raw/tiny_dataset/Pleurodema_thaul
  python scripts/360_infer_batch.py --wav-dir data/raw/tiny_dataset/Porzana_spiloptera

También puedes llamarlo como módulo (equivalente, recomendado):
  python -m scripts.360_infer_batch --wav-dir data/raw/tiny_dataset/Batrachyla_leptopus
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Bootstrap: añadir la RAÍZ del proyecto al sys.path
# Esto permite hacer "from src...." incluso si ejecutamos:
#   python scripts/360_infer_batch.py ...
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # ACUS220_Audio_DeepLearning
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- imports normales ------------------------------------------------
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.utils_audio import load_wav, pad_or_trim
from src.features import compute_logmel_db

DATA_DIR = Path("data/processed/tiny_specs")
RUN_ROOT = DATA_DIR / "cnn_run"
RUN_SUMMARY = RUN_ROOT / "run_summary.csv"


# --- MODELO ----------------------------------------------------------

class TinyMelCNN(nn.Module):
    """Misma arquitectura usada en 300_train_tiny_cnn."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# --- UTILIDAD: elegir mejor modelo ----------------------------------

def pick_best_seed() -> tuple[int, Path]:
    """Lee run_summary.csv y devuelve (seed, ruta_mejor_modelo)."""
    assert RUN_SUMMARY.exists(), (
        f"No existe {RUN_SUMMARY}. Ejecuta antes 300_train_tiny_cnn "
        "para generar los modelos y el resumen."
    )
    df = pd.read_csv(RUN_SUMMARY)
    df = df.sort_values("test_acc", ascending=False)
    row = df.iloc[0]
    return int(row["seed"]), Path(row["best_model"])


# --- GRÁFICO RESUMEN -------------------------------------------------

def plot_summary(df: pd.DataFrame, class_names, out_png: Path) -> None:
    """
    Grafica cuántas veces se predijo cada clase.
    Útil para ver si el modelo se 'inclina' hacia alguna.
    """
    counts = df["pred_label"].value_counts().reindex(class_names).fillna(0)

    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", edgecolor="k")
    plt.ylabel("Número de predicciones")
    plt.xlabel("Clase predicha")
    plt.title("Distribución de clases predichas")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"Guardado gráfico resumen: {out_png}")


# --- MAIN ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Inferencia en lote sobre una carpeta de WAVs."
    )
    ap.add_argument("--wav-dir", type=str, required=True,
                    help="Carpeta con archivos .wav a clasificar")
    ap.add_argument("--seed", type=int, default=None,
                    help="Si se omite, se usa el mejor modelo de run_summary.csv")
    ap.add_argument("--sr", type=int, default=22050,
                    help="Frecuencia de muestreo de trabajo (Hz)")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="Duración fija de los ejemplos (s)")
    ap.add_argument("--n_mels", type=int, default=64,
                    help="Número de bandas Mel para el log-Mel")
    args = ap.parse_args()

    # Carpeta input ----------------------------------------------------
    wav_dir = Path(args.wav_dir)
    assert wav_dir.exists(), f"No existe {wav_dir}"
    wavs = sorted(wav_dir.glob("*.wav"))
    assert wavs, f"No hay archivos .wav en {wav_dir}"

    # Sufijo basado en la carpeta (ej. 'Batrachyla_taeniata')
    suffix = wav_dir.name.replace(" ", "_")

    # Selección del modelo --------------------------------------------
    if args.seed is None:
        seed, model_path = pick_best_seed()
        print(f"🔍 Usando MEJOR modelo según run_summary.csv (seed={seed})")
    else:
        seed = args.seed
        model_path = RUN_ROOT / f"seed_{seed}" / "best_model.pt"
        assert model_path.exists(), (
            f"No existe {model_path}. "
            "¿Ejecutaste 300_train_tiny_cnn con esa seed?"
        )
        print(f"Usando modelo de la seed fija {seed}")

    # Clases usadas en el entrenamiento
    class_file = model_path.parent / "class_names.json"
    assert class_file.exists(), f"Falta {class_file}"
    labels = json.loads(class_file.read_text())

    # Cargar modelo en CPU o GPU --------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMelCNN(len(labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Carpetas de salida ----------------------------------------------
    out_dir = model_path.parent
    preview_dir = out_dir / f"inference_previews_{suffix}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Proceso sobre todos los WAV -------------------------------------
    for wav_path in wavs:
        # 1) Cargar y ajustar a duración fija
        y, _ = load_wav(wav_path, sr=args.sr)
        y = pad_or_trim(y, int(args.sr * args.duration))

        # 2) Log-Mel
        mel_db = compute_logmel_db(y, args.sr, n_mels=args.n_mels)
        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # 3) Inferencia
        with torch.no_grad():
            probs = model(x).softmax(dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        top_p = float(probs[top_idx])

        # 4) Guardar visual (espectrograma + predicción) ---------------
        fig = plt.figure(figsize=(5, 3))
        plt.imshow(mel_db, aspect="auto", origin="lower")
        plt.title(f"{top_label} ({top_p:.2f})")
        plt.xticks([])
        plt.yticks([])
        out_png = preview_dir / f"{wav_path.stem}_pred.png"
        plt.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close()

        # 5) Registrar fila para el CSV --------------------------------
        rows.append({
            "wav": str(wav_path),
            "pred_label": top_label,
            "pred_prob": top_p,
            "seed": seed,
            "model_path": str(model_path),
            "preview_png": str(out_png),
        })

    # CSV de salida ----------------------------------------------------
    df = pd.DataFrame(rows)
    out_csv = out_dir / f"inference_results_{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"CSV guardado: {out_csv}")

    # Gráfico resumen --------------------------------------------------
    summary_png = out_dir / f"summary_{suffix}.png"
    plot_summary(df, labels, summary_png)

    print("Inferencia batch completada.")


if __name__ == "__main__":
    main()