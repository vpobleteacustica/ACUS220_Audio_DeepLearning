#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
350_infer_one.py
-----------------
Inferencia sobre UNA grabación de audio (.wav) usando la CNN entrenada.

Uso básico:
    python -m scripts.350_infer_one --wav path/to/audio.wav

Usar una semilla específica:
    python -m scripts.350_infer_one --wav file.wav --seed 42

Este script:
1. Carga el modelo correspondiente a la mejor semilla (según run_summary.csv)
   o a la semilla indicada por el usuario.
2. Carga un WAV, lo normaliza a 2 segundos y lo convierte a log-Mel.
3. Ejecuta la predicción.
4. Guarda una figura PNG con el espectrograma y la clase predicha.
"""

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.utils_audio import load_wav, pad_or_trim
from src.features import compute_logmel_db


# ================================
# CONFIGURACIÓN GLOBAL DEL PROYECTO
# ================================
DATA_DIR = Path("data/processed/tiny_specs")
RUN_ROOT = DATA_DIR / "cnn_run"
RUN_SUMMARY = RUN_ROOT / "run_summary.csv"


# ================================
# DEFINICIÓN DEL MODELO (TinyMelCNN)
# IMPORTANTE: Debe coincidir EXACTAMENTE con el usado en entrenamiento
# ================================
class TinyMelCNN(nn.Module):
    """
    Pequeña CNN para log-Mel (idéntica a la usada en scripts/300_train_tiny_cnn.py).
    """
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# FUNCIONES AUXILIARES
# ================================
def pick_best_seed():
    """
    Selecciona la mejor semilla según el archivo run_summary.csv.
    (usa test_acc).
    """
    import pandas as pd

    if not RUN_SUMMARY.exists():
        raise FileNotFoundError(f"No existe {RUN_SUMMARY}. Debes entrenar antes.")

    df = pd.read_csv(RUN_SUMMARY)
    df = df.sort_values(by="test_acc", ascending=False)

    best_row = df.iloc[0]
    seed = int(best_row["seed"])
    model_path = Path(best_row["best_model"])
    return seed, model_path


# ================================
# PROCESO DE INFERENCIA
# ================================
def main():
    parser = argparse.ArgumentParser(description="Clasifica un archivo WAV usando TinyMelCNN.")
    parser.add_argument("--wav", type=str, required=True, help="Ruta al archivo WAV.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla específica.")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--n_mels", type=int, default=64)
    args = parser.parse_args()

    # -------------------------
    # Selección del modelo
    # -------------------------
    if args.seed is None:
        seed, model_path = pick_best_seed()
        print(f"🔍 Usando la MEJOR semilla según run_summary.csv → seed={seed}")
    else:
        seed = args.seed
        model_path = RUN_ROOT / f"seed_{seed}" / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"No existe el modelo: {model_path}")
        print(f"🔍 Usando semilla forzada: seed={seed}")

    # Clases
    class_file = model_path.parent / "class_names.json"
    if not class_file.exists():
        raise FileNotFoundError(f"Falta archivo de clases: {class_file}")

    labels = json.loads(class_file.read_text())

    # -------------------------
    # Cargar modelo y enviar a CPU/GPU
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMelCNN(len(labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # -------------------------
    # Procesamiento del WAV → Log-Mel
    # -------------------------
    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"No existe WAV: {wav_path}")

    y, _ = load_wav(wav_path, sr=args.sr)
    y = pad_or_trim(y, int(args.sr * args.duration))

    mel_db = compute_logmel_db(y, args.sr, n_mels=args.n_mels)

    # Normalización simple (igual que entrenamiento)
    mel_db_norm = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    # Tensor: [1, 1, n_mels, T]
    x = torch.tensor(mel_db_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # -------------------------
    # Forward y predicción
    # -------------------------
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    pred_label = labels[idx]
    pred_prob = float(probs[idx])

    # -------------------------
    # Guardar imagen del espectrograma
    # -------------------------
    out_png = model_path.parent / f"infer_{wav_path.stem}.png"

    plt.figure(figsize=(6, 3))
    plt.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    plt.title(f"Pred: {pred_label} (p={pred_prob:.2f})")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    # -------------------------
    # Resumen final
    # -------------------------
    print("\n============================")
    print("RESULTADO DE INFERENCIA")
    print("============================")
    print(f"WAV: {wav_path}")
    print(f"Modelo: {model_path}")
    print(f"Predicción: {pred_label} (p={pred_prob:.3f})")
    print(f"Imagen guardada en: {out_png}")


if __name__ == "__main__":
    main()