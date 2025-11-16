#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
300_train_tiny_cnn_split.py

Entrena TinyMelCNN usando el dataset *ya particionado* en train/val/test:

    data/processed/tiny_specs_split/
        train/<clase>/*.npz
        val/<clase>/*.npz
        test/<clase>/*.npz
        metadata_split.csv

Flujo:
  1) Lee metadata_split.csv
  2) Crea DataLoaders para train / val / test
  3) Entrena TinyMelCNN para una o varias semillas
  4) Guarda:
        - best_model.pt (según mejor val_acc)
        - history.csv (por época)
        - class_names.json
        - curvas_loss_acc.png
        - test_metrics.json
        - matrices de confusión (conteos y normalizada)
  5) Actualiza run_summary_split.csv con un resumen por semilla.

Uso básico (una semilla):
    python scripts/300_train_tiny_cnn_split.py --epochs 20 --batch-size 16 --seeds 42

Uso con varias semillas:
    python scripts/300_train_tiny_cnn_split.py --epochs 20 --batch-size 16 --seeds 42,7,123
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Bootstrap para poder importar src/ cuando se ejecuta como script
# --------------------------------------------------------------------
import sys

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_audio import pad_or_trim  # no lo usamos directamente, pero queda de referencia
# (las features ya están construidas; cargamos mel_db desde los .npz)

# --------------------------------------------------------------------
# Paths principales
# --------------------------------------------------------------------

DATA_ROOT = Path("data/processed/tiny_specs_split")
METADATA_CSV = DATA_ROOT / "metadata_split.csv"
RUN_ROOT = DATA_ROOT / "cnn_run_split"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Modelo TinyMelCNN (igual que en inferencia)
# --------------------------------------------------------------------


class TinyMelCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------
# Dataset: carga mel_db desde los .npz
# --------------------------------------------------------------------


class MelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: dict[str, int]):
        """
        df: DataFrame con columnas 'npz' y 'label'
        label2idx: mapeo label -> índice entero
        """
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npz_path = Path(row["npz"])
        label = row["label"]

        data = np.load(npz_path)
        mel_db = data["mel_db"].astype(np.float32)  # (n_mels, T)

        # Normalización simple (0..1 aprox, asumiendo rango -80..0 dB)
        x = (mel_db + 80.0) / 80.0

        # Añadimos dimensión de canal: (1, n_mels, T)
        x = np.expand_dims(x, axis=0)

        y = self.label2idx[label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# --------------------------------------------------------------------
# Utilidades varias
# --------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Opcional: comportamiento algo más determinista
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)


def train_one_seed(
    seed: int,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    class_names: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int = 0,
) -> dict:
    """
    Entrena un modelo para una semilla y devuelve métricas en un dict.
    """
    print(f"\n========================")
    print(f" Entrenando con seed = {seed}")
    print(f"========================")

    set_seed(seed)

    label2idx = {c: i for i, c in enumerate(class_names)}
    n_classes = len(class_names)

    train_ds = MelDataset(df_train, label2idx)
    val_ds = MelDataset(df_val, label2idx)
    test_ds = MelDataset(df_test, label2idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = TinyMelCNN(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_dir = RUN_ROOT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history_rows = []
    best_val_acc = 0.0
    best_epoch = -1
    best_model_path = run_dir / "best_model.pt"

    # Guardamos nombres de clases (útil para inferencia)
    class_file = run_dir / "class_names.json"
    class_file.write_text(json.dumps(class_names, indent=2))

    for epoch in range(1, epochs + 1):
        # --------- TRAIN -------------
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += compute_accuracy(logits, yb)
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_acc = running_acc / max(1, n_batches)

        # --------- VAL -------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_batches_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item()
                val_acc += compute_accuracy(logits, yb)
                n_batches_val += 1

        val_loss = val_loss / max(1, n_batches_val)
        val_acc = val_acc / max(1, n_batches_val)

        print(
            f"[Seed {seed}] Epoch {epoch:02d}/{epochs} "
            f"- train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # Guardar mejor modelo según val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    # Guardar history.csv
    history_df = pd.DataFrame(history_rows)
    history_csv = run_dir / "history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"Historial guardado en: {history_csv}")
    print(f"Mejor val_acc={best_val_acc:.3f} en epoch {best_epoch}")

    # --------- Graficar curvas ---------
    curves_png = run_dir / "curvas_loss_acc.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss train/val")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy train/val")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(curves_png, dpi=160)
    plt.close(fig)
    print(f"Curvas de entrenamiento guardadas en: {curves_png}")

    # --------- Evaluación en TEST (con mejor modelo) ---------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_loss = 0.0
    test_acc = 0.0
    n_batches_test = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            test_loss += loss.item()
            test_acc += compute_accuracy(logits, yb)
            n_batches_test += 1

            preds = logits.argmax(dim=1)
            all_targets.extend(yb.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    test_loss = test_loss / max(1, n_batches_test)
    test_acc = test_acc / max(1, n_batches_test)

    print(f"[Seed {seed}] TEST - loss={test_loss:.4f}, acc={test_acc:.3f}")

    # ----------------------------------------------------------------
    # Matrices de confusión: conteos + normalizada (proporciones)
    # ----------------------------------------------------------------
    from sklearn.metrics import confusion_matrix

    # Conteos
    cm_counts = confusion_matrix(all_targets, all_preds, labels=list(range(n_classes)))

    # Normalizada por fila (proporción por clase real)
    cm_norm = cm_counts.astype(np.float32)
    cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)

    # DataFrames
    df_counts = pd.DataFrame(cm_counts, index=class_names, columns=class_names)
    df_norm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

    # CSVs
    cm_csv = run_dir / "confusion_matrix.csv"               # compatibilidad: conteos
    cm_counts_csv = run_dir / "confusion_matrix_counts.csv" # conteos explícito
    cm_norm_csv = run_dir / "confusion_matrix_norm.csv"     # proporciones

    df_counts.to_csv(cm_csv)
    df_counts.to_csv(cm_counts_csv)
    df_norm.to_csv(cm_norm_csv)

    print(f"Matriz de confusión (conteos) guardada en: {cm_csv}")
    print(f"Matriz de confusión (conteos) guardada en: {cm_counts_csv}")
    print(f"Matriz de confusión (normalizada) guardada en: {cm_norm_csv}")

    # Figura usando la matriz normalizada
    cm_png = run_dir / "confusion_matrix.png"
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)

    cbar = plt.colorbar(im, ax=ax_cm)
    cbar.set_label("Proporción")

    ax_cm.set_xticks(np.arange(n_classes))
    ax_cm.set_yticks(np.arange(n_classes))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_cm.set_yticklabels(class_names)
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Etiqueta real")
    ax_cm.set_title("Matriz de confusión (test, proporciones)")

    # Escribir valor numérico en cada celda (probabilidad)
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_norm[i, j]
            text = f"{value:.2f}"
            color = "white" if value > 0.5 else "black"
            ax_cm.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    fig_cm.savefig(cm_png, dpi=160)
    plt.close(fig_cm)
    print(f"Figura de matriz de confusión guardada en: {cm_png}")

    # Guardamos métricas de test en JSON
    test_metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "best_model": str(best_model_path),
    }
    metrics_json = run_dir / "test_metrics.json"
    metrics_json.write_text(json.dumps(test_metrics, indent=2))
    print(f"Métricas de test guardadas en: {metrics_json}")

    return test_metrics


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Entrena TinyMelCNN usando tiny_specs_split (train/val/test).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Lista de semillas separadas por coma, ej: '42,7,123'",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    assert METADATA_CSV.exists(), f"No existe {METADATA_CSV}. Ejecuta antes 210_split_... y 220_build_..."

    df = pd.read_csv(METADATA_CSV)
    # Nos aseguramos de que splits existan
    assert set(df["split"].unique()) >= {"train", "val", "test"}, \
        f"metadata_split.csv no contiene los 3 splits (train/val/test)."

    class_names = sorted(df["label"].unique().tolist())
    print("Clases:", class_names)

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"Tamaños -> train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    summary_rows = []
    for seed in seeds:
        metrics = train_one_seed(
            seed=seed,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            class_names=class_names,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
        )
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RUN_ROOT / "run_summary_split.csv"
    if summary_csv.exists():
        # Si ya existía, lo combinamos (evitamos duplicados por seed)
        old = pd.read_csv(summary_csv)
        # Filtramos seeds nuevos
        old = old[~old["seed"].isin(summary_df["seed"])]
        summary_df = pd.concat([old, summary_df], ignore_index=True)

    summary_df.to_csv(summary_csv, index=False)
    print(f"\nResumen de seeds guardado en: {summary_csv}")
    print(summary_df)


if __name__ == "__main__":
    main()