# scripts/300_train_tiny_cnn.py
# -*- coding: utf-8 -*-
"""
Entrena una CNN simple sobre Log-Mel con división 70/15/15 (train/val/test).
Soporta multi-seed y guarda artefactos por semilla:

data/processed/tiny_specs/cnn_run/
  └── seed_<SEED>/
      ├── best_model.pt
      ├── class_names.json
      ├── loss_curves.png
      ├── val_confusion_matrix.png
      ├── val_report.txt
      ├── test_confusion_matrix.png
      ├── test_report.txt
      ├── test_predictions.csv
      └── splits/
          ├── train.csv
          ├── val.csv
          └── test.csv

Además genera un resumen:
  data/processed/tiny_specs/cnn_run/run_summary.csv
"""

# # 1) Asegura que el mini-dataset exista
# python -m scripts.200_build_tiny_dataset
# 
# # 2) Entrena con varias semillas y 70/15/15
# python -m scripts.300_train_tiny_cnn --epochs 25 --batch-size 8 --lr 1e-3 --seeds 42,7,123

import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# --------------------- Rutas base ---------------------
DATA_DIR = Path("data/processed/tiny_specs")
META_CSV = DATA_DIR / "metadata.csv"
RUN_ROOT = DATA_DIR / "cnn_run"
RUN_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------- Utils ---------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ejecución determinista (útil para CPU/Mac)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- Dataset ---------------------
class MelNPZDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, class_to_idx: dict):
        self.df = meta_df.reset_index(drop=True)
        self.class_to_idx = class_to_idx

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        npz_path = self.df.loc[i, "npz"]
        d = np.load(npz_path, allow_pickle=True)
        mel_db = d["mel_db"]
        meta = json.loads(d["meta"].item())
        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # (1, F, T)
        y = torch.tensor(self.class_to_idx[meta["label"]], dtype=torch.long)
        return x, y


# --------------------- Modelo ---------------------
class TinyMelCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# --------------------- Loop de entrenamiento/validación ---------------------
def run_epoch(model, dl, crit, opt=None, device="cpu"):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        if train: opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        if train:
            loss.backward()
            opt.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


# --------------------- Entrenar una semilla ---------------------
def train_one_seed(df, seed, epochs, batch_size, lr, device):
    set_seeds(seed)
    out_dir = RUN_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    labels = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(labels)}
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(labels, f, indent=2)

    # --- Split 70/15/15 estratificado ---
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )

    # Guardar splits para reproducibilidad
    train_df.to_csv(out_dir / "splits" / "train.csv", index=False)
    val_df.to_csv(out_dir / "splits" / "val.csv", index=False)
    test_df.to_csv(out_dir / "splits" / "test.csv", index=False)

    # DataLoaders
    train_dl = DataLoader(MelNPZDataset(train_df, class_to_idx),
                          batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(MelNPZDataset(val_df, class_to_idx),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(MelNPZDataset(test_df, class_to_idx),
                         batch_size=batch_size, shuffle=False, num_workers=0)

    # Modelo/opt/crit
    model = TinyMelCNN(len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_path = -1.0, out_dir / "best_model.pt"

    # --- Entrenamiento ---
    for ep in range(1, epochs + 1):
        tr_loss, _ = run_epoch(model, train_dl, crit, opt, device)
        va_loss, va_acc = run_epoch(model, val_dl, crit, None, device)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        print(f"[seed {seed}] Epoch {ep:02d}/{epochs} | train {tr_loss:.3f}  val {va_loss:.3f}  acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)

    # Curvas
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()

    # --- Evaluación VALIDACIÓN (para matriz/report val) ---
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x.to(device))
            y_pred_val += logits.argmax(1).cpu().tolist()
            y_true_val += y.tolist()

    cm_val = confusion_matrix(y_true_val, y_pred_val, labels=list(range(len(labels))))
    ConfusionMatrixDisplay(cm_val, display_labels=labels).plot(
        xticks_rotation=45, colorbar=False, cmap="Blues"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "val_confusion_matrix.png", dpi=150)
    plt.close()

    rep_val = classification_report(
        y_true_val, y_pred_val, target_names=labels, digits=3, zero_division=0
    )
    with open(out_dir / "val_report.txt", "w") as f:
        f.write(rep_val)

    # --- Evaluación TEST (final y “honesta”) ---
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for x, y in test_dl:
            logits = model(x.to(device))
            y_pred_test += logits.argmax(1).cpu().tolist()
            y_true_test += y.tolist()

    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=list(range(len(labels))))
    ConfusionMatrixDisplay(cm_test, display_labels=labels).plot(
        xticks_rotation=45, colorbar=False, cmap="Blues"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "test_confusion_matrix.png", dpi=150)
    plt.close()

    rep_test = classification_report(
        y_true_test, y_pred_test, target_names=labels, digits=3, zero_division=0,
        output_dict=False
    )
    with open(out_dir / "test_report.txt", "w") as f:
        f.write(rep_test)

    # Guardar predicciones de test (útil para inspección)
    test_pred_rows = []
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    for i, (yt, yp) in enumerate(zip(y_true_test, y_pred_test)):
        test_pred_rows.append({"idx": i, "y_true": idx_to_class[yt], "y_pred": idx_to_class[yp]})
    pd.DataFrame(test_pred_rows).to_csv(out_dir / "test_predictions.csv", index=False)

    # Accuracy test (para resumen)
    test_acc = (np.array(y_true_test) == np.array(y_pred_test)).mean() if len(y_true_test) else 0.0

    print(f"[seed {seed}] 💾 Mejor modelo: {best_path}")
    print(f"[seed {seed}] Val best acc: {best_acc:.3f} | Test acc: {test_acc:.3f}")
    print(f"[seed {seed}] Reporte test guardado en: {out_dir/'test_report.txt'}")

    return {
        "seed": seed,
        "val_best_acc": float(best_acc),
        "test_acc": float(test_acc),
        "best_model": str(best_path)
    }


# --------------------- Main (multi-seed) ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=str, default="42")  # e.g., "42,7,123"
    args = ap.parse_args()

    assert META_CSV.exists(), f"No existe {META_CSV}. Corre primero 200_build_tiny_dataset.py"
    df = pd.read_csv(META_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    results = []
    for s in seeds:
        res = train_one_seed(
            df=df, seed=s, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device
        )
        results.append(res)

    # Resumen
    summary_df = pd.DataFrame(results).sort_values(by="test_acc", ascending=False)
    summary_path = RUN_ROOT / "run_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n📊 Resumen multi-seed:")
    print(summary_df.to_string(index=False))
    print(f"💾 Guardado resumen en: {summary_path}")


if __name__ == "__main__":
    main()