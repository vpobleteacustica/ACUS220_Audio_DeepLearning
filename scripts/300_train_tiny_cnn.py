#!/usr/bin/env python3

# Entrena una CNN que consume Log-Mel (más simple y efectivo para clase).

from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# --- bootstrap sys.path to project root ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------

def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    return d["mel_db"], json.loads(d["meta"].item())

class MelNPZDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str,int]):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        mel, meta = load_npz(Path(self.df.loc[i, "npz"]))
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, F, T)
        y = torch.tensor(self.class_to_idx[meta["label"]], dtype=torch.long)
        return x, y

class TinyMelCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(64*4*4,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.net(x)

def main(data_dir: Path, epochs: int, batch: int, run_name: str):
    meta_csv = data_dir / "metadata.csv"
    assert meta_csv.exists(), f"No existe {meta_csv}. Corre 200_build_tiny_dataset.py"
    df = pd.read_csv(meta_csv)

    labels = sorted(df["label"].unique())
    class_to_idx = {c:i for i,c in enumerate(labels)}
    (data_dir / run_name).mkdir(parents=True, exist_ok=True)
    with open(data_dir / run_name / "class_names.json","w") as f:
        json.dump(labels, f, indent=2)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_dl = DataLoader(MelNPZDataset(train_df, class_to_idx), batch_size=batch, shuffle=True)
    val_dl   = DataLoader(MelNPZDataset(val_df,   class_to_idx), batch_size=batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMelCNN(len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    def run_epoch(dl, train=True):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            if train: opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            if train:
                loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += x.size(0)
        return loss_sum/total, correct/total

    hist = {"train":[], "val":[]}
    best_acc, best_path = -1, data_dir / run_name / "best_model.pt"
    for ep in range(1, epochs+1):
        tl, _  = run_epoch(train_dl, True)
        vl, va = run_epoch(val_dl, False)
        hist["train"].append(tl); hist["val"].append(vl)
        print(f"Epoch {ep:02d}/{epochs} | train {tl:.3f}  val {vl:.3f}  acc {va:.3f}")
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), best_path)

    # Curvas
    plt.figure(figsize=(6,4))
    plt.plot(hist["train"], label="train"); plt.plot(hist["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(data_dir / run_name / "loss_curves.png", dpi=150)

    # Matriz + reporte
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x,y in val_dl:
            p = model(x.to(device)).argmax(1).cpu().numpy().tolist()
            y_pred += p; y_true += y.numpy().tolist()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5.5,5)); disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout(); plt.savefig(data_dir / run_name / "confusion_matrix.png", dpi=150)

    rep = classification_report(y_true, y_pred, target_names=labels, digits=3)
    with open(data_dir / run_name / "report.txt","w") as f: f.write(rep)
    print("\n📋 Classification report:\n", rep)
    print("\n✅ Artefactos en:", data_dir / run_name)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/tiny_specs"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--run", type=str, default="cnn_run01")
    args = ap.parse_args()
    main(args.data, args.epochs, args.batch, args.run)