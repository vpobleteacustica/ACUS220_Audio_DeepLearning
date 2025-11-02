# scripts/350_infer_one.py
# -*- coding: utf-8 -*-
"""
Inferencia de UNA grabación WAV usando el mejor modelo disponible.
Uso:
  python -m scripts.350_infer_one --wav path/to/file.wav
  # o forzar una semilla concreta
  python -m scripts.350_infer_one --wav path/to/file.wav --seed 123
"""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.utils_audio import load_wav, pad_or_trim
from src.features import compute_logmel_db

DATA_DIR = Path("data/processed/tiny_specs")
RUN_ROOT = DATA_DIR / "cnn_run"
RUN_SUMMARY = RUN_ROOT / "run_summary.csv"

class TinyMelCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(64*4*4,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,n_classes)
        )
    def forward(self,x): return self.net(x)

def pick_best_seed():
    import pandas as pd
    assert RUN_SUMMARY.exists(), f"No existe {RUN_SUMMARY}. Entrena primero."
    df = pd.read_csv(RUN_SUMMARY)
    df = df.sort_values(by="test_acc", ascending=False)
    best = df.iloc[0]
    seed = int(best["seed"])
    model_path = Path(best["best_model"])
    return seed, model_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--seed", type=int, default=None, help="Si omites, usa el mejor de run_summary.csv")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--n_mels", type=int, default=64)
    args = ap.parse_args()

    if args.seed is None:
        seed, model_path = pick_best_seed()
    else:
        seed = args.seed
        model_path = RUN_ROOT/f"seed_{seed}"/"best_model.pt"
        assert model_path.exists(), f"No existe {model_path}. Revisa entrenamiento."
    class_file = model_path.parent/"class_names.json"
    assert class_file.exists(), f"Falta {class_file}"

    labels = json.loads(Path(class_file).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMelCNN(len(labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Log-Mel del WAV
    wav_path = Path(args.wav)
    y, sr = load_wav(wav_path, sr=args.sr)
    y = pad_or_trim(y, int(args.sr*args.duration))
    mel_db = compute_logmel_db(y, args.sr, n_mels=args.n_mels)
    x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1).cpu().numpy()[0]
        top_idx = int(np.argmax(probs))
        top_cls = labels[top_idx]
        top_p = float(probs[top_idx])

    # Visual de apoyo
    plt.figure(figsize=(6,3))
    plt.imshow(mel_db, aspect="auto", origin="lower")
    plt.title(f"Pred: {top_cls} ({top_p:.2f})")
    plt.xticks([]); plt.yticks([])
    out_png = model_path.parent / f"infer_{wav_path.stem}.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

    print(f"✅ WAV: {wav_path}")
    print(f"🔎 Modelo: {model_path}")
    print(f"🎯 Predicción: {top_cls} (p={top_p:.3f})")
    print(f"🖼️ Vista: {out_png}")

if __name__ == "__main__":
    main()