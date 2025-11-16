# src/utils_audio.py
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


def load_wav(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Lee WAV con soundfile y remuestrea con librosa si es necesario."""
    y, orig_sr = sf.read(str(path), always_2d=False)
    y = y.astype(np.float32)
    if y.ndim > 1:  # mezcla a mono si es estéreo
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y, sr


def pad_or_trim(
    y: np.ndarray,
    target_len: int,
    mode: str = "right",
) -> np.ndarray:
    """
    Ajusta la señal a longitud target_len (en muestras).

    mode:
      - "right"  -> padding al final (comportamiento clásico).
      - "left"   -> padding al inicio.
      - "center" -> padding repartido mitad-inicio / mitad-final.

    Si la señal es más larga, se recorta según el mismo criterio.
    """
    n = len(y)

    if n < target_len:
        pad_total = target_len - n
        if mode == "right":
            pad = (0, pad_total)
        elif mode == "left":
            pad = (pad_total, 0)
        elif mode == "center":
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad = (pad_left, pad_right)
        else:
            raise ValueError(f"mode desconocido: {mode}")
        y = np.pad(y, pad)
    elif n > target_len:
        if mode == "right":
            # nos quedamos con el inicio
            y = y[:target_len]
        elif mode == "left":
            # nos quedamos con el final
            y = y[-target_len:]
        elif mode == "center":
            extra = n - target_len
            start = extra // 2
            y = y[start:start + target_len]
        else:
            raise ValueError(f"mode desconocido: {mode}")

    return y.astype(np.float32)