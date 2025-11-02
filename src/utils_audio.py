# src/utils_audio.py
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

def load_wav(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Lee WAV con soundfile y remuestrea con librosa si es necesario."""
    y, orig_sr = sf.read(str(path), always_2d=False)
    y = y.astype(np.float32)
    if y.ndim > 1:  # mono
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y, sr

def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    """Ajusta a longitud target_len (muestras)."""
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)