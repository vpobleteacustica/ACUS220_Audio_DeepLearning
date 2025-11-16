# src/features.py
from __future__ import annotations
import numpy as np
import librosa


def power_to_db(
    x: np.ndarray,
    ref: float | np.ndarray = 1.0,
    amin: float = 1e-10
) -> np.ndarray:
    """
    Convierte potencia (o energía) a escala dB.
    ref puede ser un escalar (ej. 1.0) o el máximo de x.
    """
    ref_val = np.maximum(amin, ref if np.isscalar(ref) else np.max(ref))
    return 10.0 * np.log10(np.maximum(amin, x) / ref_val)


def compute_stft_db(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 256
) -> np.ndarray:
    """Espectrograma de potencia en dB a partir de la STFT."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    return power_to_db(S, ref=S)


def compute_logmel_db(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 256
) -> np.ndarray:
    """Espectrograma Log-Mel en dB."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    M = mel @ S  # (n_mels, T)
    return power_to_db(M, ref=M)


def compute_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 256
) -> np.ndarray:
    """Coeficientes MFCC estándar a partir de Log-Mel."""
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def compute_gammatone_db(
    y: np.ndarray,
    sr: int,
    n_bins: int = 64,
    win: float = 0.025,
    hop: float = 0.010
) -> np.ndarray | None:
    """
    Intenta usar el paquete 'gammatone' (pip install gammatone) para un gammatonegram.
    Si no está instalado o falla, devuelve None.
    """
    try:
        from gammatone.gtgram import gtgram

        S = gtgram(y.astype(float), sr, win, hop, n_bins, f_min=50.0)
        # gtgram devuelve energía ~ amplitud; pasamos a dB
        return power_to_db(S, ref=S)
    except Exception:
        return None