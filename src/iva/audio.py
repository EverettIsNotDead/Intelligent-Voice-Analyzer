"""Audio loading and preprocessing helpers."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as mono and resample it."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    return audio, sample_rate


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Remove silence at boundaries and normalize peak amplitude."""
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
    if peak > 0:
        trimmed = trimmed / peak
    return trimmed.astype(np.float32)


def extract_features(audio: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Extract the MFCC statistic vector planned by the roadmap."""
    if audio.size == 0:
        return np.zeros(40 * 2, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate((mean, std)).astype(np.float32)
