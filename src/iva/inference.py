"""Model loading and prediction interfaces."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class Analyzer:
    """Predict gender and speaker with optional trained models."""

    def __init__(self, gender_model_path: str | Path | None = None,
                 speaker_model_path: str | Path | None = None):
        self.gender_model = self._load(gender_model_path)
        self.speaker_model = self._load(speaker_model_path)

    @staticmethod
    def _load(path: str | Path | None):
        if path is None:
            return None
        path = Path(path)
        return joblib.load(path) if path.exists() else None

    @staticmethod
    def _predict(model, features: np.ndarray, fallback_label: str) -> Prediction:
        if model is None:
            return Prediction(fallback_label, 0.0)

        x = features.reshape(1, -1)
        label = str(model.predict(x)[0])
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x)[0]
            confidence = float(np.max(probabilities))
        return Prediction(label, confidence)

    def predict(self, features: np.ndarray) -> tuple[Prediction, Prediction]:
        gender = self._predict(self.gender_model, features, "Model yüklenmedi")
        speaker = self._predict(self.speaker_model, features, "Model yüklenmedi")
        return gender, speaker
