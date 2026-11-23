import numpy as np

from iva.audio import extract_features, normalize_audio


def test_normalize_audio_trims_and_normalizes():
    audio = np.concatenate((np.zeros(100), np.linspace(-0.5, 0.5, 1000)))
    normalized = normalize_audio(audio.astype(np.float32))

    assert normalized.size > 0
    assert float(np.max(np.abs(normalized))) == 1.0


def test_extract_features_has_expected_size():
    rng = np.random.default_rng(42)
    features = extract_features(rng.normal(size=16_000).astype(np.float32), 16_000)

    assert features.shape == (80,)
