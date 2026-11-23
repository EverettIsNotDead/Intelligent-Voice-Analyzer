"""Streamlit entry point for Intelligent Voice Analyzer."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
src = PROJECT_ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from iva.audio import extract_features, normalize_audio
from iva.inference import Analyzer

st.set_page_config(page_title="Intelligent Voice Analyzer", page_icon="🎙️", layout="wide")
st.title("Intelligent Voice Analyzer")
st.caption("Akustik sinyal analizi ile cinsiyet ve konuşmacı tahmini")

with st.sidebar:
    st.header("Model Dosyaları")
    gender_model = st.text_input("Cinsiyet modeli", "models/gender_model.joblib")
    speaker_model = st.text_input("Konuşmacı modeli", "models/speaker_model.joblib")
    st.divider()

analyzer = Analyzer(PROJECT_ROOT / gender_model, PROJECT_ROOT / speaker_model)

tab_upload, tab_mic = st.tabs(["Dosya Yükleme", "Mikrofon Kaydı"])


def analyze_audio(raw_audio: np.ndarray, sample_rate: int) -> None:
    audio = normalize_audio(raw_audio)
    if audio.size == 0:
        st.warning("İşlenecek ses bulunamadı.")
        return

    features = extract_features(audio, sample_rate)
    gender, speaker = analyzer.predict(features)

    col1, col2 = st.columns(2)
    col1.metric("Cinsiyet Tahmini", gender.label, f"{gender.confidence:.1%}" if gender.confidence else "Bekliyor")
    col2.metric("Konuşmacı Tahmini", speaker.label, f"{speaker.confidence:.1%}" if speaker.confidence else "Bekliyor")

    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.plot(np.linspace(0, len(audio) / sample_rate, len(audio)), audio)
    ax.set_xlabel("Zaman (sn)")
    ax.set_ylabel("Genlik")
    ax.set_title("Ses Dalga Formu")
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True, use_container_width=False)


with tab_upload:
    uploaded = st.file_uploader("WAV veya ses dosyası seçin", type=["wav", "mp3", "ogg", "flac"])
    if uploaded:
        import io

        import soundfile as sf

        data, sr = sf.read(io.BytesIO(uploaded.getvalue()), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1)
        st.audio(uploaded.getvalue())
        if st.button("Ses Dosyasını Analiz Et", type="primary"):
            analyze_audio(data, sr)

with tab_mic:
    st.info("Mikrofon kaydı için tarayıcı ve Streamlit sürümünüzün desteklediğinden emin olun.")
    recorded = st.audio_input("Mikrofon ile konuşma kaydı")
    if recorded:
        st.audio(recorded)
        if st.button("Kaydı Analiz Et", type="primary"):
            import io

            import soundfile as sf

            data, sr = sf.read(io.BytesIO(recorded.getvalue()), dtype="float32", always_2d=False)
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            analyze_audio(data, sr)
