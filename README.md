# Intelligent Voice Analyzer (IVA)

Akustik sinyal analizi ile ses dosyası veya mikrofon kaydı üzerinden cinsiyet tahmini ve bireysel konuşmacı tanıma yapan Python tabanlı bir prototip proje.

## Özellikler

- Ses dosyası yükleme: WAV, MP3, OGG ve FLAC
- Mikrofon üzerinden canlı ses kaydı
- Ses ön işlemesi ve 16 kHz mono dönüştürme
- 40 MFCC katsayısının ortalama ve standart sapma özeti
- Cinsiyet ve konuşmacı tahminleri için arayüz çıktıları
- Ses dalga formu görselleştirmesi
- Eğitilmiş scikit-learn modelleri için Joblib entegrasyonu

> Mevcut sürüm, kullanıcı arayüzü ve model çıkarım iskeletini sağlar. Gerçek tahminler için eğitilmiş `gender_model.joblib` ve `speaker_model.joblib` dosyaları gereklidir.

## Gereksinimler

- Python 3.10 veya üzeri
- Mikrofon desteği için çalışan ses aygıtları
- Tarayıcı üzerinden Streamlit uygulama erişimi

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell ortamında sanal ortamı etkinleştirmek için:

```powershell
.venv\Scripts\Activate.ps1
```

## Çalıştırma

```bash
streamlit run app/main.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

## Model Dosyaları (Model Eğitecek Arkadaşlar Dikkat)

Cinsiyet ve konuşmacı modellerini şu konumlara yerleştirin:

```text
models/gender_model.joblib
models/speaker_model.joblib
```

Modeller `.predict()` metodunu desteklemelidir. Olasılık yüzdesi için `.predict_proba()` yöntemini destekleyen modeller tercih edilir. İki modelin de aynı öznitelik çıkarımı ve ölçekleme şemasına uygun olması gerekir.

## Test ve Kod Kalitesi

```bash
pytest -q
ruff check src tests app
```

## Proje Yapısı

```text
app/
  main.py                 Streamlit arayüzü
src/iva/
  __init__.py             Paket tanımı
  audio.py                Ses yükleme, normalizasyon ve öznitelik çıkarımı
  inference.py            Model yükleme ve tahmin arabirimi
models/                   Eğitilmiş Joblib model dosyaları
data/                     Ham ve işlenmiş veri
assets/                   Proje görselleri ve sunum varlıkları
tests/                    Birim testleri
pyproject.toml            Paket ve bağımlılık yapılandırması
.gitignore                Git dışlama kuralları
```
