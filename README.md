# edupredictmath-ai

> 🇮🇩 Bahasa Indonesia | 🇬🇧 [English](#english-version)

---

## 🇮🇩 Versi Bahasa Indonesia

Repositori ini berisi **AI Service** dari project EduPredictMath — bagian yang bertanggung jawab atas prediksi penguasaan konsep matematika siswa menggunakan model Knowledge Tracing.

Service utama ada di folder `ai-service/` dan menyediakan endpoint **POST** `/predict` untuk:
- menjalankan inference model DKT (LSTM atau Causal Cross-Transformer),
- menghitung agregasi mastery pada 6 top kategori matematika,
- (opsional) memicu hint step-by-step Bahasa Indonesia yang dipersonalisasi menggunakan Google Gemini bila siswa terdeteksi “struggling”.

### 📁 Struktur Repositori

```text
edupredictmath-ai/
├── ai-service/                                        # FastAPI inference service (deployable)
│   ├── Dockerfile                                     # Image build untuk deployment
│   ├── inference_api.py                               # Entry point FastAPI + pipeline inference
│   ├── top_category.json                              # Mapping 6 top category + skill indices
│   ├── final/                                         # Artifak model untuk inference
│   │   ├── causal_cross_transformer_dkt_model_train/
│   │   │   ├── causal_cross_transformer_dkt_model.keras
│   │   │   └── vocab.json
│   │   └── lstm_dkt_model_train/
│   │       ├── lstm_dkt_model.keras
│   │       └── vocab.json
│   └── README.md                                      # Dokumentasi endpoint & pipeline inference
├── notebooks/                                         # Eksperimen & training (tidak di-deploy)
│   ├── causal_cross_transformer_dkt_model_train/
│   │   ├── causal_cross_transformer_dkt_model_train.ipynb
│   │   ├── requirements.txt
│   │   ├── vocab.json
│   │   ├── assets/
│   │   └── logs/
│   │       ├── train/
│   │       └── val/
│   └── lstm_dkt_model_train/
│       ├── lstm_dkt_model_train.ipynb
│       ├── requirements.txt
│       ├── vocab.json
│       ├── assets/
│       └── logs/
│           ├── train/
│           └── val/
├── data/
│   ├── raw/                                           # Data mentah (placeholder .gitkeep)
│   └── processed/                                     # Data hasil preprocessing (placeholder .gitkeep)
├── pyproject.toml                                     # Dependencies (FastAPI, Uvicorn, Pydantic, dsb.)
├── uv.lock
└── README.md
```

> ⚠️ Folder `data/raw/` dan `data/processed/` biasanya tidak di-commit (hanya `.gitkeep`). Artifak model inference ada di `ai-service/final/`.

---

### ⚙️ Setup & Cara Menjalankan (Local)

#### Prasyarat
- Python 3.10+
- (Opsional) `uv` untuk install dependency dari `pyproject.toml`
- (Opsional) Gemini API key jika ingin hint personalisasi

#### 1. Install dependencies

Opsi A (disarankan jika memakai `uv`):
```bash
uv sync
```

Opsi B (pip):
```bash
pip install -e .
```

#### 2. Set environment variable (opsional)

Jika ingin hint Gemini aktif, set `GEMINI_API_KEY`:
```bash
setx GEMINI_API_KEY "YOUR_KEY"
```

#### 3. Jalankan API

```bash
cd ai-service
uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000
```

API akan berjalan di `http://localhost:8000`.
Dokumentasi otomatis tersedia di `http://localhost:8000/docs`.

---

### 🔗 Endpoint

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/predict` | Prediksi penguasaan konsep siswa + hint opsional |

Detail schema request/response, aturan trigger Gemini, dan contoh payload lengkap ada di `ai-service/README.md`.

---

### 🤝 Cara Kontribusi

1. Pastikan kamu sudah di-invite sebagai **Collaborator** di repo ini
2. Jangan langsung push ke branch `main`
3. Buat branch baru untuk setiap fitur atau perbaikan:
```bash
git checkout -b feat/nama-fitur
```
4. Setelah selesai, buat **Pull Request** ke branch `main`
5. Minta review ke AI Engineer sebelum merge

---

### 📌 Catatan Penting

- File `.env` jangan pernah di-push ke GitHub
- Model yang dipakai dapat di-switch di `ai-service/inference_api.py` melalui konstanta `MODEL_TYPE` (`LSTM` atau `Transformer`)
- Konfigurasi container (port 7860, dll.) ada di `ai-service/Dockerfile`

---
---

## English Version

This repository contains the **AI Service** of the EduPredictMath project — the component responsible for predicting students' math concept mastery using Knowledge Tracing.

The main service is in `ai-service/` and exposes **POST** `/predict` to:
- run DKT inference (LSTM or Causal Cross-Transformer),
- aggregate mastery into 6 top math categories,
- (optionally) trigger a personalized Indonesian step-by-step hint via Google Gemini when a student is detected as “struggling”.

### 📁 Repository Structure

```text
edupredictmath-ai/
├── ai-service/                                        # FastAPI inference service (deployable)
│   ├── Dockerfile                                     # Image build for deployment
│   ├── inference_api.py                               # FastAPI entry point + inference pipeline
│   ├── top_category.json                              # 6 top categories + skill indices
│   ├── final/                                         # Inference model artifacts
│   │   ├── causal_cross_transformer_dkt_model_train/
│   │   │   ├── causal_cross_transformer_dkt_model.keras
│   │   │   └── vocab.json
│   │   └── lstm_dkt_model_train/
│   │       ├── lstm_dkt_model.keras
│   │       └── vocab.json
│   └── README.md                                      # Detailed endpoint & pipeline docs
├── notebooks/                                         # Experiments & training (not deployed)
│   ├── causal_cross_transformer_dkt_model_train/
│   │   ├── causal_cross_transformer_dkt_model_train.ipynb
│   │   ├── requirements.txt
│   │   ├── vocab.json
│   │   ├── assets/
│   │   └── logs/
│   │       ├── train/
│   │       └── val/
│   └── lstm_dkt_model_train/
│       ├── lstm_dkt_model_train.ipynb
│       ├── requirements.txt
│       ├── vocab.json
│       ├── assets/
│       └── logs/
│           ├── train/
│           └── val/
├── data/
│   ├── raw/                                           # Raw data (placeholder .gitkeep)
│   └── processed/                                     # Processed data (placeholder .gitkeep)
├── pyproject.toml                                     # Dependencies (FastAPI, Uvicorn, Pydantic, etc.)
├── uv.lock
└── README.md
```

> ⚠️ `data/raw/` and `data/processed/` are typically not committed (only `.gitkeep`). Inference model artifacts are located in `ai-service/final/`.

---

### ⚙️ Setup & How to Run (Local)

#### Prerequisites
- Python 3.10+
- (Optional) `uv` to install dependencies from `pyproject.toml`
- (Optional) Gemini API key if you want personalized hints

#### 1. Install dependencies

Option A (recommended if you use `uv`):
```bash
uv sync
```

Option B (pip):
```bash
pip install -e .
```

#### 2. Set environment variable (optional)

To enable Gemini hints, set `GEMINI_API_KEY`:
```bash
setx GEMINI_API_KEY "YOUR_KEY"
```

#### 3. Run the API

```bash
cd ai-service
uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000
```

The API will run at `http://localhost:8000`.
Auto-generated docs are available at `http://localhost:8000/docs`.

---

### 🔗 Endpoint

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict mastery + optional hint |

For the full request/response schema, Gemini trigger rules, and complete payload examples, see `ai-service/README.md`.

---

### 🤝 How to Contribute

1. Make sure you have been invited as a **Collaborator** on this repo
2. Never push directly to the `main` branch
3. Create a new branch for each feature or fix:
```bash
git checkout -b feat/feature-name
```
4. When done, open a **Pull Request** to the `main` branch
5. Request a review from the AI Engineer before merging

---

### 📌 Important Notes

- Never commit `.env`
- The model can be switched in `ai-service/inference_api.py` via the `MODEL_TYPE` constant (`LSTM` or `Transformer`)
- Container settings (port 7860, etc.) are defined in `ai-service/Dockerfile`
