# EduPredict Math — AI Service (Unified Inference API)

> 🇮🇩 Bahasa Indonesia | 🇬🇧 [English](#english-version)

---

## 🇮🇩 Versi Bahasa Indonesia

Dokumentasi ini menjelaskan pipeline inference yang digunakan oleh `inference_api.py` dan spesifikasi endpoint **POST** `/predict`.

### Ringkasan

Service ini melakukan:
1. Memuat artifak model (LSTM atau Causal Cross-Transformer) + `vocab.json`.
2. Menjalankan inference untuk memprediksi *mastery probability* tiap skill.
3. Mengagregasi hasil menjadi 6 top kategori matematika berbasis `top_category.json`.
4. (Opsional) Memanggil Google Gemini untuk memberi hint step-by-step Bahasa Indonesia jika siswa terdeteksi “struggling”.

---

## ⚙️ Menjalankan Service

### Local (tanpa Docker)

Dari root repo:
```bash
cd ai-service
uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000
```

Buka:
- Swagger UI: `http://localhost:8000/docs`
- Endpoint: `http://localhost:8000/predict`

### Environment variable (Gemini)

Jika ingin hint Gemini aktif:
- `GEMINI_API_KEY`: API key Google Gemini.

Jika `GEMINI_API_KEY` tidak diset, service tetap berjalan dan akan mengembalikan placeholder offline saat trigger terpenuhi.

---

## 🔌 Endpoint: POST /predict

### Request Body

```json
{
  "student_history": [
    {
      "skill_id": "311",
      "correctness": 0,
      "ms_first_response": 15000,
      "question": "Selesaikan nilai x: x + 1 = 5"
    }
  ],
  "personal_preference": "video games"
}
```

#### Field

- `student_history`: daftar interaksi berurutan (oldest → newest)
  - `skill_id` (string): skill ID, dapat berupa compound dipisah underscore (contoh: `"2_37"`, `"311_1"`)
  - `correctness` (int): `1` benar, `0` salah
  - `ms_first_response` (int): waktu menjawab dalam milidetik
  - `question` (string, opsional): teks soal terakhir (dipakai untuk hint step-by-step yang spesifik)
- `personal_preference` (string): minat/hobi siswa untuk personalisasi hint (mis. `"video games"`, `"music"`)

### Response Body

```json
{
  "category_mastery": {
    "Data Analysis, Statistics, and Probability": 0.4539,
    "Geometry and Spatial Reasoning": null,
    "Measurement, Area, and Volume": null,
    "Number Sense, Properties, and Operations": 0.4291,
    "Ratios, Proportions, and Percentages": null,
    "Algebraic Thinking, Equations, and Inequalities": 0.4191
  },
  "explanation": "..."
}
```

- `category_mastery`: dictionary 6 kategori → skor rata-rata (0.0–1.0) atau `null` jika tidak ada data
- `explanation`: string hint Bahasa Indonesia (maks 10 baris, jika trigger terpenuhi) atau `null`

---

## 🧠 Bagaimana Pipeline Inference Bekerja

Bagian ini merangkum perilaku aktual sesuai implementasi di `inference_api.py`.

### 1) Startup: Load artifak
Saat startup FastAPI:
- Memuat model Keras dari `final/.../*.keras`.
- Memuat vocabulary dan mapping dari `final/.../vocab.json`.
- Memuat top kategori dari `top_category.json` (kategori `"Miscellaneous / Other"` difilter).

### 2) Pre-processing sequence
Untuk tiap interaksi $t$:
- `skill_id` diubah menjadi vektor multi-hot panjang `vocab_size`.
- `correctness` membentuk vektor correct multi-hot.
- `ms_first_response` diubah menjadi fitur log-time:

$$r = \log(1 + \text{seconds})$$

Dengan aturan:
- nilai negatif dibulatkan ke 0
- outlier dicap: 8 menit (LSTM) atau 10 menit (Transformer)

### 3) Inference (dua mode)
Mode ditentukan oleh konstanta `MODEL_TYPE` di `inference_api.py`:

- **LSTM**
  - Input dibentuk ukuran `(1, MAX_SEQ_LEN, 2*vocab_size + 1)`.
  - Sequence dipre-pad sehingga step aktif berada di akhir window.
  - Mastery diambil dari prediksi step terakhir.

- **Transformer (Causal Cross-Transformer)**
  - Untuk mendapatkan probabilitas untuk *semua* skill, dibuat batch ukuran `vocab_size`.
  - Setiap item batch merepresentasikan kandidat target skill pada step terakhir.
  - Input lalu di-truncate/post-pad menjadi `(vocab_size, MAX_SEQ_LEN, 3*vocab_size + 1)`.
  - Probabilitas diambil pada time-step valid terakhir.

### 4) Trigger hint Gemini (berdasarkan interaksi terakhir)
Service memeriksa interaksi terakhir $T_{last}$:

- Pecah `skill_id` terakhir menjadi komponen (mis. `"311_9"` → `"311"`, `"9"` bila ada; dan `"2_37"` → `"2"`, `"37"`).
- Untuk tiap komponen skill:
  - Ambil mastery $p$ dari output model.
  - Hitung berapa kali skill tersebut muncul dalam seluruh `student_history`.

Trigger terpenuhi jika:
- $p < 0.50$ (`STRUGGLE_THRESHOLD`) **dan**
- `attempts > 3` (`LENGTH_THRESHOLD`, strictly greater)

Jika terpenuhi, service membangun prompt berisi:
- daftar konsep yang struggling (nama skill),
- `personal_preference`,
- (opsional) teks `question`,

lalu memanggil model `gemini-2.5-flash` untuk menghasilkan hint singkat (strictly < 10 baris).

### 5) Agregasi 6 kategori
Untuk setiap top kategori:
- Ambil daftar `skill_id_included` (berupa index string) dari `top_category.json`.
- Masukkan mastery skill jika skill tersebut memiliki setidaknya 1 kemunculan di history (`AVG_LEN_THRESHOLD = 1`).
- Skor kategori = rata-rata sederhana semua skill eligible.
- Jika tidak ada yang eligible → `null`.

---

## 📬 Contoh cURL (berdasarkan dokumentasi endpoint)

### Contoh 1 — Trigger aktif (explanation terisi)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "student_history": [
      { "skill_id": "2", "correctness": 1, "ms_first_response": 12000 },
      { "skill_id": "2_37", "correctness": 0, "ms_first_response": 25000 },
      { "skill_id": "2", "correctness": 0, "ms_first_response": 18000 },
      { "skill_id": "311_1", "correctness": 0, "ms_first_response": 39320 },
      { "skill_id": "311_9", "correctness": 0, "ms_first_response": 30000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 10000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 10000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 10000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 10000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 15000, "question": "Selesaikan nilai x: x + 1 = 5" }
    ],
    "personal_preference": "video games"
  }'
```

Response (contoh):
```json
{
  "category_mastery": {
    "Data Analysis, Statistics, and Probability": 0.4539,
    "Geometry and Spatial Reasoning": null,
    "Measurement, Area, and Volume": null,
    "Number Sense, Properties, and Operations": 0.4291,
    "Ratios, Proportions, and Percentages": null,
    "Algebraic Thinking, Equations, and Inequalities": 0.4191
  },
  "explanation": "Halo, gamer! Anggap `x` itu nyawa awal karaktermu.\nKamu dapat `+1` potion nyawa, total nyawa jadi `5`.\n..."
}
```

### Contoh 2 — Trigger tidak aktif (explanation = null)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "student_history": [
      { "skill_id": "2", "correctness": 1, "ms_first_response": 12000 },
      { "skill_id": "2_37", "correctness": 0, "ms_first_response": 25000 },
      { "skill_id": "2", "correctness": 0, "ms_first_response": 18000 },
      { "skill_id": "311_1", "correctness": 0, "ms_first_response": 39320 },
      { "skill_id": "311_9", "correctness": 1, "ms_first_response": 30000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 15000, "question": "Selesaikan nilai x: x + 1 = 5" }
    ],
    "personal_preference": "video games"
  }'
```

Response (contoh):
```json
{
  "category_mastery": {
    "Data Analysis, Statistics, and Probability": 0.6578,
    "Geometry and Spatial Reasoning": null,
    "Measurement, Area, and Volume": null,
    "Number Sense, Properties, and Operations": 0.6416,
    "Ratios, Proportions, and Percentages": null,
    "Algebraic Thinking, Equations, and Inequalities": 0.6652
  },
  "explanation": null
}
```

---
---

## English Version

This document describes the inference pipeline implemented in `inference_api.py` and the **POST** `/predict` endpoint.

### Summary

This service:
1. Loads model artifacts (LSTM or Causal Cross-Transformer) and `vocab.json`.
2. Runs inference to produce per-skill mastery probabilities.
3. Aggregates scores into 6 top math categories using `top_category.json`.
4. (Optional) Calls Google Gemini to generate a short Indonesian step-by-step hint when the student is detected as “struggling”.

---

## ⚙️ Running the Service

### Local (without Docker)

From the repo root:
```bash
cd ai-service
uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000
```

Open:
- Swagger UI: `http://localhost:8000/docs`
- Endpoint: `http://localhost:8000/predict`

### Environment variable (Gemini)

To enable Gemini hints:
- `GEMINI_API_KEY`: Google Gemini API key.

If `GEMINI_API_KEY` is not set, the service still works and returns an offline placeholder when the trigger is met.

---

## 🔌 Endpoint: POST /predict

### Request Body

```json
{
  "student_history": [
    {
      "skill_id": "311",
      "correctness": 0,
      "ms_first_response": 15000,
      "question": "Solve for x: x + 1 = 5"
    }
  ],
  "personal_preference": "video games"
}
```

#### Fields

- `student_history`: chronological interaction list (oldest → newest)
  - `skill_id` (string): skill ID, may be underscore-separated compound (e.g. `"2_37"`, `"311_1"`)
  - `correctness` (int): `1` for correct, `0` for incorrect
  - `ms_first_response` (int): milliseconds before submitting an answer
  - `question` (string, optional): the question text (used to generate a specific step-by-step hint)
- `personal_preference` (string): student hobby/interest for personalization

### Response Body

- `category_mastery`: mapping of the 6 categories → average mastery (0.0–1.0) or `null` when insufficient interactions
- `explanation`: Indonesian hint text (max 10 lines) when triggered, otherwise `null`

---

## 🧠 How the Inference Pipeline Works

This section summarizes the behavior as implemented in `inference_api.py`.

### 1) Startup: load artifacts
On FastAPI startup the service:
- Loads the Keras model from `final/.../*.keras`.
- Loads vocabulary + mappings from `final/.../vocab.json`.
- Loads top categories from `top_category.json` (filters out `"Miscellaneous / Other"`).

### 2) Sequence preprocessing
For each interaction $t$:
- `skill_id` is encoded into a multi-hot vector of length `vocab_size`.
- `correctness` produces a multi-hot correctness vector.
- `ms_first_response` becomes a log-time feature:

$$r = \log(1 + \text{seconds})$$

Rules:
- negative values are clamped to 0
- response times are capped: 8 minutes (LSTM) or 10 minutes (Transformer)

### 3) Inference (two modes)
Controlled by the `MODEL_TYPE` constant in `inference_api.py`.

- **LSTM**
  - Builds input `(1, MAX_SEQ_LEN, 2*vocab_size + 1)`.
  - Pre-pads so active steps are at the end.
  - Uses the final time-step predictions.

- **Transformer (Causal Cross-Transformer)**
  - Builds a batch of size `vocab_size` to score *all* target skills in parallel.
  - Truncates / post-pads to `(vocab_size, MAX_SEQ_LEN, 3*vocab_size + 1)`.
  - Reads probabilities at the last valid time-step.

### 4) Gemini hint trigger (based on the last interaction)
For the last interaction $T_{last}$:

- Split the last `skill_id` into components (e.g. `"2_37"` → `"2"` and `"37"`).
- For each component skill:
  - read mastery $p$ from model output
  - count how many times it appears across the whole history

The trigger is met when:
- $p < 0.50$ (`STRUGGLE_THRESHOLD`) **and**
- `attempts > 3` (`LENGTH_THRESHOLD`, strictly greater)

When triggered, the service builds a prompt with:
- struggling concept names,
- `personal_preference`,
- optional `question` text,

then calls `gemini-2.5-flash` to generate a concise hint (strictly < 10 lines).

### 5) Category aggregation
For each top category:
- Reads `skill_id_included` (string indices) from `top_category.json`.
- Includes a skill if it appears at least once in history (`AVG_LEN_THRESHOLD = 1`).
- Category score = simple mean of included skills.
- If no eligible skill → `null`.

---

## 📬 cURL examples (from endpoint documentation)

The following examples mirror the payload/response format used by the unified `/predict` endpoint.

### Example 1 — Triggered (non-null `explanation`)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "student_history": [
      { "skill_id": "2", "correctness": 1, "ms_first_response": 12000 },
      { "skill_id": "2_37", "correctness": 0, "ms_first_response": 25000 },
      { "skill_id": "2", "correctness": 0, "ms_first_response": 18000 },
      { "skill_id": "311_1", "correctness": 0, "ms_first_response": 39320 },
      { "skill_id": "311_9", "correctness": 0, "ms_first_response": 30000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 10000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 10000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 10000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 10000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 15000, "question": "Selesaikan nilai x: x + 1 = 5" }
    ],
    "personal_preference": "video games"
  }'
```

Sample response:
```json
{
  "category_mastery": {
    "Data Analysis, Statistics, and Probability": 0.4539,
    "Geometry and Spatial Reasoning": null,
    "Measurement, Area, and Volume": null,
    "Number Sense, Properties, and Operations": 0.4291,
    "Ratios, Proportions, and Percentages": null,
    "Algebraic Thinking, Equations, and Inequalities": 0.4191
  },
  "explanation": "Halo, gamer! Anggap `x` itu nyawa awal karaktermu.\nKamu dapat `+1` potion nyawa, total nyawa jadi `5`.\n..."
}
```

### Example 2 — Not triggered (`explanation = null`)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "student_history": [
      { "skill_id": "2", "correctness": 1, "ms_first_response": 12000 },
      { "skill_id": "2_37", "correctness": 0, "ms_first_response": 25000 },
      { "skill_id": "2", "correctness": 0, "ms_first_response": 18000 },
      { "skill_id": "311_1", "correctness": 0, "ms_first_response": 39320 },
      { "skill_id": "311_9", "correctness": 1, "ms_first_response": 30000 },
      { "skill_id": "311", "correctness": 0, "ms_first_response": 15000, "question": "Selesaikan nilai x: x + 1 = 5" }
    ],
    "personal_preference": "video games"
  }'
```

Sample response:
```json
{
  "category_mastery": {
    "Data Analysis, Statistics, and Probability": 0.6578,
    "Geometry and Spatial Reasoning": null,
    "Measurement, Area, and Volume": null,
    "Number Sense, Properties, and Operations": 0.6416,
    "Ratios, Proportions, and Percentages": null,
    "Algebraic Thinking, Equations, and Inequalities": 0.6652
  },
  "explanation": null
}
```
