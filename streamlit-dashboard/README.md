# EduPredictMath Dashboard

## Bahasa Indonesia

### Deskripsi

EduPredictMath Dashboard adalah aplikasi berbasis Streamlit yang digunakan untuk melakukan eksplorasi data, monitoring performa siswa, dan demonstrasi prediksi Knowledge Tracing menggunakan model Deep Learning.

Dashboard ini merupakan frontend analitik dari proyek EduPredictMath. Proses inferensi model dilakukan melalui API yang telah dideploy secara online menggunakan Hugging Face Spaces, sehingga pengguna tidak perlu menjalankan AI Service secara lokal.

---

## Fitur Dashboard

### 1. EDA & Model Exploration

Halaman eksplorasi data untuk memahami karakteristik dataset dan perilaku belajar siswa.

Fitur:
* Dataset Overview
* Student Behavior Analysis
* Hardest Concepts Analysis
* Easiest Concepts Analysis
* Business Insights

### 2. Class Student Monitoring

Dashboard monitoring siswa yang dirancang untuk kebutuhan guru dan pengajar.

Fitur:
* Student Accuracy
* Learning Progression
* Risk Indicator
* Skill Performance
* Recent Learning History

### 3. DKT Playground

Halaman interaktif untuk menguji model Deep Knowledge Tracing.

Fitur:
* Dataset History Prediction
* Manual Quiz Simulation
* Category Mastery Prediction
* AI Explanation

---

## Arsitektur Sistem

```text
User
  ↓
Streamlit Dashboard
  ↓
utils/dkt_api.py
  ↓
Hugging Face DKT API
  ↓
Deep Knowledge Tracing Model
  ↓
Prediction Result
```

---

## Struktur Folder

```text
streamlit-dashboard/
│
├── app.py
├── requirements.txt
│
├── pages/
│   ├── 1_EDA_Model_Exploration.py
│   ├── 2_Class_Student_Monitoring.py
│   └── 3_DKT_Playground.py
│
├── utils/
│   ├── load_data.py
│   ├── metrics.py
│   ├── charts.py
│   └── dkt_api.py
│
└── README.md
```

---

## Menjalankan Dashboard

### 1. Clone Repository

```bash
git clone <repository-url>
cd streamlit-dashboard
```

### 2. Install Dependency

```bash
pip install -r requirements.txt
```

### 3. Jalankan Dashboard

```bash
streamlit run app.py
```

Dashboard akan tersedia pada:

```text
http://localhost:8501
```

---

## Konfigurasi API

Dashboard menggunakan API publik:

```text
https://edupredictmath-edupredict-dkt-api.hf.space
```

Pengaturan endpoint terdapat pada:

```text
utils/dkt_api.py
```

Contoh:

```python
BASE_URL = "https://edupredictmath-edupredict-dkt-api.hf.space"
PREDICT_URL = f"{BASE_URL}/predict"
```

---

## Sumber Data

Dataset tidak disimpan di repository GitHub.

Data dimuat menggunakan utilitas:

```text
utils/load_data.py
```

Pastikan sumber dataset yang digunakan masih dapat diakses.

---

## Troubleshooting

### AI Service Offline

Periksa:

* URL API pada `utils/dkt_api.py`
* Endpoint Hugging Face sedang aktif
* Koneksi internet tersedia
* Health check API berhasil

### Prediction Failed

Periksa:

* Format data input
* Struktur payload yang dikirim ke API
* Log aplikasi Streamlit

---

# English Version

## Description

EduPredictMath Dashboard is a Streamlit-based application for educational analytics, student monitoring, and Deep Knowledge Tracing prediction demonstrations.

The dashboard acts as the frontend of the EduPredictMath project. Model inference is performed through a publicly deployed API hosted on Hugging Face Spaces, so users do not need to run the AI service locally.

---

## Dashboard Features

### 1. EDA & Model Exploration

Explore dataset characteristics and student learning behavior.

Features:

* Dataset Overview
* Student Behavior Analysis
* Hardest Concepts Analysis
* Easiest Concepts Analysis
* Business Insights

### 2. Class Student Monitoring

Teacher-oriented student monitoring dashboard.

Features:

* Student Accuracy
* Learning Progression
* Risk Indicator
* Skill Performance
* Recent Learning History

### 3. DKT Playground

Interactive Deep Knowledge Tracing prediction interface.

Features:

* Dataset History Prediction
* Manual Quiz Simulation
* Category Mastery Prediction
* AI Explanation

---

## System Architecture

```text
User
  ↓
Streamlit Dashboard
  ↓
utils/dkt_api.py
  ↓
Hugging Face DKT API
  ↓
Deep Knowledge Tracing Model
  ↓
Prediction Result
```

---

## Folder Structure

```text
streamlit-dashboard/
│
├── app.py
├── requirements.txt
│
├── pages/
│   ├── 1_EDA_Model_Exploration.py
│   ├── 2_Class_Student_Monitoring.py
│   └── 3_DKT_Playground.py
│
├── utils/
│   ├── load_data.py
│   ├── metrics.py
│   ├── charts.py
│   └── dkt_api.py
│
└── README.md
```

---

## Running the Dashboard

### 1. Clone Repository

```bash
git clone <repository-url>
cd streamlit-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## API Configuration

The dashboard uses a publicly deployed API:

```text
https://edupredictmath-edupredict-dkt-api.hf.space
```

The endpoint configuration is located in:

```text
utils/dkt_api.py
```

Example:

```python
BASE_URL = "https://edupredictmath-edupredict-dkt-api.hf.space"
PREDICT_URL = f"{BASE_URL}/predict"
```

---

## Data Source

Datasets are not included in the GitHub repository.

Data loading is handled through:

```text
utils/load_data.py
```

Make sure the dataset source remains accessible.

---

## Troubleshooting

### AI Service Offline

Check:
* API URL configuration
* Hugging Face Space availability
* Internet connection
* API health check status

### Prediction Failed

Check:
* Input data format
* API payload structure
* Streamlit application logs
