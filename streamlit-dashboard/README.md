# EduPredictMath Dashboard

## Bahasa Indonesia

### Deskripsi

EduPredictMath Dashboard adalah aplikasi berbasis Streamlit yang digunakan untuk melakukan eksplorasi data, monitoring performa siswa, dan demonstrasi prediksi Knowledge Tracing menggunakan model Deep Learning.

Dashboard ini merupakan frontend analitik dari proyek EduPredictMath dan tidak menjalankan model AI secara langsung. Seluruh proses inferensi dilakukan melalui AI Service berbasis FastAPI yang menyediakan endpoint prediksi.

---

## Fitur Dashboard

### 1. EDA & Model Exploration

Halaman ini digunakan untuk mengeksplorasi karakteristik dataset dan perilaku belajar siswa.

Fitur:
* Dataset Overview
* Student Behavior Analysis
* Hardest Concepts Analysis
* Easiest Concepts Analysis
* Business Insights

---

### 2. Class Student Monitoring

Halaman monitoring siswa yang mensimulasikan dashboard guru.

Fitur:
* Student Accuracy
* Learning Progression
* Risk Indicator
* Skill Performance
* Recent Learning History

---

### 3. DKT Playground

Halaman demonstrasi model Deep Knowledge Tracing.

Fitur:
* Dataset History Prediction
* Manual Quiz Simulation
* CSV Upload Prediction
* Category Mastery Visualization
* AI Explanation

---

## Arsitektur Sistem

```text
User
 │
 ▼
Streamlit Dashboard
 │
 ▼
utils/dkt_api.py
 │
 ▼
AI Service (FastAPI)
 │
 ▼
DKT Model
 │
 ▼
Prediction Result
```

---

## Struktur Folder

```text
streamlit-dashboard/
│
├── app.py
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
├── requirements.txt
└── README.md
```

---

## Menjalankan Dashboard

### 1. Install Dependency

Masuk ke folder dashboard:

```bash
cd streamlit-dashboard
```

Install dependency:

```bash
pip install -r requirements.txt
```

---

### 2. Jalankan AI Service

Sebelum menjalankan dashboard, AI Service harus aktif terlebih dahulu.

Masuk ke folder AI Service:

```bash
cd ../ai-service
```

Jalankan FastAPI:

```bash
uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

atau

```bash
python inference_api.py
```

Jika berhasil, API dapat diakses melalui:

```text
http://localhost:8000
```

---

### 3. Jalankan Dashboard

Kembali ke folder dashboard:

```bash
cd ../streamlit-dashboard
```

Jalankan Streamlit:

```bash
streamlit run app.py
```

Dashboard akan tersedia pada:

```text
http://localhost:8501
```

---

## Sumber Data

Dataset tidak disimpan dalam repository GitHub.

Data dimuat melalui Google Drive menggunakan utilitas:

```text
utils/load_data.py
```

Pastikan URL dataset yang digunakan masih aktif dan dapat diakses publik.

---

## Catatan

Jika halaman DKT Playground menampilkan:

```text
AI Service Offline
```

maka periksa:

* AI Service sudah berjalan
* URL API pada `utils/dkt_api.py` benar
* Port 8000 tidak digunakan aplikasi lain
* Model berhasil dimuat oleh AI Service

---

# English Version

## Description

EduPredictMath Dashboard is a Streamlit-based application designed for educational analytics, student monitoring, and Knowledge Tracing prediction demonstrations using Deep Learning models.

This dashboard acts as the frontend layer of the EduPredictMath project and does not execute AI models directly. All predictions are performed through a FastAPI-based AI Service.

---

## Dashboard Features

### 1. EDA & Model Exploration

Explore dataset characteristics and learning behavior.

Features:

* Dataset Overview
* Student Behavior Analysis
* Hardest Concepts Analysis
* Easiest Concepts Analysis
* Business Insights

---

### 2. Class Student Monitoring

A teacher-oriented student monitoring dashboard.

Features:

* Student Accuracy
* Learning Progression
* Risk Indicator
* Skill Performance
* Recent Learning History

---

### 3. DKT Playground

Interactive Deep Knowledge Tracing demonstration.

Features:

* Dataset History Prediction
* Manual Quiz Simulation
* CSV Upload Prediction
* Category Mastery Visualization
* AI Explanation

---

## System Architecture

```text
User
 │
 ▼
Streamlit Dashboard
 │
 ▼
utils/dkt_api.py
 │
 ▼
AI Service (FastAPI)
 │
 ▼
DKT Model
 │
 ▼
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
└── assets/
```

---

## Running the Dashboard

### 1. Install Dependencies

Navigate to the dashboard folder:

```bash
cd streamlit-dashboard
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

### 2. Start AI Service

The AI Service must be running before launching the dashboard.

Navigate to the AI Service folder:

```bash
cd ../ai-service
```

Run FastAPI:

```bash
uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

or

```bash
python inference_api.py
```

The API should be available at:

```text
http://localhost:8000
```

---

### 3. Launch Streamlit Dashboard

Return to the dashboard folder:

```bash
cd ../streamlit-dashboard
```

Run Streamlit:

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## Data Source

Datasets are not included in the GitHub repository.

Data is loaded through Google Drive using:

```text
utils/load_data.py
```

Make sure the dataset URLs remain publicly accessible.

---

## Notes

If DKT Playground displays:

```text
AI Service Offline
```

please verify:

* AI Service is running
* API URL in `utils/dkt_api.py` is correct
* Port 8000 is available
* The model has been loaded successfully by the AI Service
