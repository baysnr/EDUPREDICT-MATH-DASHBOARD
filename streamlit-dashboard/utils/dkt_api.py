import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"  # Ganti dengan URL API yang sesuai
TIMEOUT = 60
DEFAULT_RESPONSE_TIME = 10000

def health_check():
    """Memastikan endpoint API aktif sebelum melakukan request berat."""
    try:
        response = requests.get(f"{API_URL}/docs", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def build_student_history(df):
    """Mentransformasi DataFrame siswa ke format payload JSON untuk API."""
    history = []
    for _, row in df.iterrows():
        history.append({
                "skill_id": str(row["skill_id"]),
                "correctness": int(row["correct"]),
                "ms_first_response": int(row["ms_first_response"]),
                "question": None
            })
    return history

def predict(student_history, personal_preference="education"):
    """Mengirim data ke model Deep Knowledge Tracing (DKT) untuk prediksi."""
    payload = {
        "student_history": student_history,
        "personal_preference": personal_preference
    }
    response = requests.post( f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()

def predict_from_student_history(history_df, preference="education"):
    student_history = (build_student_history(history_df))
    return predict(student_history,preference)

def predict_from_manual_sequence(sequence, skill_id="1", preference="education"):
    history = []
    for result in sequence:
        history.append({
                "skill_id": skill_id,
                "correctness": int(result),
                "ms_first_response": 10000,
                "question": None
            })
    return predict(history, preference)

