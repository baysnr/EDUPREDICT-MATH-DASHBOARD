import pandas as pd

# DATASET OVERVIEW 
def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Summary metrics untuk halaman EDA."""
    total_students = df["user_id"].nunique()
    total_interactions = len(df)
    total_skills = df["skill_id"].nunique()
    avg_sequence_length = (df.groupby("user_id").size().mean())

    return {"total_students": total_students,
        "total_interactions": total_interactions,
        "total_skills": total_skills,
        "avg_sequence_length": float(round(avg_sequence_length, 2))}

# STUDENT METRICS   
def get_student_metrics(df: pd.DataFrame, student_id: int) -> dict:
    """Statistik dasar per siswa."""
    student_df = df[df["user_id"] == student_id]
    total_attempts = len(student_df)
    
    if total_attempts == 0:
        return {
            "student_id": student_id, "total_attempts": 0, "correct_answers": 0,
            "wrong_answers": 0, "accuracy": 0.0, "avg_response_time": 0.0, "avg_hints": 0.0
        }
    
    correct_answers = int(student_df["correct"].sum())
    wrong_answers = total_attempts - correct_answers
    accuracy = (correct_answers / total_attempts)
    avg_response_time = (student_df["ms_first_response"].mean() / 1000)
    avg_hints = student_df["hint_count"].mean()

    return {
        "student_id": student_id,
        "total_attempts": total_attempts,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "accuracy": round(accuracy * 100, 2),
        "avg_response_time": float(round(avg_response_time, 1)),
        "avg_hints": float(round(avg_hints, 2))
        }

# STUDENT HISTORY V
def get_student_history(df: pd.DataFrame, student_id: int) -> pd.DataFrame:
    """Ambil seluruh history siswa."""
    return (df[df["user_id"] == student_id].sort_values("order_id").reset_index(drop=True))

# RUNNING ACCURACY V
def get_student_progress(df: pd.DataFrame, student_id: int) -> pd.DataFrame:
    """Digunakan untuk grafik learning progression."""
    history = get_student_history(df, student_id)

    history["attempt_number"] = range(1, len(history) + 1)
    history["running_correct"] = (history["correct"].cumsum())
    history["running_accuracy"] = (history["running_correct"]/ history["attempt_number"]) * 100
    return history

# SKILL PERFORMANCE V
def get_student_skill_performance(df: pd.DataFrame, student_id: int) -> pd.DataFrame:
    """Accuracy per skill."""
    history = get_student_history(df, student_id)
    skill_perf = (history.groupby("skill_name").agg(attempts=("correct", "count"), accuracy=("correct", "mean")).reset_index())
    
    skill_perf["accuracy"] *= 100
    return (skill_perf.sort_values("accuracy", ascending=False))

# CONCEPT PERFORMANCE (EXPLODED)
def get_concept_performance(exploded_df: pd.DataFrame) -> pd.DataFrame: 
    """Menganalisis tingkat kesulitan materi secara agregat dari data exploded."""   
    skill_perf = (exploded_df.groupby("skill_name").agg(attempts=("correct", "count"), accuracy=("correct", "mean")).reset_index())
    
    skill_perf["accuracy"] *= 100
    return skill_perf.sort_values("accuracy",ascending=False)
