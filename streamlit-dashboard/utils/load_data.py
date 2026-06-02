import streamlit as st
import pandas as pd
import json
from pathlib import Path

# PATH CONFIGURATION
CLEAN_DATA_URL = "https://drive.google.com/uc?export=download&id=197sFEYM9A6bxYe1zgonh_gvItj4w2h1n"
EXPLODED_DATA_URL = "https://drive.google.com/uc?export=download&id=1GoYNqtlUxoTo864uwD_aYfHzAweZEQ_A"

# LOAD DATASET
@st.cache_data
def load_dataset():
    """Load main cleaned dataset."""
    df = pd.read_csv(CLEAN_DATA_URL)
    return df

# LOAD DATASET (EXPLODED)
@st.cache_data
def load_exploded_dataset():
    return pd.read_csv(EXPLODED_DATA_URL)

# STUDENT HELPERS
@st.cache_data
def get_student_list():
    df = load_dataset()
    students = (df["user_id"].dropna().unique().tolist())
    students.sort()
    return students

# STUDENT HISTORY
@st.cache_data
def get_student_history(student_id):
    df = load_dataset()
    history = (df[df["user_id"] == student_id].sort_values("order_id").reset_index(drop=True))
    return history