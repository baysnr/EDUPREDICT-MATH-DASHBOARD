import pandas as pd
import streamlit as st
from utils.load_data import load_dataset
from utils.metrics import get_dataset_overview

# PAGE CONFIG
st.set_page_config(
    page_title="EduPredictMath",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown(
    """
    <style>
    .hero-container {
        padding: 2rem 1rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1E3A8A, #2563EB);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        color: #111827;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .info-card {
        padding: 1.2rem;
        border-radius: 12px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.title("📚 EduPredictMath")
st.sidebar.info(
    """
    **AI-Powered Knowledge Tracing Dashboard**

    Navigate using the pages menu above.
    """
)

# HERO SECTION
st.markdown(
    """
    <div class="hero-container">
        <h1 style="margin-bottom:0;">EduPredictMath</h1>
        <p style="font-size:22px; margin-top:10px;">AI-Powered Knowledge Tracing Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    df = load_dataset()
    overview = get_dataset_overview(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", f"{overview['total_students']:,}")
    with col2:
        st.metric("Total Interactions", f"{overview['total_interactions']:,}")
    with col3:
        st.metric("Unique Skills", f"{overview['total_skills']:,}")
    with col4:
        st.metric("Avg Sequence Length", f"{overview['avg_sequence_length']}")
except Exception:
    # Fallback jika data gagal dimuat secara dinamis
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Students", "4,151")
    with col2: st.metric("Interactions", "251,405")
    with col3: st.metric("Skills", "136")
    with col4: st.metric("Avg Sequence", "60.56")

st.divider()

# PROJECT OVERVIEW
st.header("🎯 Project Overview")
st.markdown(
    """
    **EduPredictMath** is an educational analytics platform that utilizes
    **Deep Knowledge Tracing** (DKT) models to estimate student mastery levels
    based on their historical learning interactions.

    The objective is to support *adaptive learning* by identifying learning
    patterns, monitoring student progress, and predicting future performance.
    """
)

st.divider()

# SYSTEM ARCHITECTURE
col_arch, col_flow = st.columns(2)

with col_arch:
    st.subheader("🏗️ System Architecture")
    st.code(
        "User\n"
        " │\n"
        " ▼\n"
        "Streamlit Dashboard\n"
        " │\n"
        " ▼\n"
        "FastAPI AI Service\n"
        " │\n"
        " ▼\n"
        "DKT Model\n"
        " │\n"
        " ▼\n"
        "Mastery Prediction",
        language="text"
    )

with col_flow:
    st.subheader("🔄 Learning Analytics Workflow")
    st.code(
        "Learning Interaction Data\n"
        "      ↓\n"
        "Exploratory Data Analysis (EDA)\n"
        "      ↓\n"
        "Individual Student Monitoring\n"
        "      ↓\n"
        "Knowledge Tracing Prediction\n"
        "      ↓\n"
        "Adaptive Learning Intervention",
        language="text"
    )   

st.divider()

# DASHBOARD FEATURES
st.header("🚀 Dashboard Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h4>📊 EDA & Model Exploration</h4>
            Explore dataset characteristics, student behaviour, and model performance.
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div class="info-card">
            <h4>👨‍🎓 Student Monitoring</h4>
            Analyze student learning progress, accuracy trends, and skill performance.
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        """
        <div class="info-card">
            <h4>🤖 DKT Playground</h4>
            Interact with the Knowledge Tracing model and simulate mastery predictions.
        </div>
        """,
        unsafe_allow_html=True
    )
    
st.divider()

# GET STARTED
st.success(
    """
    Use the navigation menu on the left sidebar to explore:
    1. EDA & Model Exploration
    2. Class Student Monitoring
    3. DKT Playground
    """
)
