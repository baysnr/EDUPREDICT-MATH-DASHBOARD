import streamlit as st
import pandas as pd
from utils.load_data import (load_dataset, get_student_list, get_student_history)
from utils.charts import create_mastery_chart
from utils.dkt_api import (
    predict_from_student_history,
    predict_from_manual_sequence,
    health_check
)

# PAGE CONFIG
st.set_page_config(
    page_title="DKT Playground",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DKT Playground")
st.markdown(
    "Interact directly with the AI Knowledge Tracing engine. "
    "Simulate custom user behaviors or pull direct historical patterns to view predicted outcomes."
)

st.divider()

# LOAD DATA
df = load_dataset()
students = get_student_list()

# HELPER FUNCTIONS
def display_prediction_result(response):
    """Memproses dan menampilkan visualisasi hasil kembalian dari endpoint API DKT."""
    mastery = response.get("category_mastery", {})
    mastery = { k: v for k, v in mastery.items() if v is not None}

    if not mastery:
        st.warning("No mastery prediction returned.")
        return
    
    mastery_df = pd.DataFrame({"Category": mastery.keys(), "Mastery": mastery.values()})
    mastery_df["Mastery"] *= 100

    # Section 1: Dataframe & Chart Presentation
    st.subheader("📊 Category Mastery")
    st.dataframe(mastery_df, width='stretch')
    fig = create_mastery_chart(mastery_df)
    st.plotly_chart(fig, width='stretch')

    # Section 2: Key Evaluation Badges
    best_skill = mastery_df.loc[mastery_df["Mastery"].idxmax()]
    worst_skill = mastery_df.loc[mastery_df["Mastery"].idxmin()]

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 Strongest: {best_skill['Category']} ({best_skill['Mastery']:.1f}%)")
    with col2:
        st.warning(f"⚠️ Needs Improvement: {worst_skill['Category']} ({worst_skill['Mastery']:.1f}%)")

    # Section 3: Natural Language Model Explanation
    explanation = response.get("explanation")
    st.subheader("🧠 AI Explanation")
    if explanation:
        st.info(explanation)
    else:
        st.warning("No AI explanation available for this prediction.")

# MODE SELECTOR
mode = st.radio("Prediction Mode", ["Dataset History", "Manual Quiz"], horizontal=True)

# MODE A: EVALUASI BERDASARKAN SELEKSI HISTORI SISWA DATASET
if mode == "Dataset History":
    st.subheader("📚 Dataset History Prediction")
    selected_student = st.selectbox("Select Student", students)
    history = get_student_history(selected_student)
    st.write(f"History Length: {len(history)}")

    st.markdown("### Last 10 Student Interactions Log")
    st.dataframe(history.tail(10),width='stretch')

    if st.button("Run Prediction", key="dataset_predict"):
        try:
            with st.spinner("Running DKT prediction..."):
                response = (predict_from_student_history(history))
            st.success("Prediction completed")
            display_prediction_result(response)
        except Exception as e:
            st.error(f"Inference execution failed: {str(e)}")

# MODE B: SIMULASI JAWABAN REAL-TIME MANUAL
elif mode == "Manual Quiz":
    st.subheader("Quick Learning Simulation")
    st.markdown("Build a custom sequence of student learning outcomes to evaluate model sensitivity in real-time.")

    if "quiz_sequence" not in st.session_state:
        st.session_state.quiz_sequence = []

    # Button Controllers Layout
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("✅ Correct"):
            st.session_state.quiz_sequence.append(1)
    with col2:
        if st.button("❌ Wrong"):
            st.session_state.quiz_sequence.append(0)
    with col3:
        if st.button("🗑 Reset"):
            st.session_state.quiz_sequence = []

    st.divider()

    # Current Timeline Stream Output
    sequence = st.session_state.quiz_sequence
    st.markdown("### Current Learning History")
    if len(sequence) == 0:
        st.info("No interactive attempts have been generated yet. Click the buttons above to test response sequences.")
    else:
        timeline = []
        for result in sequence:
            timeline.append("✅" if result == 1 else "❌")
        st.markdown(" ".join(timeline))
        st.caption(f"{len(sequence)} attempts recorded")
        st.metric("Accuracy", f"{sum(sequence)/len(sequence)*100:.1f}%")

    st.divider()

    if len(sequence) >= 3:
        if st.button("🚀 Predict Mastery", width='stretch'):
            try:
                response = predict_from_manual_sequence(sequence)
                st.success("Prediction completed")
                display_prediction_result(response)
            except Exception as e:
                st.error(f"Simulation inference error: {str(e)}")
    else:
        st.warning("⚠️ For educational statistical validity, append at least **3 simulation points** to unlock AI processing.")



