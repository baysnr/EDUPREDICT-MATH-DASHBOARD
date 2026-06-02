import streamlit as st
from utils.load_data import (load_dataset, get_student_list)
from utils.metrics import (
    get_student_metrics,
    get_student_progress,
    get_student_skill_performance,
    get_student_history
)
from utils.charts import (
    plot_learning_progression,
    plot_student_skill_performance,
    plot_recent_performance
)

st.set_page_config(
    page_title="Class Student Monitoring", 
    page_icon="🎓", 
    layout="wide"
)
df = load_dataset()
student_list = get_student_list()

st.title("🎓 Class Student Monitoring")
st.markdown(
    "Monitor individual student performance, learning progression, "
    "and skill mastery patterns to deliver timely educational interventions."
)

st.divider()

# STUDENT SELECTOR & METRICS SUMMARY
selected_student = st.selectbox("Select Student ID", student_list)
metrics = get_student_metrics(df, selected_student)    

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{metrics['accuracy']}%")
with col2:
    st.metric("Attempts", metrics["total_attempts"])
with col3:
    st.metric("Correct", metrics["correct_answers"])
with col4:
    st.metric("Wrong", metrics["wrong_answers"])

col5, col6 = st.columns(2)
with col5:
    st.metric("Avg Response Time (s)", metrics["avg_response_time"])
with col6:
    st.metric("Avg Hints Used", metrics["avg_hints"])

# Visualisasi Status Risiko Penguasaan Materi (Mastery Status)
st.markdown("### 🔔 Current Learning Status")
accuracy = metrics["accuracy"]
if accuracy >= 80:
    st.success("🟢 **High Mastery:** Student shows excellent comprehension across mathematical concepts.")
elif accuracy >= 50:
    st.warning("🟡 **Moderate Risk:** Student struggles with specific sub-concepts. Requires casual monitoring.")
else:
    st.error("🔴 **At Risk:** High error rates detected. Direct instructor intervention is highly recommended.")

st.divider()

# LEARNING ANALYTICS VISUALIZATION (CHARTS)
st.subheader("📈 Learning Progression")
progress_df = get_student_progress(df, selected_student)
st.plotly_chart(plot_learning_progression(progress_df), width='stretch')

st.divider()

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("🏆 Skill Mastery Performance")
    skill_perf = get_student_skill_performance(df, selected_student)
    st.plotly_chart(plot_student_skill_performance(skill_perf), width='stretch')

with chart_col2:
    st.subheader("📝 Recent Learning Outcomes (Last 20)")
    history = get_student_history(df, selected_student)
    st.plotly_chart(plot_recent_performance(history), width='stretch')

st.divider()

# DATA LOG HISTORY TABLE
st.subheader("📋 Recent Interaction History")
display_cols = ["order_id", "skill_name", "correct", "hint_count", "attempt_count"]
st.dataframe(history[display_cols].tail(20), width='stretch')


