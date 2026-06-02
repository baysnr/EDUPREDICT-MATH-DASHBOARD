import streamlit as st
from utils.load_data import (load_dataset, load_exploded_dataset)
from utils.metrics import (get_dataset_overview, get_concept_performance)
from utils.charts import (
    plot_correct_distribution,
    plot_sequence_length_distribution,
    plot_hint_distribution,
    plot_response_time_distribution,
    plot_hardest_skills,
    plot_easiest_skills
)

# PAGE CONFIG
st.set_page_config(
    page_title="EDA & Model Exploration",
    page_icon="📊",
    layout="wide"
)

# LOAD DATA
df = load_dataset()
overview = get_dataset_overview(df)
exploded_df = load_exploded_dataset()
skill_perf = get_concept_performance(exploded_df)

# HEADER
st.title("📊 EDA & Model Exploration")
st.markdown(
    "Explore dataset characteristics, student behavior, and learning patterns" 
    "that motivate the use of Knowledge Tracing models.")

st.divider()

#MAIN TAB
tab1, tab2 = st.tabs(["📊 Dataset Exploration","🤖 Model Exploration"])
with tab1:
    st.markdown(
        "In this section, we analyze the dataset to understand"
        "student interactions, skill performance, and learning patterns.")
    
# DATASET OVERVIEW
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Students", f"{overview['total_students']:,}")
    with col2:
        st.metric("Interactions", f"{overview['total_interactions']:,}")
    with col3:
        st.metric("Skills", f"{overview['total_skills']:,}")
    with col4:
        st.metric("Avg Sequence", overview["avg_sequence_length"])

    st.divider()

    # STUDENT BEHAVIOR ANALYSIS
    st.subheader("Student Behavior Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_correct_distribution(df), width='stretch')
    with col2:
        st.plotly_chart(plot_sequence_length_distribution(df), width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_hint_distribution(df), width='stretch')
    with col4:
        st.plotly_chart(plot_response_time_distribution(df), width='stretch')

    st.divider()

    # CONCEPT ANALYSIS
    st.subheader("Concept Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_hardest_skills(skill_perf), width='stretch')
    with col2:
        st.plotly_chart(plot_easiest_skills(skill_perf), width='stretch')

    st.divider()

    # BUSINESS INSIGHTS
    st.subheader("Business Insights")
    st.info(
    f"""
        📌 Dataset contains **{overview['total_interactions']:,}** learning interactions from **{overview['total_students']:,} students**.
        
        📌 Students interact with **{overview['total_skills']:,} unique mathematical skills**.
        
        📌 Average learning sequence length is **{overview['avg_sequence_length']:.2f} interactions**.
        
        📌 Some concepts exhibit significantly lower accuracy, indicating potential learning bottlenecks.
        
        📌 These learning patterns motivate the need for Knowledge Tracing models that can estimate student mastery and support adaptive learning interventions.
    """
    )

#MODEL EXPLORATION
with tab2:
    st.markdown(
        "In this section, we will explore the performance of Knowledge Tracing models" 
        "and how they can be used to predict student learning outcomes."
    )

    st.subheader("🚀 Model Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Architecture", value="Transformer")
    with col2:
        st.metric(label="Vocabulary Size", value="90")
    with col3:
        st.metric(label="AUC-ROC", value="0.75")
    with col4:
        st.metric(label="Accuracy", value="72%")

    st.divider()

    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC-ROC", "0.75")
        st.progress(0.75)
    with col2:
        st.metric("Accuracy", "72%")
        st.progress(0.72)

    st.divider()

    st.subheader("🧠 Model Architecture")
    st.info(
        """
        **Causal Cross-Transformer DKT**

        The model learns sequential student interactions and estimates future mastery probabilities for mathematical concepts.

        Workflow Sequence:
        Student History → Skill Embedding → Cross Transformer Layers → Knowledge Representation → Mastery Prediction
        """
    )

    st.divider()

    st.subheader("💡 Business Impact")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 For Students & Platform")
        st.info(
            "- Early identification of at-risk students\n"
            "- Personalized learning recommendations\n"
            "- Continuous mastery monitoring"
        )
    with col2:
        st.markdown("### 👩‍🏫 For Teachers & Instructors")
        st.info(
            "- Adaptive learning support\n"
            "- Teacher intervention insights\n"
            "- Better overall learning outcomes"
        )
    st.divider()

# E. Final Model Interpretation
    st.subheader("📌 Model Interpretation")
    st.warning(
        "The model achieved an AUC-ROC of approximately 0.75 and an Accuracy of 72%. "
        "These results indicate a strong ability to distinguish between future correct "
        "and incorrect student responses, making the model suitable for mastery prediction "
        "and adaptive learning applications."
    )