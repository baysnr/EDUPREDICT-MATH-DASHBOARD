import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def apply_chart_theme(fig):
    fig.update_layout(template="plotly_white", height=450, margin=dict(l=20, r=20, t=60, b=20), title_x=0.05)
    return fig

# Correct vs wrong distribution
def plot_correct_distribution(df):
    counts = (df["correct"].value_counts().rename(index={1: "Correct", 0: "Wrong"}))
    fig = px.pie(values=counts.values, names=counts.index, title="Correct vs Wrong Distribution")
    return apply_chart_theme(fig)

# Sequence length distribution
def plot_sequence_length_distribution(df):
    sequence_lengths = (df.groupby("user_id").size().reset_index(name="sequence_length"))
    fig = px.histogram(sequence_lengths, x="sequence_length", nbins=40, title="Student Sequence Length Distribution")
    fig.update_layout(xaxis_title="Sequence Length", yaxis_title="Number of Students")
    return apply_chart_theme(fig)

# Hardest skills
def plot_hardest_skills(skill_perf, top_n=10):
    hardest = (skill_perf.sort_values("accuracy").head(top_n))
    fig = px.bar(hardest, x="accuracy", y="skill_name", orientation="h", title=f"Top {top_n} Hardest Skills")
    return apply_chart_theme(fig)

# Easiest skill
def plot_easiest_skills(skill_perf, top_n=10):
    easiest = (skill_perf.sort_values("accuracy", ascending=False).head(top_n))
    fig = px.bar(easiest, x="accuracy", y="skill_name", orientation="h", title=f"Top {top_n} Easiest Skills")
    return apply_chart_theme(fig)

# Learning progression
def plot_learning_progression(progress_df):
    fig = px.line(progress_df, x="attempt_number", y="running_accuracy", title="Student Learning Progression")
    fig.update_layout(xaxis_title="Attempt Number", yaxis_title="Running Accuracy (%)")
    return apply_chart_theme(fig)

# Student skill performance
def plot_student_skill_performance(skill_df, top_n=10):
    top_skills = (skill_df.head(top_n))
    fig = px.bar(top_skills, x="accuracy", y="skill_name", orientation="h", title="Student Skill Performance")
    return apply_chart_theme(fig)

# Recent performance
def plot_recent_performance(history_df):
    recent = history_df.tail(20).copy()
    recent["Result"] = recent["correct"].map({1: "Correct",0: "Wrong"})
    fig = px.scatter(recent, x="order_id", y="correct", color="Result", title="Recent Learning Outcomes")
    fig.update_layout(yaxis=dict(tickvals=[0, 1], ticktext=["Wrong", "Correct"]))
    return apply_chart_theme(fig)

# Hint usage distribution
def plot_hint_distribution(df):
    fig = px.histogram(df, x="hint_count", nbins=20, title="Hint Usage Distribution")
    return apply_chart_theme(fig)

# Response time distribution
def plot_response_time_distribution(df):
    temp = df.copy()
    temp["response_seconds"] = (temp["ms_first_response"] / 1000)
    temp = temp[(temp["response_seconds"] >= 0) & (temp["response_seconds"] <= 300)]
    fig = px.histogram(temp, x="response_seconds", nbins=40, title="Response Time Distribution")
    fig.update_layout(xaxis_title="Seconds")
    return apply_chart_theme(fig)

# Mastery chart
def create_mastery_chart(df):
    return px.bar(df, x="Category", y="Mastery", title="Category Mastery Prediction")